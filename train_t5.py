import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    AdamW,
    get_linear_schedule_with_warmup,
)
from load_data import T5Dataset
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true", default=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_n_epochs", type=int, default=15)
    parser.add_argument("--patience_epochs", type=int, default=5)
    parser.add_argument("--scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_epochs", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--model_name", type=str, default="google-t5/t5-small")
    return parser.parse_args()


def get_dataloaders(args):
    from torch.utils.data import DataLoader
    train_dset = T5Dataset("data", "train", args.model_name, args.max_len)
    dev_dset = T5Dataset("data", "dev", args.model_name, args.max_len)
    test_dset = T5Dataset("data", "test", args.model_name, args.max_len)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)
    return train_loader, dev_loader, test_loader


def initialize_model_and_optimizer(args, train_loader_len):
    print("\n🔄 Loading pretrained model:", args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        args.model_name, force_download=True
    )
    sample_embed = model.shared.weight[0][:5].detach().cpu().numpy()
    print("✅ Sample embedding weights:", sample_embed)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = train_loader_len * args.max_n_epochs
    warmup_steps = int(total_steps * (args.num_warmup_epochs / args.max_n_epochs))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return model, optimizer, scheduler


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        labels[labels == PAD_IDX] = -100
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, tokenizer, gt_sql_path, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    preds, refs = [], []
    total_loss = 0
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        labels[labels == PAD_IDX] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            num_beams=4,
        )
        preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        decoded_labels = labels.clone()
        decoded_labels[decoded_labels == -100] = tokenizer.pad_token_id
        refs.extend(tokenizer.batch_decode(decoded_labels, skip_special_tokens=True))

    avg_loss = total_loss / len(loader)
    save_queries_and_records(preds, model_sql_path, model_record_path)
    _, _, record_f1, _ = compute_metrics(gt_sql_path, model_sql_path, gt_record_path, model_record_path)
    return avg_loss, record_f1


def main():
    args = get_args()
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)
    train_loader, dev_loader, test_loader = get_dataloaders(args)
    model, optimizer, scheduler = initialize_model_and_optimizer(args, len(train_loader))

    best_f1 = -1
    patience = 0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.max_n_epochs):
        print(f"\nEpoch {epoch+1}/{args.max_n_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Train loss: {train_loss:.4f}")

        dev_loss, dev_f1 = eval_epoch(
            model, dev_loader, tokenizer,
            "data/dev.sql", "results/t5_ft_dev.sql",
            "records/ground_truth_dev.pkl", "records/t5_ft_dev.pkl"
        )
        print(f"Dev loss: {dev_loss:.4f} | Record F1: {dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience = 0
            model.save_pretrained("checkpoints/best_t5_sql")
            tokenizer.save_pretrained("checkpoints/best_t5_sql")
            print("✅ New best model saved.")
        else:
            patience += 1
            if patience >= args.patience_epochs:
                print("Early stopping triggered.")
                break

    # Final inference on test set
    print("\nLoading best model for test inference...")
    model = T5ForConditionalGeneration.from_pretrained("checkpoints/best_t5_sql").to(DEVICE)
    model.eval()
    preds = []
    for batch in tqdm(test_loader, desc="Testing"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=256, num_beams=4)
        preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    os.makedirs("results", exist_ok=True)
    os.makedirs("records", exist_ok=True)
    save_queries_and_records(preds, "results/t5_ft_test.sql", "records/t5_ft_test.pkl")
    print("✅ Test predictions saved to results/t5_ft_test.sql and records/t5_ft_test.pkl")


if __name__ == "__main__":
    main()

