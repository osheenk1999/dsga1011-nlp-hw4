import os
import argparse
from tqdm import tqdm
import torch
import wandb
from transformers import T5ForConditionalGeneration

from t5_utils import (
    initialize_model,
    initialize_optimizer_and_scheduler,
    save_model,
    load_model_from_checkpoint,
    setup_wandb,
)
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD_IDX = 0


# ============================
#   Argument parser
# ============================
def get_args():
    parser = argparse.ArgumentParser(description="T5 training loop")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--scheduler_type", type=str, default="linear")
    parser.add_argument("--num_warmup_epochs", type=int, default=1)
    parser.add_argument("--max_n_epochs", type=int, default=15)
    parser.add_argument("--patience_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


# ============================
#   Training loop
# ============================
def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_loss = float("inf")
    patience = 0
    model_type = "ft" if args.finetune else "scr"
    checkpoint_dir = os.path.join("checkpoints", f"{model_type}_experiments", args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("records", exist_ok=True)

    for epoch in range(args.max_n_epochs):
        print(f"\nEpoch {epoch+1}/{args.max_n_epochs}")
        train_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Train loss: {train_loss:.4f}")

        dev_loss, record_f1 = eval_epoch(
            args,
            model,
            dev_loader,
            "data/dev.sql",
            "results/t5_ft_dev.sql",
            "records/ground_truth_dev.pkl",
            "records/t5_ft_dev.pkl",
        )
        print(f"Dev loss: {dev_loss:.4f} | Record F1: {record_f1:.4f}")

        if dev_loss < best_loss:
            best_loss = dev_loss
            patience = 0
            save_model(checkpoint_dir, model, best=True)
            print("✅ New best model saved.")
        else:
            patience += 1
            if patience >= args.patience_epochs:
                print("⏹ Early stopping triggered.")
                break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()

        encoder_input = batch["input_ids"].to(DEVICE)
        encoder_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=encoder_input, attention_mask=encoder_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


@torch.no_grad()
def eval_epoch(args, model, loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    preds, refs = [], []
    total_loss = 0
    tokenizer = model.tokenizer

    for batch in tqdm(loader, desc="Evaluating"):
        encoder_input = batch["input_ids"].to(DEVICE)
        encoder_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=encoder_input, attention_mask=encoder_mask, labels=labels)
        total_loss += outputs.loss.item()

        generated = model.generate(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
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


@torch.no_grad()
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    tokenizer = model.tokenizer
    preds = []

    for batch in tqdm(test_loader, desc="Testing"):
        encoder_input = batch["input_ids"].to(DEVICE)
        encoder_mask = batch["attention_mask"].to(DEVICE)
        generated = model.generate(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            max_length=256,
            num_beams=4,
        )
        preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    save_queries_and_records(preds, model_sql_path, model_record_path)
    print(f"✅ Test predictions saved to {model_sql_path} and {model_record_path}")


# ============================
#   Main
# ============================
def main():
    args = get_args()
    if args.use_wandb:
        setup_wandb(args)

    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    print("\n========== Training ==========")
    model, tokenizer = initialize_model(args)
    model.tokenizer = tokenizer

    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    print("\n========== Test Inference ==========")
    model, tokenizer = load_model_from_checkpoint(args, best=True)
    model.tokenizer = tokenizer
    test_inference(args, model, test_loader, "results/t5_ft_test.sql", "records/t5_ft_test.pkl")

    print("✅ All done!")


if __name__ == "__main__":
    main()

