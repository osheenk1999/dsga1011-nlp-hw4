import os
from torch.utils.data import Dataset, DataLoader
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):
    def __init__(self, data_folder, split, model_name, max_len=256):
        """
        Loads .nl and .sql files and prepares tokenized input-output pairs for T5.
        """
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.max_len = max_len
        self.split = split

        nl_path = os.path.join(data_folder, f"{split}.nl")
        self.inputs = [line.strip() for line in open(nl_path) if line.strip()]

        # ✅ Only load targets if not test
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            self.targets = [line.strip() for line in open(sql_path) if line.strip()]
            assert len(self.inputs) == len(self.targets), f"{split}: mismatched NL and SQL lines"
        else:
            self.targets = None  # no SQL file for test set

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = f"translate text to SQL: {self.inputs[idx]}"
        source = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = source["input_ids"].squeeze()
        attention_mask = source["attention_mask"].squeeze()

        # ✅ For test set, return only encoder input
        if self.split == "test":
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        target_text = self.targets[idx]
        target = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ✅ Make sure this function is completely outside the class (no indentation)
def load_t5_data(batch_size=16, test_batch_size=16):
    """
    Returns train, dev, and test DataLoaders for T5 fine-tuning.
    """
    model_name = "google-t5/t5-small"
    max_len = 256

    train_dset = T5Dataset("data", "train", model_name, max_len)
    dev_dset = T5Dataset("data", "dev", model_name, max_len)
    test_dset = T5Dataset("data", "test", model_name, max_len)

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dset, batch_size=test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dset, batch_size=test_batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader

