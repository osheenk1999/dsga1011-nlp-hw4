import os
from torch.utils.data import Dataset
from transformers import T5TokenizerFast
import torch


PAD_IDX = 0

class T5Dataset(Dataset):
    def __init__(self, data_folder, split, model_name, max_len=256):
        """
        Loads .nl and .sql files and prepares tokenized input-output pairs for T5.

        Args:
            data_folder: directory containing data files (e.g., 'data')
            split: one of {'train', 'dev', 'test'}
            model_name: Hugging Face model name (e.g., 'google-t5/t5-small')
            max_len: maximum token length for both encoder and decoder
        """
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.max_len = max_len

        nl_path = os.path.join(data_folder, f"{split}.nl")
        sql_path = os.path.join(data_folder, f"{split}.sql")

        with open(nl_path, "r") as f:
            self.inputs = [line.strip() for line in f if line.strip()]

        with open(sql_path, "r") as f:
            self.targets = [line.strip() for line in f if line.strip()]

        assert len(self.inputs) == len(self.targets), f"{split}: mismatched NL and SQL lines"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        # Prefix helps T5 focus on task (optional but good practice)
        input_text = f"translate text to SQL: {input_text}"

        # Tokenize
        source = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        target = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Flatten tensors and replace pad tokens with -100 in labels
        input_ids = source["input_ids"].squeeze()
        attention_mask = source["attention_mask"].squeeze()
        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
