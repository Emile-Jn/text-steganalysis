"""
This module defines a PyTorch Dataset for loading text data and labels from a TSV file.
Each row in the TSV file should contain a 'text' column and a 'label' column.
"""

import pandas as pd
from torch.utils.data import Dataset

class TSVTextDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_len=128):
        self.df = pd.read_csv(tsv_path, sep="\t")
        self.texts = self.df["text"].tolist()
        self.labels = self.df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": label
        }