# load the AutoTokenizer method and build the tokens for using
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Load tokenizer
def build_tokenizer(ds_config):
    tokenizer = AutoTokenizer.from_pretrained(ds_config["model_name"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # for LLaMA
    return tokenizer

# Tokenize prompt + answer pair
def format_and_tokenize(example, tokenizer, ds_config):
    prompt = f"{ds_config['Q_role']}: {example[ds_config['Q_role']]}\n{ds_config['A_role']}: {example[ds_config['A_role']]}"

    tokenized = tokenizer(
        prompt,
        max_length=ds_config["max_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    tokenized["labels"] = tokenized["input_ids"].clone()  
    return {k: v.squeeze(0) for k, v in tokenized.items()}


# PyTorch Dataset Wrapper
class BuildDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer_q, tokenizer_as, role_user, role_ass, max_length):
        self.tokenizer_q = tokenizer_q
        self.tokenizer_as = tokenizer_as
        self.raw_dataset = raw_dataset
        self.role_user = role_user
        self.role_ass = role_ass
        self.max_length = max_length

        self.processed = self.raw_dataset.map(
            self.tokenize_fn,
            remove_columns=self.raw_dataset.column_names
        )

    def tokenize_fn(self, example):
        prompt = f"{self.role_user}: {example[self.role_user]}\n{self.role_ass}: {example[self.role_ass]}"
        tokenized = self.tokenizer_q(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in tokenized.items()}

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        item = self.processed[idx]
        if isinstance(item, list):
            return item[0]
        return item