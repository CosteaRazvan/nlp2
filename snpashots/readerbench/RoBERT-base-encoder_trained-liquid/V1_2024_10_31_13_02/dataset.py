import torch

from src.config import config

class NitroDataset(torch.utils.data.Dataset):
    def __init__(self, texts, targets, tokenizer, seq_len=config.max_length):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokenized = self.tokenizer(
            text,
            max_length=self.seq_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            "ids": tokenized["input_ids"].clone().detach().flatten(),
            "masks": tokenized["attention_mask"].clone().detach().flatten(),
            "target": torch.tensor(self.targets[idx], dtype=torch.float).flatten()
        }