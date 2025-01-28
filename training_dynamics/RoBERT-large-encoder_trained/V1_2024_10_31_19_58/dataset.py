import torch
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

from config import config

class NitroDataset(torch.utils.data.Dataset):
    def __init__(self, texts, targets, tokenizer, seq_len=config.max_length, oversample=False):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        if oversample:
            self.texts, self.targets = self.oversample(self.texts, self.targets, config.max_samples_per_class)

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

    def oversample(self, texts, targets, max_samples_per_class):
        # Encode the targets if they are not numerical
        le = LabelEncoder()
        targets_encoded = le.fit_transform(targets)

        # Calculate the number of samples for each class
        class_counts = np.bincount(targets_encoded)

        # Define the sampling strategy: only oversample classes with samples < max_samples_per_class
        sampling_strategy = {
            cls: max_samples_per_class for cls, count in enumerate(class_counts) if count < max_samples_per_class
        }

        # Perform limited oversampling only on undersampled classes
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        texts_resampled, targets_resampled = ros.fit_resample(np.array(texts).reshape(-1, 1), targets_encoded)
        
        # Decode the targets back to original form if they were encoded
        targets_resampled = le.inverse_transform(targets_resampled)

        # Flatten resampled texts back to the original format
        return texts_resampled.ravel(), targets_resampled