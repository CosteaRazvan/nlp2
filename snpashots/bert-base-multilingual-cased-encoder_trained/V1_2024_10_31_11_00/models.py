import torch.nn as nn
from transformers import AutoModel

from src.config import config

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.BatchNorm1d(hidden_size*2),
            nn.Dropout(0.25),
            nn.GELU(),
            nn.Linear(hidden_size*2, 5),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        x = self.seq(x)
        return x

class Classificator(nn.Module):
    def __init__(self, encoder_name, encoder_freeze = False):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(encoder_name)
        self.transformer.training = not encoder_freeze
        self.transformer.max_length = config.max_length
        # self.transformer._requires_grad = True

        hidden_size = self.transformer.config.hidden_size

        self.classifier = MLP(hidden_size)

        # Set requires_grad for transformer parameters
        for param in self.transformer.parameters():
            param.requires_grad = not encoder_freeze

   
    def forward(self, input_ids, attention_mask):
        raw_output = self.transformer(input_ids, attention_mask, return_dict=True)
        x = raw_output["pooler_output"]
        out = self.classifier(x)
        return out