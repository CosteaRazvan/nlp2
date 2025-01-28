import torch
import torch.nn as nn
from transformers import AutoModel

from src.config import config

class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.fc(out[:, -1, :])
        return out

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell = nn.RNNCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_t = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        for t in range(x.size(1)):
            h_t = self.rnn_cell(x[:, t, :], h_t)
        out = self.fc(h_t)
        return out

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
    def __init__(self, encoder_name, encoder_freeze = False, liquid=False, avg=False):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(encoder_name)
        self.transformer.training = not encoder_freeze
        self.transformer.max_length = config.max_length
        # self.transformer._requires_grad = True

        hidden_size = self.transformer.config.hidden_size

        self.liquid = liquid
        self.avg = avg

        if liquid:
            self.classifier = LiquidNeuralNetwork(hidden_size, hidden_size*2, 5)
        else:
            self.classifier = MLP(hidden_size)

        # Set requires_grad for transformer parameters
        for param in self.transformer.parameters():
            param.requires_grad = not encoder_freeze

   
    def forward(self, input_ids, attention_mask):
        raw_output = self.transformer(input_ids, attention_mask, return_dict=True)
        if self.liquid:
            x = raw_output["last_hidden_state"]
        elif self.avg:
            x = raw_output["last_hidden_state"].mean(dim=1)
        else:
            x = raw_output["pooler_output"]
        out = self.classifier(x)
        return out