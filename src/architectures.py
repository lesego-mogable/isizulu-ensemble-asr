import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair

# --- 1. Custom CNN-RNN-CTC Model (Agent 3) ---
class ASRModel(nn.Module):
    def __init__(self, n_mels, rnn_dim, vocab_size, n_cnn_layers, n_rnn_layers, dropout):
        super().__init__()
        cnn_channels = 32
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.BatchNorm2d(cnn_channels),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1, stride=1), nn.ReLU(),
            nn.BatchNorm2d(cnn_channels),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )
        cnn_output_dim = (n_mels // 2) * cnn_channels
        self.rnn = nn.GRU(
            input_size=cnn_output_dim, hidden_size=rnn_dim,
            num_layers=n_rnn_layers, bidirectional=True,
            batch_first=True, dropout=dropout if n_rnn_layers > 1 else 0
        )
        self.classifier = nn.Linear(rnn_dim * 2, vocab_size)

    def forward(self, input_values, **kwargs):
        x = input_values.unsqueeze(1)
        x = self.cnn(x)
        batch_size, channels, freq_dim, seq_len = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, seq_len, -1)
        x, _ = self.rnn(x)
        logits = self.classifier(x)
        return {"logits": logits}

# --- 2. Fusion Model (GRU Ensemble) ---
class FusionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=2, bidirectional=True, batch_first=True, dropout=0.2
        )
        self.classifier = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)
        logits = self.classifier(x)
        return logits