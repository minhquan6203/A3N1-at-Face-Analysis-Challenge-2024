import torch
from torch import nn
from typing import List, Dict, Optional,Any
import numpy as np
from torch.nn import functional as F
import math
from .attention import MultiHeadAtt

class PositionWiseFeedForward(nn.Module):
    def __init__(self, config) -> None:
        super(PositionWiseFeedForward, self).__init__()

        d_model = config['attention']['d_model']
        d_ff = config['attention']['d_ff']
        dropout = config['attention']['dropout']

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input) -> torch.Tensor:
        out = self.dropout_1(F.gelu(self.fc1(input)))
        out = self.dropout_2(self.fc2(out))
        out = self.layer_norm(input + out)

        return out

