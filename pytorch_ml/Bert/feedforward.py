import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self,):
        super().__init__()
        self.linear1 = self.Linear(d_model,d_ff)
        self.linear2 = self.Linear(d_ff,d_model)
        self.dropout = self.Dropout(dropout)
        self.gelu = nn.Gelu()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
