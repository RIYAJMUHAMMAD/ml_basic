from .multihead_attn import MultiHeadAttention
from .feedforward import FeedForward
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = self.MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = self.FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):

        attn_output = self.attention(x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x+ attn_output)

        feed_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(feed_out))
        return x
