from pytorch_ml.Bert.embedding import BertEmbedding
from pytorch_ml.Bert.transformer import TransformerBlock
import torch.nn as nn


class Bert(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_len=512,
        dropout=0.1
    ):
        super().__init__()
        self.embedding = BertEmbedding(vocab_size, d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])

    def forward(self, x, segment_ids=None, mask=None):
        if mask is None:
            mask = (x != 0).unsqueeze(1).unsqueeze(2)

        x = self.embedding(x, segment_ids)
        for encoder in self.encoder_layers:
            x = encoder(x, mask)
        return x