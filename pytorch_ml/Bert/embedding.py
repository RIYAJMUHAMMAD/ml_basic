import torch
import torch.nn as nn

class BertEmbedding(nn.Module):

    def __init__(self,vocab_size,d_model,max_len=512,dropout=0.1):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(max_len,d_model)
        self.segment_emebed = nn.Embedding(2, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, segment_ids = None):

        d = x.size(1)
        token_embed = self.token_embed(x)
        pos = torch.arange(d, device=x.device).unsqueeze(0).expand_as(x)
        pos_embed = self.position_embed(pos)

        embed = pos_embed + token_embed

        if segment_ids is not None:
            embed += self.segment_embed(segment_ids)

        embed = self.dropout(embed)
        embed = self.norm(embed)

        return embed


