import torch
import torch.nn as nn

class DebertaEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx =0)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.embedding_drop)

    def forward(self, input_ids):
        embeddings = self.word_embedding(input_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RelativePositionEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.max_relative_positions = config.max_position_embeddings
        self.embeddings = nn.Embedding(
            2* config.max_position_embeddings +1,
            config.hidden_size
        )

    def forward(self, seq_len, device, batch_size=None):

        # Create position indices [0,1,2,.....,seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long, device=device)
        
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        # clip to [-max_pos, + max_pos]
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_positions,
            self.max_relative_positions
        )

        relative_positions = relative_positions + self.max_relative_positions
        pos_embeddings = self.embeddings(relative_positions)  # (seq_len, seq_len, hidden_size)
        
        # For disentangled attention, we need to return the full relative position matrix
        # The attention mechanism will handle the proper indexing
        
        # If batch_size is provided, expand to (batch_size, seq_len, seq_len, hidden_size)
        if batch_size is not None:
            pos_embeddings = pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        return pos_embeddings

