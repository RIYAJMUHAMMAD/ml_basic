import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DisentangledSelfAttention(nn.Module):
    """
    Implementation of Disentangled Self-Attention, inspired by DeBERTa.
    This version assumes absolute position embeddings are provided.
    """
    def __init__(self, hidden_size, num_heads, attention_dropout=0.1, max_position=512):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Content projections
        self.query_proj = nn.Linear(hidden_size, self.all_head_size)
        self.key_proj = nn.Linear(hidden_size, self.all_head_size)
        self.value_proj = nn.Linear(hidden_size, self.all_head_size)

        # Positional projections
        self.pos_key_proj = nn.Linear(hidden_size, self.all_head_size)
        self.pos_query_proj = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout)

    def transpose_for_scores(self, x):
        """
        Transposes a 3D tensor into a 4D tensor for multi-head attention.
        (batch_size, seq_len, all_head_size) -> (batch_size, num_heads, seq_len, head_size)
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, pos_emb):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input states, shape `(batch_size, seq_len, hidden_size)`.
            attention_mask (`torch.FloatTensor`):
                Mask to prevent attention to padded tokens, shape `(batch_size, 1, 1, seq_len)`.
                Values should be 0 for tokens to attend to and a large negative number for masked tokens.
            pos_emb (`torch.FloatTensor`):
                Relative positional embeddings, shape `(batch_size, seq_len, seq_len, hidden_size)`.
                pos_emb[b, i, j] represents the relative position embedding from position i to position j.
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Project content heads
        query = self.transpose_for_scores(self.query_proj(hidden_states))  # (batch, heads, seq, head_size)
        key = self.transpose_for_scores(self.key_proj(hidden_states))      # (batch, heads, seq, head_size)
        value = self.transpose_for_scores(self.value_proj(hidden_states))  # (batch, heads, seq, head_size)

        # 1. Content-to-Content (c2c): standard self-attention
        # query[i] · key[j]^T
        c2c_scores = torch.matmul(query, key.transpose(-1, -2))  # (batch, heads, seq, seq)

        # 2. Content-to-Position (c2p) and Position-to-Content (p2c)
        # Project positional embeddings
        # Reshape pos_emb for batch processing: (batch, seq, seq, hidden) -> (batch * seq * seq, hidden)
        pos_emb_flat = pos_emb.reshape(batch_size * seq_len * seq_len, hidden_size)
        
        # Project and reshape for multi-head attention
        pos_key_proj = self.pos_key_proj(pos_emb_flat)    # (batch * seq * seq, all_head_size)
        pos_query_proj = self.pos_query_proj(pos_emb_flat)  # (batch * seq * seq, all_head_size)
        
        # Reshape to (batch, seq, seq, heads, head_size) then permute to (batch, heads, seq, seq, head_size)
        pos_key = pos_key_proj.view(batch_size, seq_len, seq_len, self.num_attention_heads, self.attention_head_size)
        pos_key = pos_key.permute(0, 3, 1, 2, 4)  # (batch, heads, seq_i, seq_j, head_size)
        
        pos_query = pos_query_proj.view(batch_size, seq_len, seq_len, self.num_attention_heads, self.attention_head_size)
        pos_query = pos_query.permute(0, 3, 1, 2, 4)  # (batch, heads, seq_i, seq_j, head_size)
        
        # Content-to-Position scores: query[i] · pos_key[i,j]
        # query: (batch, heads, seq_i, head_size)
        # pos_key: (batch, heads, seq_i, seq_j, head_size)
        # result: (batch, heads, seq_i, seq_j)
        c2p_scores = torch.einsum('bhid,bhijd->bhij', query, pos_key)
        
        # Position-to-Content scores: pos_query[i,j] · key[j]
        # pos_query: (batch, heads, seq_i, seq_j, head_size)
        # key: (batch, heads, seq_j, head_size)
        # result: (batch, heads, seq_i, seq_j)
        p2c_scores = torch.einsum('bhjid,bhjd->bhij', pos_query, key)
        
        # Combine all three attention scores and scale by sqrt(head_size)
        attention_scores = (c2c_scores + c2p_scores + p2c_scores) / math.sqrt(self.attention_head_size)

        # Apply the attention mask (add large negative values to masked positions)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Weighted sum of value vectors
        context = torch.matmul(attention_probs, value)  # (batch, heads, seq, head_size)

        # Reshape context layer back to (batch_size, seq_len, all_head_size)
        context = context.permute(0, 2, 1, 3).contiguous()  # (batch, seq, heads, head_size)
        new_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_shape)  # (batch, seq, all_head_size)
        
        return context