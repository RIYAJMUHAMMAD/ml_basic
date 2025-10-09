from .disentangled_attention import DisentangledSelfAttention
import torch
import torch.nn as nn


class DebertaLayer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.attention = DisentangledSelfAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            attention_dropout=config.attention_probs_dropout_prob,
            max_position=config.max_position_embeddings
        )
        self.attention_nn = nn.Sequential(
        nn.Linear(config.hidden_size, config.hidden_size),
        nn.Dropout(config.hidden_dropout_prob)
        )
        self.attention_layernorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

        # feed-forward network
        self.ffn = nn.Sequential(
        nn.Linear(config.hidden_size, config.intermediate_size),
        nn.GELU(),
        nn.Linear(config.intermediate_size, config.hidden_size),
        nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask, pos_emb):

        attention_output = self.attention(hidden_states, attention_mask, pos_emb)
        attention_output = self.attention_nn(attention_output)
        hidden_states = self.attention_layernorm(hidden_states + attention_output)

        layer_output = self.ffn(hidden_states)
        out = self.ffn_norm(hidden_states + layer_output)
        return out
