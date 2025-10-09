import torch
import torch.nn as nn
from .embedding import DebertaEmbedding, RelativePositionEmbedding
from .transformer import DebertaLayer

class DeBERTa(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.config = config
        self.embeddings = DebertaEmbedding(config)
        self.layers = nn.ModuleList(
            [
                DebertaLayer(config) for _ in range(config.num_hidden_layers)
            ]
        )
        self.relative_pos_emb = RelativePositionEmbedding(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):

            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)
        batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
        device = hidden_states.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        pos_emb = self.relative_pos_emb(seq_len, device, batch_size)

        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_attention_mask, pos_emb)

        encoder_output = hidden_states
        return encoder_output
