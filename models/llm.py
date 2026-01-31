import torch
import torch.nn as nn
import math
from typing import Optional
from configs.llm_config import BlueberryConfig
from models.layers import TransformerBlock, HyperTransformerBlock


class MinimalLLM(nn.Module):
    """Minimal dense LLM"""

    def __init__(self, config: BlueberryConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for i in range(config.n_layers):
            if config.use_hyper_connections:
                block = HyperTransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.max_seq_len,
                    layer_id=i,
                    rate=config.hyper_rate,
                    dynamic=config.hyper_dynamic,
                    dropout=config.dropout,
                    n_kv_heads=config.n_kv_heads,
                )
            else:
                block = TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.max_seq_len,
                    config.dropout,
                    n_kv_heads=config.n_kv_heads,
                )
            self.transformer_blocks.append(block)

        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head (tied with embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Pass through transformer blocks
        if self.config.use_hyper_connections:
            # Initialize hyper hidden matrix h: (B, L, N, D)
            # We start by repeating x along the N dimension
            # or by putting x in the first slot and zeros elsewhere
            h = torch.zeros(
                (x.size(0), x.size(1), self.config.hyper_rate, x.size(2)),
                device=x.device,
                dtype=x.dtype,
            )
            h[..., 0, :] = x
            
            for block in self.transformer_blocks:
                h = block(h)
            
            # Final output is taken from the first slot
            x = h[..., 0, :]
        else:
            for block in self.transformer_blocks:
                x = block(x)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        return logits
