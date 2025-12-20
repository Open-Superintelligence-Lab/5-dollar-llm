import torch
import torch.nn as nn
import math
from typing import Optional
from configs.llm_config import BlueberryConfig
from models.layers import TransformerBlock


class MinimalLLM(nn.Module):
    """Minimal dense LLM"""

    def __init__(self, config: BlueberryConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Token Smearing gate (optional)
        self.use_token_smearing = getattr(config, 'use_token_smearing', False)
        if self.use_token_smearing:
            gate_dim = getattr(config, 'smear_gate_dim', 12)
            self.smear_gate = nn.Linear(gate_dim, 1, bias=False)
            self.smear_lambda = getattr(config, 'smear_lambda', 0.07)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.max_seq_len,
                    config.dropout,
                    n_kv_heads=config.n_kv_heads,
                )
                for i in range(config.n_layers)
            ]
        )

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

        # Token Smearing: Blend current token with previous token
        # E_smeared[t] = E[t] + λ * sigmoid(W * E[t][:gate_dim]) * E[t-1]
        if self.use_token_smearing:
            gate_dim = self.smear_gate.in_features
            gate_input = x[..., :gate_dim]  # [B, T, gate_dim]
            gate = torch.sigmoid(self.smear_gate(gate_input))  # [B, T, 1]

            # Shift embeddings for E[t-1] (pad with zeros for t=0)
            x_prev = torch.cat([
                torch.zeros_like(x[:, :1, :]),  # Zero for first position
                x[:, :-1, :]  # Previous embeddings
            ], dim=1)

            x = x + self.smear_lambda * gate * x_prev

        x = self.position_dropout(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        return logits
