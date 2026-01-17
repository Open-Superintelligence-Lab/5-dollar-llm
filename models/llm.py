import torch
import torch.nn as nn
import math
from typing import Optional
from configs.llm_config import BlueberryConfig
from models.layers import TransformerBlock


class MinimalLLM(nn.Module):
    """Minimal dense LLM"""

    # Updated init signature to accept tokenizer
    def __init__(self, config: BlueberryConfig, tokenizer=None):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)
        
        # Engram Setup
        # Check if we should enable Engram
        engram_layers_set = set(config.engram_layers) if hasattr(config, "engram_layers") else set()
        
        if engram_layers_set and tokenizer is None:
             print("Warning: Engram layers configured but no tokenizer provided. Engram disabled.")
             engram_layers_set = set() # Disable if no tokenizer
        elif engram_layers_set:
             print(f"Engram V2 enabled on layers: {engram_layers_set}")


        blocks = []
        for i in range(config.n_layers):
            # Check if this layer gets an Engram Module
            engram_mod = None
            if i in engram_layers_set and tokenizer is not None:
                from models.engram_v2 import EngramModuleV2
                # Instantiating V2 module
                engram_mod = EngramModuleV2(config, tokenizer, layer_id=i)

            blocks.append(
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    config.max_seq_len,
                    config.dropout,
                    n_kv_heads=config.n_kv_heads,
                    engram_module=engram_mod
                )
            )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(blocks)

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

    def forward(self, input_ids):
        # Token embeddings
        # input_ids serves as the "x" for embeddings, and "input_ids" for Engram
        x = self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, input_ids=input_ids)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        return logits
