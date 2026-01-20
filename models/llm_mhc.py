"""
MinimalLLM with Manifold-Constrained Hyper-Connections (mHC)

This model replaces standard residual connections with mHC, providing:
- Expanded residual stream width (n streams instead of 1)
- Learnable feature routing across streams
- Stable training via doubly stochastic constraint on mixing matrices
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from configs.llm_config import BlueberryConfig
from models.mhc import (
    mHCTransformerBlock,
    StreamExpansion,
    StreamContraction,
)


class MinimalLLM_mHC(nn.Module):
    """
    Minimal LLM with Manifold-Constrained Hyper-Connections.
    
    Key differences from standard MinimalLLM:
    1. Maintains an n-stream residual throughout the model
    2. Uses mHCTransformerBlock instead of standard TransformerBlock
    3. Includes stream expansion at input and contraction at output
    """

    def __init__(self, config: BlueberryConfig):
        super().__init__()
        self.config = config
        
        # mHC-specific parameters
        self.expansion_rate = getattr(config, 'mhc_expansion_rate', 4)
        self.alpha_init = getattr(config, 'mhc_alpha_init', 0.01)
        self.sinkhorn_iters = getattr(config, 'mhc_sinkhorn_iters', 20)
        self.stream_init_mode = getattr(config, 'mhc_stream_init', 'replicate')
        self.stream_contract_mode = getattr(config, 'mhc_stream_contract', 'mean')

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Stream expansion: [B, T, C] -> [B, T, n, C]
        self.stream_expansion = StreamExpansion(
            config.d_model, 
            self.expansion_rate,
            self.stream_init_mode
        )

        # mHC Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            mHCTransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                expansion_rate=self.expansion_rate,
                dropout=config.dropout,
                n_kv_heads=config.n_kv_heads,
                alpha_init=self.alpha_init,
                sinkhorn_iters=self.sinkhorn_iters,
            )
            for _ in range(config.n_layers)
        ])

        # Stream contraction: [B, T, n, C] -> [B, T, C]
        self.stream_contraction = StreamContraction(
            config.d_model,
            self.expansion_rate,
            self.stream_contract_mode
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mHC residual connections.
        
        Args:
            x: Input token IDs [batch, seq_len]
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        # Token embeddings: [B, T] -> [B, T, C]
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Expand to n-stream residual: [B, T, C] -> [B, T, n, C]
        x_streams = self.stream_expansion(x)

        # Pass through mHC transformer blocks
        for block in self.transformer_blocks:
            x_streams = block(x_streams)

        # Contract back to single stream: [B, T, n, C] -> [B, T, C]
        x = self.stream_contraction(x_streams)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        return logits
    
    def get_num_params(self, include_embeddings: bool = True) -> int:
        """Calculate total number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if not include_embeddings:
            n_params -= self.token_embedding.weight.numel()
        return n_params
    
    def get_mhc_stats(self) -> dict:
        """Get statistics about mHC components for debugging."""
        stats = {
            'expansion_rate': self.expansion_rate,
            'sinkhorn_iters': self.sinkhorn_iters,
            'alpha_init': self.alpha_init,
        }
        
        # Collect alpha values from all layers
        alpha_pre_vals = []
        alpha_post_vals = []
        alpha_res_vals = []
        
        for block in self.transformer_blocks:
            alpha_pre_vals.extend([
                block.mhc_attn.alpha_pre.item(),
                block.mhc_ffn.alpha_pre.item()
            ])
            alpha_post_vals.extend([
                block.mhc_attn.alpha_post.item(),
                block.mhc_ffn.alpha_post.item()
            ])
            alpha_res_vals.extend([
                block.mhc_attn.alpha_res.item(),
                block.mhc_ffn.alpha_res.item()
            ])
        
        stats['alpha_pre_mean'] = sum(alpha_pre_vals) / len(alpha_pre_vals)
        stats['alpha_post_mean'] = sum(alpha_post_vals) / len(alpha_post_vals)
        stats['alpha_res_mean'] = sum(alpha_res_vals) / len(alpha_res_vals)
        
        return stats


class MinimalLLM_mHC_Hybrid(nn.Module):
    """
    Hybrid LLM that uses standard residuals for leading layers and mHC for the rest.
    
    This follows the paper's approach where some initial "dense" layers use
    standard residuals for stability during the initial signal flow.
    """

    def __init__(self, config: BlueberryConfig):
        super().__init__()
        self.config = config
        
        # mHC-specific parameters
        self.expansion_rate = getattr(config, 'mhc_expansion_rate', 4)
        self.alpha_init = getattr(config, 'mhc_alpha_init', 0.01)
        self.sinkhorn_iters = getattr(config, 'mhc_sinkhorn_iters', 20)
        self.leading_dense_layers = getattr(config, 'mhc_leading_dense_layers', 1)

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Leading dense layers (standard residual connections)
        from models.layers import TransformerBlock
        self.leading_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                n_kv_heads=config.n_kv_heads,
            )
            for _ in range(self.leading_dense_layers)
        ])

        # Stream expansion after leading layers
        self.stream_expansion = StreamExpansion(
            config.d_model, 
            self.expansion_rate,
            'replicate'
        )

        # mHC layers
        n_mhc_layers = config.n_layers - self.leading_dense_layers
        self.mhc_blocks = nn.ModuleList([
            mHCTransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                expansion_rate=self.expansion_rate,
                dropout=config.dropout,
                n_kv_heads=config.n_kv_heads,
                alpha_init=self.alpha_init,
                sinkhorn_iters=self.sinkhorn_iters,
            )
            for _ in range(n_mhc_layers)
        ])

        # Stream contraction
        self.stream_contraction = StreamContraction(
            config.d_model,
            self.expansion_rate,
            'mean'
        )

        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head
        self.lm_head = nn.Linear(config.vocab_size, config.d_model, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Leading dense layers with standard residuals
        for block in self.leading_blocks:
            x = block(x)

        # Expand to n-stream
        x_streams = self.stream_expansion(x)

        # mHC layers
        for block in self.mhc_blocks:
            x_streams = block(x_streams)

        # Contract and output
        x = self.stream_contraction(x_streams)
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        return logits
