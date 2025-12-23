"""
Parallel Transformer Block (GPT-J / PaLM style)

Instead of sequential: x = x + attn(norm(x)); x = x + ffn(norm(x))
Uses parallel:         x = x + attn(norm(x)) + ffn(norm(x))

Benefits:
- ~15% faster training (reduced sequential dependency)
- Same parameter count
- Proven in GPT-J, PaLM, and other large models
"""

import torch
import torch.nn as nn
from .layers import MultiHeadAttention
from .components import SwiGLUFeedForward


class ParallelTransformerBlock(nn.Module):
    """
    Parallel transformer block where attention and FFN run conceptually in parallel.
    
    Architecture: x = x + Attn(norm(x)) + FFN(norm(x))
    
    This reduces the sequential dependency from:
        norm1 -> attn -> add -> norm2 -> ffn -> add
    To:
        norm -> [attn, ffn] -> add
    
    Used in: GPT-J, PaLM, Falcon, and other efficient architectures.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        
        # Single pre-norm (shared between attn and ffn)
        self.norm = nn.RMSNorm(d_model)
        
        # Attention and FFN
        self.attention = MultiHeadAttention(
            d_model, n_heads, max_seq_len, dropout, n_kv_heads
        )
        self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)
        
        # Optional dropout (usually 0 for pretraining)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Single normalization
        normed = self.norm(x)
        
        # Parallel attention + FFN
        attn_out = self.attention(normed)
        ff_out = self.feed_forward(normed)
        
        # Combine: residual + attention + feedforward
        return x + self.dropout(attn_out + ff_out)
