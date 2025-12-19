import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import SwiGLUFeedForward

# Try to import torchtune, fallback to local implementation
try:
    from torchtune.modules import RotaryPositionalEmbeddings
    TORCHTUNE_AVAILABLE = True
except ImportError:
    TORCHTUNE_AVAILABLE = False


class FallbackRotaryPositionalEmbeddings(nn.Module):
    """Fallback RoPE implementation when torchtune is not available."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to input tensor [B, T, H, D]."""
        seq_len = x.shape[1]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        cos = self.cos_cached[:seq_len].to(x.dtype)
        sin = self.sin_cached[:seq_len].to(x.dtype)

        # Reshape for broadcasting: [1, T, 1, D]
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        # Rotate half
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)

        return x * cos + rotated * sin


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        if TORCHTUNE_AVAILABLE:
            self.rope = RotaryPositionalEmbeddings(
                dim=dim, max_seq_len=max_seq_len, base=10000
            )
        else:
            self.rope = FallbackRotaryPositionalEmbeddings(
                dim=dim, max_seq_len=max_seq_len, base=10000
            )

    def forward(self, x_BTHD: torch.Tensor):
        # x_BTHD shape: [B, T, H, D] - need to convert to [B, T, H, D] for torchtune
        # torchtune expects [batch, seq_len, num_heads, head_dim]
        # Our input is already [B, T, H, D] which matches torchtune's expectation
        return self.rope(x_BTHD)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)

        # Calculate queries, keys, and values
        Q = self.q_proj(x).reshape(
            batch_size, seq_len, self.n_heads, self.d_k
        )  # [B, T, H, D]
        K = self.k_proj(x).reshape(
            batch_size, seq_len, self.n_kv_heads, self.d_k
        )  # [B, T, KV_H, D]
        V = self.v_proj(x).reshape(
            batch_size, seq_len, self.n_kv_heads, self.d_k
        )  # [B, T, KV_H, D]

        # Apply RoPE
        Q = self.rotary(self.q_norm(Q))
        K = self.rotary(self.k_norm(K))

        # Repeat K/V for GQA if needed
        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)

        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        # attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        return self.w_o(attn_output)


class TransformerBlock(nn.Module):
    """Standard transformer block with dense feed-forward"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout, n_kv_heads)
        self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # Feed-forward
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x
