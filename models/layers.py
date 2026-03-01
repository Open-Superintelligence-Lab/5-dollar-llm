import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .components import SquaredReLUFeedForward


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.rope = RotaryPositionalEmbeddings(
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
        
        # ============ MERGED QKVO PROJECTION ============
        # Instead of 4 separate Linear layers, use single merged projection
        q_size = d_model
        kv_size = self.n_kv_heads * self.d_k
        o_size = d_model
        
        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_size = q_size + 2 * kv_size  # Q + K + V sizes
        
        # Single parameter tensor for all projections
        # Shape: [Q_size + K_size + V_size + O_size, d_model]
        self.qkvo_proj = nn.Parameter(
            torch.empty(q_size + 2 * kv_size + o_size, d_model)
        )
        
        # Initialize all weights with std=0.02
        with torch.no_grad():
            torch.nn.init.normal_(self.qkvo_proj, mean=0.0, std=0.02)
        # ================================================
        
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)
        
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        
        # ============ MERGED QKV PROJECTION ============
        # Single matmul instead of 3 separate projections
        qkv = F.linear(x, self.qkvo_proj[:self.qkv_size])
        
        # Split the result into Q, K, V
        Q, K, V = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # ================================================
        
        # Reshape to multi-head format
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_kv_heads, self.d_k)
        
        # Apply RoPE
        Q = self.rotary(self.q_norm(Q))
        K = self.rotary(self.k_norm(K))
        
        # Repeat K/V for GQA if needed
        if self.n_kv_heads != self.n_heads:
            K = torch.repeat_interleave(K, self.num_key_value_groups, dim=2)
            V = torch.repeat_interleave(V, self.num_key_value_groups, dim=2)
        
        # Transpose for attention
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        
        # Compute attention
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        
        # ============ MERGED O PROJECTION ============
        # Use the last part of qkvo_proj for output projection
        return F.linear(attn_output, self.qkvo_proj[self.qkv_size:])


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
        self.feed_forward = SquaredReLUFeedForward(d_model, d_ff, dropout)

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


class HyperConnection(nn.Module):
    def __init__(self, dim, rate, layer_id, dynamic, device=None):
        super(HyperConnection, self).__init__()
        self.rate = rate
        self.layer_id = layer_id
        self.dynamic = dynamic
        self.static_beta = nn.Parameter(torch.ones((rate,), device=device))
        
        init_alpha0 = torch.zeros((rate, 1), device=device)
        init_alpha0[layer_id % rate, 0] = 1.
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye((rate), device=device)], dim=1))
        
        if self.dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim, rate + 1), device=device))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim,), device=device))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
        
        self.layer_norm = nn.LayerNorm(dim)

    def width_connection(self, h):
        # h: (B, L, N, D)
        if self.dynamic:
            norm_h = self.layer_norm(h)
            
            wc_weight = norm_h @ self.dynamic_alpha_fn
            wc_weight = torch.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, ...]
            
            dc_weight = norm_h @ self.dynamic_beta_fn
            dc_weight = torch.tanh(dc_weight)
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta[None, None, ...]
        else:
            alpha = self.static_alpha[None, None, ...]
            beta = self.static_beta[None, None, ...]
            
        # width connection
        mix_h = alpha.transpose(-1, -2) @ h
        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        # h_o: output of sub-layer (B, L, D)
        # beta: (B, L, N)
        # mix_h: (B, L, N+1, D)
        h = torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :]
        return h


class HyperTransformerBlock(nn.Module):
    """Transformer block with Hyper-connections"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        layer_id: int,
        rate: int,
        dynamic: bool,
        dropout: float = 0.1,
        n_kv_heads: int | None = None,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout, n_kv_heads)
        self.feed_forward = SquaredReLUFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.hc1 = HyperConnection(d_model, rate, layer_id * 2, dynamic)
        self.hc2 = HyperConnection(d_model, rate, layer_id * 2 + 1, dynamic)

    def forward(self, h):
        # h shape: (B, L, N, D)
        
        # --- Sub-layer 1: Attention ---
        mix_h, beta = self.hc1.width_connection(h)
        # Input to sub-layer is typically h[..., 0, :]
        x = mix_h[..., 0, :]
        attn_out = self.attention(self.norm1(x))
        h = self.hc1.depth_connection(mix_h, self.dropout(attn_out), beta)
        
        # --- Sub-layer 2: Feed-forward ---
        mix_h, beta = self.hc2.width_connection(h)
        x = mix_h[..., 0, :]
        ff_out = self.feed_forward(self.norm2(x))
        h = self.hc2.depth_connection(mix_h, self.dropout(ff_out), beta)
        
        return h
