import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from .components import MixtureOfExperts, ZeroComputeMoe


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
        self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)
        self.rotary = Rotary(self.d_k, max_seq_len)
        self.dropout = dropout

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # B, T = x.size(0), x.size(1)
        # qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        # Q, K, V = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # [B, H, T, D]

        # Q = self.rotary(Q)
        # K = self.rotary(K)
        # Apply RoPE on [B, T, H, D]
        Q = self.rotary(self.q_norm(Q.transpose(1, 2))).transpose(1, 2)
        K = self.rotary(self.k_norm(K.transpose(1, 2))).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.d_model
        )
        # attn_output = attn_output.transpose(1, 2).reshape(B, T, self.d_model)
        return self.w_o(attn_output)


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        qk_rope_dim: int,
        qk_nope_dim: int,
        kv_lora_rank: int,
        v_dim: int,
        max_seq_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qk_dim = qk_rope_dim + qk_nope_dim
        self.qk_rope_dim, self.qk_nope_dim = qk_rope_dim, qk_nope_dim
        self.kv_lora_dim = kv_lora_rank
        self.v_dim = v_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.query = nn.Linear(d_model, n_heads * self.qk_dim, bias=False)
        self.compressed_kv = nn.Linear(d_model, kv_lora_rank + qk_rope_dim, bias=False)
        self.kv_norm = nn.RMSNorm(kv_lora_rank)
        self.decompressed_kv = nn.Linear(
            kv_lora_rank, n_heads * (qk_nope_dim + v_dim), bias=False
        )
        self.w_o = nn.Linear(v_dim * n_heads, d_model, bias=False)
        self.rotary = Rotary(qk_rope_dim, max_seq_len)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.size(0), x.size(1)

        # Query part of the mla
        q = self.query.forward(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.qk_dim)
        q_nope, q_rope = torch.split(q, (self.qk_nope_dim, self.qk_rope_dim), dim=-1)
        q_rope = self.rotary.forward(q_rope)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # KV part of the mla
        kv = self.compressed_kv.forward(x)
        kv, k_rope = torch.split(kv, (self.kv_lora_dim, self.qk_rope_dim), dim=-1)
        ## k rope part
        k_rope = k_rope.view(batch_size, seq_len, 1, self.qk_rope_dim)
        k_rope = self.rotary.forward(k_rope)
        ## v and k part
        kv = self.kv_norm.forward(kv)
        kv = self.decompressed_kv.forward(kv)
        kv = kv.view(batch_size, seq_len, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope, v = torch.split(kv, (self.qk_nope_dim, self.v_dim), dim=-1)
        k = torch.cat([k_nope, k_rope.expand(-1, -1, self.n_heads, -1)], dim=-1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        )
        return self.w_o.forward(attn_output)


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        use_mla: bool,
        qk_rope_dim: int | None,
        qk_nope_dim: int | None,
        kv_lora_rank: int | None,
        v_dim: int | None,
        max_seq_len: int,
        num_experts: int = 8,
        top_k: int = 2,
        use_zero_moe: bool = False,
        num_zero_experts: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Attention layer
        if use_mla:
            self.attention = MultiHeadLatentAttention(
                d_model,
                n_heads,
                qk_rope_dim,
                qk_nope_dim,
                kv_lora_rank,
                v_dim,
                max_seq_len,
                dropout,
            )
        else:
            self.attention = MultiHeadAttention(d_model, n_heads, max_seq_len, dropout)

        # MoE layer
        if use_zero_moe:
            self.feed_forward = ZeroComputeMoe(
                d_model, d_ff, num_experts, num_zero_experts, top_k, dropout
            )
        else:
            self.feed_forward = MixtureOfExperts(
                d_model, d_ff, num_experts, top_k, dropout
            )

        # Normalization layers
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)

        # MoE feed-forward
        ff_out, aux_loss = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, aux_loss
