"""
Liger Kernel optimized components for LLM training.

Liger Kernel provides Triton-based fused kernels that:
- Increase throughput by ~20%
- Reduce memory usage by ~60%
- Work seamlessly with torch.compile

Key optimizations:
- Fused RMSNorm: Avoids intermediate tensor materialization
- Fused SwiGLU: Combines gate/up projections with activation
- Fused RoPE: Applies rotary embeddings without extra memory
- Fused CrossEntropy: Critical for large vocab - eliminates logits materialization
- FusedLinearCrossEntropy: Ultimate optimization - combines lm_head + loss

References:
- https://github.com/linkedin/Liger-Kernel
- https://arxiv.org/abs/2410.10989
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Try to import Liger Kernel ops
try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.ops.rope import LigerRopeFunction
    from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
    from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    print("Warning: Liger Kernel not available, falling back to standard implementations")


class LigerRMSNorm(nn.Module):
    """
    Liger Kernel optimized RMSNorm.

    Benefits:
    - Fused kernel eliminates intermediate tensors
    - ~2x faster than nn.RMSNorm
    - Memory efficient
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if LIGER_AVAILABLE and hidden_states.is_cuda:
            # Use Liger's fused kernel
            return LigerRMSNormFunction.apply(
                hidden_states, self.weight, self.variance_epsilon
            )
        else:
            # Fallback to standard implementation
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states


class LigerSwiGLUFeedForward(nn.Module):
    """
    Liger Kernel optimized SwiGLU FeedForward.

    Benefits:
    - Fuses SiLU activation with element-wise multiply
    - Reduces memory by not materializing intermediate activations
    - ~40% memory savings on FFN layer
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        if LIGER_AVAILABLE and x.is_cuda:
            # Use Liger's fused SiLU * mul kernel
            hidden = LigerSiLUMulFunction.apply(gate, up)
        else:
            # Fallback to standard
            hidden = F.silu(gate) * up

        return self.down_proj(self.dropout(hidden))


class LigerRotaryEmbedding(nn.Module):
    """
    Liger Kernel optimized Rotary Position Embeddings (RoPE).

    Benefits:
    - In-place rotation without extra memory allocation
    - Fused cos/sin computation
    - Compatible with GQA
    """
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

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor of shape [B, T, H, D] or [B, H, T, D]
            seq_len: Optional sequence length (inferred from x if not provided)

        Returns:
            Tensor with rotary embeddings applied
        """
        if seq_len is None:
            # Assume [B, T, H, D] format
            seq_len = x.shape[1]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        cos = self.cos_cached[:seq_len].to(x.dtype)
        sin = self.sin_cached[:seq_len].to(x.dtype)

        if LIGER_AVAILABLE and x.is_cuda:
            # Use Liger's fused RoPE kernel
            # LigerRopeFunction expects [B, H, T, D] format
            if x.dim() == 4 and x.shape[1] != x.shape[2]:
                # Assume [B, T, H, D], need to transpose
                x_transposed = x.transpose(1, 2)  # [B, H, T, D]
                result = LigerRopeFunction.apply(x_transposed, cos, sin)
                return result.transpose(1, 2)  # Back to [B, T, H, D]
            return LigerRopeFunction.apply(x, cos, sin)
        else:
            # Fallback to standard RoPE
            return self._apply_rope_standard(x, cos, sin)

    def _apply_rope_standard(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Standard RoPE implementation as fallback."""
        # x shape: [B, T, H, D]
        seq_len = x.shape[1]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]

        # Rotate half
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        rotated = torch.cat((-x2, x1), dim=-1)

        return x * cos + rotated * sin


def liger_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Liger Kernel optimized cross entropy loss.

    Benefits:
    - Fused softmax + log + nll_loss
    - Memory efficient for large vocabularies
    - ~30% faster than F.cross_entropy

    Args:
        logits: Model predictions [B, T, V] or [B*T, V]
        labels: Target labels [B, T] or [B*T]
        ignore_index: Index to ignore in loss computation
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Cross entropy loss
    """
    if LIGER_AVAILABLE and logits.is_cuda:
        # Reshape if needed
        if logits.dim() == 3:
            B, T, V = logits.shape
            logits = logits.view(-1, V)
            labels = labels.view(-1)

        loss = LigerCrossEntropyFunction.apply(logits, labels, ignore_index)

        if reduction == "mean":
            # Count non-ignored tokens
            valid_mask = labels != ignore_index
            return loss.sum() / valid_mask.sum().clamp(min=1)
        elif reduction == "sum":
            return loss.sum()
        return loss
    else:
        # Fallback to standard
        return F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)


def liger_fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Liger Kernel's FusedLinearCrossEntropy - THE ULTIMATE OPTIMIZATION.

    This fuses the language modeling head (linear projection) with cross entropy loss,
    completely eliminating the need to materialize the logits tensor!

    For vocab_size=49152, batch_size=4, seq_len=2048:
    - Standard: Materializes 4*2048*49152*2 bytes = ~800MB of logits
    - Fused: ZERO logits materialization, ~80% memory savings

    Benefits:
    - Eliminates logits tensor allocation entirely
    - ~80% memory reduction on the loss computation
    - Critical for large vocabulary training
    - Enables larger batch sizes

    Args:
        hidden_states: Final hidden states [B*T, D]
        weight: LM head weight matrix [V, D]
        labels: Target labels [B*T]
        bias: Optional bias
        ignore_index: Index to ignore

    Returns:
        Tuple of (loss, None) - logits are not computed
    """
    if LIGER_AVAILABLE and hidden_states.is_cuda:
        # LigerFusedLinearCrossEntropyFunction.forward signature:
        # (_input, weight, target, bias=None, ce_weight=None, ignore_index=-100, ...)
        # Returns: (loss, z_loss, token_accuracy) - unpack the first element
        result = LigerFusedLinearCrossEntropyFunction.apply(
            hidden_states,  # _input: [B*T, H]
            weight,         # weight: [V, H]
            labels,         # target: [B*T]
            bias,           # bias: [V] or None
            None,           # ce_weight: per-class weights (None for no weighting)
            ignore_index,   # ignore_index
        )
        loss = result[0] if isinstance(result, tuple) else result
        return loss, None
    else:
        # Fallback: compute logits and loss separately
        logits = F.linear(hidden_states, weight, bias)
        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index)
        return loss, logits


class LigerLMHead(nn.Module):
    """
    Language modeling head with FusedLinearCrossEntropy support.

    This module wraps the LM head linear layer and provides an efficient
    forward pass that can use FusedLinearCrossEntropy during training.
    """
    def __init__(self, d_model: int, vocab_size: int, tie_weights: Optional[nn.Parameter] = None):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        if tie_weights is not None:
            # Weight tying with embeddings
            self.weight = tie_weights
        else:
            self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

        self.bias = None  # No bias for tied weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning logits."""
        return F.linear(hidden_states, self.weight, self.bias)

    def forward_with_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fused forward pass that computes loss directly without materializing logits.

        Args:
            hidden_states: [B, T, D] or [B*T, D]
            labels: [B, T] or [B*T]
            ignore_index: Label index to ignore

        Returns:
            (loss, logits) - logits may be None if fused kernel was used
        """
        # Flatten if needed
        if hidden_states.dim() == 3:
            B, T, D = hidden_states.shape
            hidden_states = hidden_states.view(-1, D)
            labels = labels.view(-1)

        return liger_fused_linear_cross_entropy(
            hidden_states, self.weight, labels, self.bias, ignore_index
        )


def get_liger_status() -> dict:
    """Get status of Liger Kernel availability and components."""
    return {
        "available": LIGER_AVAILABLE,
        "components": {
            "LigerRMSNorm": LIGER_AVAILABLE,
            "LigerSwiGLU": LIGER_AVAILABLE,
            "LigerRoPE": LIGER_AVAILABLE,
            "LigerCrossEntropy": LIGER_AVAILABLE,
            "LigerFusedLinearCrossEntropy": LIGER_AVAILABLE,
        },
        "expected_improvements": {
            "throughput": "+20%",
            "memory_reduction": "~60%",
            "fused_linear_ce_memory_savings": "~80% on loss computation"
        }
    }
