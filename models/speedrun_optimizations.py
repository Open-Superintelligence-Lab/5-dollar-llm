"""
Speedrun optimizations from modded-nanogpt.

Implements cutting-edge techniques from the LLM training speedrun community:
- Gradient Checkpointing: Trade compute for memory (~50% memory reduction)
- ReLU² Activation: Faster convergence than SwiGLU for short runs
- Untied Embeddings: More flexible output distribution learning

References:
- https://github.com/KellerJordan/modded-nanogpt
- https://arxiv.org/abs/2410.10989 (Liger Kernel)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional


class ReLUSquaredFeedForward(nn.Module):
    """
    ReLU² FeedForward layer - faster than SwiGLU for speedruns.

    Uses ReLU(x)² activation which:
    - Converges more aggressively in short training runs
    - Has simpler computation than SwiGLU
    - Maintains competitive quality for pretraining

    From modded-nanogpt speedrun.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ReLU² activation: relu(x)^2
        hidden = F.relu(self.up_proj(x))
        hidden = hidden * hidden  # Square the ReLU output
        return self.down_proj(self.dropout(hidden))


class CheckpointedTransformerBlock(nn.Module):
    """
    Transformer block with gradient checkpointing support.

    Gradient checkpointing trades computation for memory by:
    - Not storing intermediate activations during forward pass
    - Recomputing them during backward pass
    - Achieves ~50% memory reduction with ~33% compute overhead

    This enables training larger models or bigger batches on limited VRAM.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        n_kv_heads: Optional[int] = None,
        use_checkpoint: bool = True,
        use_relu_squared: bool = False,
    ):
        super().__init__()
        from models.layers import MultiHeadAttention
        from models.components import SwiGLUFeedForward

        self.use_checkpoint = use_checkpoint

        # Attention
        self.attention = MultiHeadAttention(
            d_model, n_heads, max_seq_len, dropout, n_kv_heads
        )

        # FeedForward - choose between SwiGLU and ReLU²
        if use_relu_squared:
            self.feed_forward = ReLUSquaredFeedForward(d_model, d_ff, dropout)
        else:
            self.feed_forward = SwiGLUFeedForward(d_model, d_ff, dropout)

        # Normalization
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _attention_block(self, x: torch.Tensor) -> torch.Tensor:
        """Attention sub-block for checkpointing."""
        return self.dropout_layer(self.attention(self.norm1(x)))

    def _ffn_block(self, x: torch.Tensor) -> torch.Tensor:
        """FFN sub-block for checkpointing."""
        return self.dropout_layer(self.feed_forward(self.norm2(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # Gradient checkpointing - recompute activations during backward
            x = x + checkpoint(self._attention_block, x, use_reentrant=False)
            x = x + checkpoint(self._ffn_block, x, use_reentrant=False)
        else:
            # Standard forward pass
            x = x + self._attention_block(x)
            x = x + self._ffn_block(x)
        return x


class SpeedrunLLM(nn.Module):
    """
    LLM with speedrun optimizations.

    Combines:
    - Gradient checkpointing for memory efficiency
    - ReLU² activation option for faster convergence
    - Untied embeddings option for flexible output learning
    - Compatible with Liger Kernel when available

    Usage:
        model = SpeedrunLLM(
            config,
            use_checkpoint=True,      # Enable gradient checkpointing
            use_relu_squared=True,    # Use ReLU² instead of SwiGLU
            untie_embeddings=True,    # Separate lm_head from embeddings
        )
    """
    def __init__(
        self,
        config,
        use_checkpoint: bool = True,
        use_relu_squared: bool = False,
        untie_embeddings: bool = False,
    ):
        super().__init__()
        import math

        self.config = config
        self.use_checkpoint = use_checkpoint
        self.use_relu_squared = use_relu_squared
        self.untie_embeddings = untie_embeddings

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_dropout = nn.Dropout(config.dropout)

        # Transformer blocks with checkpointing
        self.transformer_blocks = nn.ModuleList([
            CheckpointedTransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.max_seq_len,
                config.dropout,
                n_kv_heads=config.n_kv_heads,
                use_checkpoint=use_checkpoint,
                use_relu_squared=use_relu_squared,
            )
            for _ in range(config.n_layers)
        ])

        # Output layers
        self.norm = nn.RMSNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

        # Language modeling head
        if untie_embeddings:
            # Separate lm_head - more parameters but more flexible
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            # Initialize lm_head separately
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        else:
            # Tied weights (standard approach)
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Log configuration
        print(f"SpeedrunLLM Configuration:")
        print(f"  - Gradient Checkpointing: {use_checkpoint}")
        print(f"  - Activation: {'ReLU²' if use_relu_squared else 'SwiGLU'}")
        print(f"  - Embeddings: {'Untied' if untie_embeddings else 'Tied'}")
        print(f"  - Parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning logits.

        Args:
            x: Input token IDs [B, T]

        Returns:
            Logits [B, T, V]
        """
        import math

        # Token embeddings with scaling
        x = self.token_embedding(x) * math.sqrt(self.config.d_model)
        x = self.position_dropout(x)

        # Pass through transformer blocks (with checkpointing if enabled)
        for block in self.transformer_blocks:
            x = block(x)

        # Output projection
        x = self.norm(x)
        x = self.output_dropout(x)
        logits = self.lm_head(x)

        return logits

    def get_memory_estimate(self, batch_size: int, seq_len: int) -> dict:
        """
        Estimate memory usage with and without checkpointing.

        Returns dict with memory estimates in MB.
        """
        params_mb = sum(p.numel() * p.element_size() for p in self.parameters()) / 1024 / 1024

        # Activation memory per layer (rough estimate)
        hidden_size = self.config.d_model
        # Without checkpointing: store all activations
        activations_per_layer = batch_size * seq_len * hidden_size * 4 * 2  # 4 bytes * 2 (attention + FFN)
        activations_per_layer_mb = activations_per_layer / 1024 / 1024

        n_layers = self.config.n_layers

        if self.use_checkpoint:
            # With checkpointing: only store input to each layer
            total_activations_mb = activations_per_layer_mb * 2  # sqrt(n_layers) approximately
        else:
            # Without: store all layer activations
            total_activations_mb = activations_per_layer_mb * n_layers

        return {
            "params_mb": params_mb,
            "activations_mb_estimate": total_activations_mb,
            "checkpoint_enabled": self.use_checkpoint,
            "checkpoint_savings_estimate": f"~{(1 - 2/n_layers) * 100:.0f}%" if self.use_checkpoint else "N/A",
        }


def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Enable gradient checkpointing on an existing model.

    Works with any model that has transformer_blocks attribute.
    """
    if hasattr(model, 'transformer_blocks'):
        for block in model.transformer_blocks:
            if hasattr(block, 'use_checkpoint'):
                block.use_checkpoint = True
    return model


def disable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Disable gradient checkpointing on a model.
    """
    if hasattr(model, 'transformer_blocks'):
        for block in model.transformer_blocks:
            if hasattr(block, 'use_checkpoint'):
                block.use_checkpoint = False
    return model


def get_speedrun_status() -> dict:
    """Get status of speedrun optimizations."""
    return {
        "gradient_checkpointing": True,
        "relu_squared": True,
        "untied_embeddings": True,
        "expected_improvements": {
            "checkpoint_memory_reduction": "~50%",
            "checkpoint_compute_overhead": "~33%",
            "relu_squared_speedup": "Faster convergence for short runs",
        }
    }
