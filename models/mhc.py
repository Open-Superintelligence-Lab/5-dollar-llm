"""
Manifold-Constrained Hyper-Connections (mHC)

Implementation of the mHC paper: "mHC: Manifold-Constrained Hyper-Connections"
by Xie et al. (DeepSeek-AI)

mHC extends the residual connection paradigm by:
1. Expanding the residual stream width by a factor of n
2. Using learnable mappings (H_pre, H_post, H_res) for feature routing
3. Projecting H_res onto the Birkhoff polytope (doubly stochastic matrices) 
   via the Sinkhorn-Knopp algorithm to ensure training stability
4. Constraining H_pre and H_post to be non-negative via sigmoid

Key insight: The doubly stochastic constraint ensures that:
- Spectral norm of H_res is bounded by 1 (non-expansive)
- Composite mappings across layers remain doubly stochastic
- Signal/gradient propagation is stable across arbitrary depths
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def sinkhorn_knopp(
    M: torch.Tensor, 
    n_iters: int = 20, 
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm to project a matrix onto the Birkhoff polytope.
    
    The algorithm iteratively normalizes rows and columns to sum to 1,
    converging to a doubly stochastic matrix.
    
    Args:
        M: Input matrix of shape [..., n, n] (will be exponentiated)
        n_iters: Number of Sinkhorn iterations (default 20 as per paper)
        eps: Small constant for numerical stability
        
    Returns:
        Doubly stochastic matrix of same shape as input
    """
    # Exponentiate to ensure positivity (required for Sinkhorn-Knopp)
    M_pos = torch.exp(M)
    
    # Iterative row/column normalization
    for _ in range(n_iters):
        # Row normalization: T_r - divide each row by its sum
        M_pos = M_pos / (M_pos.sum(dim=-1, keepdim=True) + eps)
        # Column normalization: T_c - divide each column by its sum  
        M_pos = M_pos / (M_pos.sum(dim=-2, keepdim=True) + eps)
    
    return M_pos


class ManifoldConstrainedHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection (mHC) module.
    
    Extends standard residual connections by:
    1. Maintaining an n-stream residual (expanded feature dimension)
    2. Learning mappings H_pre (read-out), H_post (write-in), H_res (mixing)
    3. Constraining H_res to be doubly stochastic via Sinkhorn-Knopp
    4. Constraining H_pre, H_post to be non-negative via sigmoid
    
    The forward propagation follows:
        h_out = x_l+1 = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
        
    where x_l is the n-stream residual of shape [batch, seq_len, n, C]
    """
    
    def __init__(
        self,
        d_model: int,
        expansion_rate: int = 4,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
    ):
        """
        Args:
            d_model: Hidden dimension of the model (C in the paper)
            expansion_rate: Number of streams (n in the paper, default 4)
            alpha_init: Initial value for gating factors (default 0.01)
            sinkhorn_iters: Number of Sinkhorn-Knopp iterations (default 20)
        """
        super().__init__()
        
        self.d_model = d_model
        self.n = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        
        # Input dimension for dynamic mappings: n * C (flattened residual stream)
        input_dim = self.n * d_model
        
        # ============ LEARNABLE PARAMETERS ============
        # Gating factors (initialized to small values as per Eq. 5)
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))
        
        # Linear projections for dynamic mappings (Eq. 7)
        # phi_pre: R^(nC) -> R^n
        self.phi_pre = nn.Linear(input_dim, self.n, bias=False)
        # phi_post: R^(nC) -> R^n
        self.phi_post = nn.Linear(input_dim, self.n, bias=False)
        # phi_res: R^(nC) -> R^(n^2)
        self.phi_res = nn.Linear(input_dim, self.n * self.n, bias=False)
        
        # Static biases (learnable)
        self.b_pre = nn.Parameter(torch.zeros(1, self.n))
        self.b_post = nn.Parameter(torch.zeros(1, self.n))
        self.b_res = nn.Parameter(torch.zeros(self.n, self.n))
        
        # RMSNorm for input normalization
        self.rms_norm = nn.RMSNorm(input_dim)
        
        # Initialize biases for stable starting point
        self._init_biases()
        
    def _init_biases(self):
        """Initialize biases for stable identity-like behavior at start."""
        with torch.no_grad():
            # H_pre: uniform weights of 1/n (for averaging across streams)
            self.b_pre.fill_(0.0)  # Will be passed through sigmoid -> ~0.5
            
            # H_post: uniform weights (for distributing back to streams)
            self.b_post.fill_(0.0)  # Will be passed through sigmoid -> ~0.5, then *2 -> ~1.0
            
            # H_res: identity-like (diagonal dominant)
            # After Sinkhorn on exp(b_res), we want roughly identity
            # Set diagonal higher
            torch.nn.init.zeros_(self.b_res)
            self.b_res.data.fill_diagonal_(2.0)  # Higher on diagonal
            
    def compute_mappings(
        self, 
        x_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the mapping coefficients H_pre, H_post, H_res.
        
        Args:
            x_flat: Flattened residual stream of shape [batch, seq_len, n*C]
            
        Returns:
            H_pre: Shape [batch, seq_len, 1, n] - aggregates n streams to 1
            H_post: Shape [batch, seq_len, 1, n] - broadcasts 1 output to n streams
            H_res: Shape [batch, seq_len, n, n] - doubly stochastic mixing matrix
        """
        batch_size, seq_len, _ = x_flat.shape
        
        # Apply RMSNorm (Eq. 7)
        x_norm = self.rms_norm(x_flat)
        
        # Compute dynamic mappings
        H_tilde_pre = self.alpha_pre * self.phi_pre(x_norm) + self.b_pre
        H_tilde_post = self.alpha_post * self.phi_post(x_norm) + self.b_post
        H_tilde_res = self.alpha_res * self.phi_res(x_norm).view(
            batch_size, seq_len, self.n, self.n
        ) + self.b_res
        
        # Apply manifold constraints (Eq. 8)
        # H_pre: non-negative via sigmoid
        H_pre = torch.sigmoid(H_tilde_pre).unsqueeze(-2)  # [B, T, 1, n]
        
        # H_post: non-negative, scaled by 2 (as per paper Eq. 8)
        H_post = 2 * torch.sigmoid(H_tilde_post).unsqueeze(-2)  # [B, T, 1, n]
        
        # H_res: doubly stochastic via Sinkhorn-Knopp
        H_res = sinkhorn_knopp(H_tilde_res, n_iters=self.sinkhorn_iters)  # [B, T, n, n]
        
        return H_pre, H_post, H_res
    
    def read_out(self, x_streams: torch.Tensor, H_pre: torch.Tensor) -> torch.Tensor:
        """
        Aggregate n-stream residual to single layer input.
        
        H_pre @ x_l: [B, T, 1, n] @ [B, T, n, C] -> [B, T, 1, C] -> [B, T, C]
        
        Args:
            x_streams: n-stream residual [batch, seq_len, n, C]
            H_pre: Pre-mapping [batch, seq_len, 1, n]
            
        Returns:
            Layer input [batch, seq_len, C]
        """
        # [B, T, 1, n] @ [B, T, n, C] -> [B, T, 1, C]
        out = torch.matmul(H_pre, x_streams)
        return out.squeeze(-2)  # [B, T, C]
    
    def write_in_and_mix(
        self, 
        x_streams: torch.Tensor,
        layer_output: torch.Tensor,
        H_post: torch.Tensor,
        H_res: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply residual mixing and write layer output back to streams.
        
        x_l+1 = H_res @ x_l + H_post^T @ F(...)
        
        Args:
            x_streams: Current n-stream residual [batch, seq_len, n, C]
            layer_output: Output from layer F [batch, seq_len, C]
            H_post: Post-mapping [batch, seq_len, 1, n]
            H_res: Residual (mixing) mapping [batch, seq_len, n, n]
            
        Returns:
            Updated n-stream residual [batch, seq_len, n, C]
        """
        # Residual mixing: H_res @ x_l
        # [B, T, n, n] @ [B, T, n, C] -> [B, T, n, C]
        mixed = torch.matmul(H_res, x_streams)
        
        # Write-in: H_post^T @ layer_output
        # layer_output: [B, T, C] -> [B, T, 1, C]
        # H_post^T: [B, T, 1, n] -> [B, T, n, 1]
        # Result: [B, T, n, 1] @ [B, T, 1, C] -> [B, T, n, C]
        layer_out_expanded = layer_output.unsqueeze(-2)  # [B, T, 1, C]
        H_post_T = H_post.transpose(-2, -1)  # [B, T, n, 1]
        write_in = torch.matmul(H_post_T, layer_out_expanded)  # [B, T, n, C]
        
        return mixed + write_in


class mHCTransformerBlock(nn.Module):
    """
    Transformer block with mHC residual connections.
    
    Unlike standard transformer blocks that use simple residual connections,
    this block uses mHC for both attention and FFN sub-layers.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        expansion_rate: int = 4,
        dropout: float = 0.1,
        n_kv_heads: Optional[int] = None,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
    ):
        super().__init__()
        
        from models.layers import MultiHeadAttention
        from models.components import SquaredReLUFeedForward
        
        self.d_model = d_model
        self.n = expansion_rate
        
        # Core layers
        self.attention = MultiHeadAttention(
            d_model, n_heads, max_seq_len, dropout, n_kv_heads
        )
        self.feed_forward = SquaredReLUFeedForward(d_model, d_ff, dropout)
        
        # mHC modules for attention and FFN
        self.mhc_attn = ManifoldConstrainedHyperConnection(
            d_model, expansion_rate, alpha_init, sinkhorn_iters
        )
        self.mhc_ffn = ManifoldConstrainedHyperConnection(
            d_model, expansion_rate, alpha_init, sinkhorn_iters
        )
        
        # Pre-norm layers
        self.norm_attn = nn.RMSNorm(d_model)
        self.norm_ffn = nn.RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_streams: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mHC residuals.
        
        Args:
            x_streams: n-stream residual [batch, seq_len, n, C]
            
        Returns:
            Updated n-stream residual [batch, seq_len, n, C]
        """
        batch_size, seq_len, n, d_model = x_streams.shape
        
        # ============ ATTENTION SUB-LAYER ============
        # Flatten for mHC coefficient computation
        x_flat = x_streams.view(batch_size, seq_len, n * d_model)
        
        # Compute mHC mappings
        H_pre_attn, H_post_attn, H_res_attn = self.mhc_attn.compute_mappings(x_flat)
        
        # Read out: aggregate n streams to single layer input
        attn_input = self.mhc_attn.read_out(x_streams, H_pre_attn)  # [B, T, C]
        
        # Apply attention
        attn_output = self.attention(self.norm_attn(attn_input))
        attn_output = self.dropout(attn_output)
        
        # Write in and mix: update n-stream residual
        x_streams = self.mhc_attn.write_in_and_mix(
            x_streams, attn_output, H_post_attn, H_res_attn
        )
        
        # ============ FFN SUB-LAYER ============
        # Flatten again (x_streams has been updated)
        x_flat = x_streams.view(batch_size, seq_len, n * d_model)
        
        # Compute mHC mappings
        H_pre_ffn, H_post_ffn, H_res_ffn = self.mhc_ffn.compute_mappings(x_flat)
        
        # Read out
        ffn_input = self.mhc_ffn.read_out(x_streams, H_pre_ffn)  # [B, T, C]
        
        # Apply FFN
        ffn_output = self.feed_forward(self.norm_ffn(ffn_input))
        ffn_output = self.dropout(ffn_output)
        
        # Write in and mix
        x_streams = self.mhc_ffn.write_in_and_mix(
            x_streams, ffn_output, H_post_ffn, H_res_ffn
        )
        
        return x_streams


class StreamExpansion(nn.Module):
    """
    Expands a single-stream input to n-stream residual.
    Used at the start of the model.
    """
    
    def __init__(self, d_model: int, expansion_rate: int = 4, init_mode: str = "replicate"):
        """
        Args:
            d_model: Hidden dimension
            expansion_rate: Number of streams (n)
            init_mode: How to initialize streams
                - "replicate": Copy input to all streams
                - "zeros_except_first": Only first stream gets input
                - "learned": Learn a projection
        """
        super().__init__()
        self.d_model = d_model
        self.n = expansion_rate
        self.init_mode = init_mode
        
        if init_mode == "learned":
            self.expansion_proj = nn.Linear(d_model, self.n * d_model, bias=False)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, C]
            
        Returns:
            n-stream residual [batch, seq_len, n, C]
        """
        if self.init_mode == "replicate":
            # Replicate input to all streams
            return x.unsqueeze(-2).expand(-1, -1, self.n, -1).clone()
        
        elif self.init_mode == "zeros_except_first":
            # Only first stream gets the input
            batch_size, seq_len, d_model = x.shape
            streams = torch.zeros(
                batch_size, seq_len, self.n, d_model,
                device=x.device, dtype=x.dtype
            )
            streams[:, :, 0, :] = x
            return streams
        
        elif self.init_mode == "learned":
            batch_size, seq_len, _ = x.shape
            return self.expansion_proj(x).view(batch_size, seq_len, self.n, self.d_model)


class StreamContraction(nn.Module):
    """
    Contracts n-stream residual back to single stream.
    Used at the end of the model.
    """
    
    def __init__(self, d_model: int, expansion_rate: int = 4, mode: str = "mean"):
        """
        Args:
            d_model: Hidden dimension  
            expansion_rate: Number of streams (n)
            mode: How to aggregate streams
                - "mean": Average across streams
                - "first": Take only first stream
                - "learned": Learn a weighted combination
        """
        super().__init__()
        self.d_model = d_model
        self.n = expansion_rate
        self.mode = mode
        
        if mode == "learned":
            self.weights = nn.Parameter(torch.ones(self.n) / self.n)
            
    def forward(self, x_streams: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_streams: n-stream residual [batch, seq_len, n, C]
            
        Returns:
            Contracted output [batch, seq_len, C]
        """
        if self.mode == "mean":
            return x_streams.mean(dim=-2)
        
        elif self.mode == "first":
            return x_streams[:, :, 0, :]
        
        elif self.mode == "learned":
            # Weighted sum across streams
            weights = F.softmax(self.weights, dim=0)  # [n]
            # [B, T, n, C] * [n, 1] -> [B, T, n, C] -> sum -> [B, T, C]
            return (x_streams * weights.unsqueeze(-1)).sum(dim=-2)
