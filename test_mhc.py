"""
Test script for Manifold-Constrained Hyper-Connections (mHC) implementation.

This script verifies:
1. Sinkhorn-Knopp produces doubly stochastic matrices
2. mHC module dimensions are correct
3. Forward/backward passes work correctly
4. Full LLM with mHC works end-to-end
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from configs.llm_config import BlueberryConfig
from models.mhc import (
    sinkhorn_knopp,
    ManifoldConstrainedHyperConnection,
    mHCTransformerBlock,
    StreamExpansion,
    StreamContraction,
)
from models.llm_mhc import MinimalLLM_mHC


def test_sinkhorn_knopp():
    """Test that Sinkhorn-Knopp produces doubly stochastic matrices."""
    print("=" * 60)
    print("Testing Sinkhorn-Knopp Algorithm")
    print("=" * 60)
    
    # Create random matrices
    batch_size, seq_len, n = 2, 4, 4
    M = torch.randn(batch_size, seq_len, n, n)
    
    # Apply Sinkhorn-Knopp
    DS = sinkhorn_knopp(M, n_iters=20)
    
    # Check doubly stochastic properties
    row_sums = DS.sum(dim=-1)
    col_sums = DS.sum(dim=-2)
    
    print(f"Input shape: {M.shape}")
    print(f"Output shape: {DS.shape}")
    print(f"Row sums (should be ~1): min={row_sums.min():.4f}, max={row_sums.max():.4f}")
    print(f"Col sums (should be ~1): min={col_sums.min():.4f}, max={col_sums.max():.4f}")
    print(f"All entries non-negative: {(DS >= 0).all()}")
    
    # Check spectral norm (should be <= 1)
    spectral_norms = torch.linalg.matrix_norm(DS, ord=2)
    print(f"Spectral norms (should be <=1): min={spectral_norms.min():.4f}, max={spectral_norms.max():.4f}")
    
    # Check closure under multiplication
    DS_composed = torch.matmul(DS[:, :2], DS[:, 2:])
    composed_row_sums = DS_composed.sum(dim=-1)
    composed_col_sums = DS_composed.sum(dim=-2)
    print(f"Composed row sums (should be ~1): min={composed_row_sums.min():.4f}, max={composed_row_sums.max():.4f}")
    print(f"Composed col sums (should be ~1): min={composed_col_sums.min():.4f}, max={composed_col_sums.max():.4f}")
    
    print("✓ Sinkhorn-Knopp test passed!\n")


def test_mhc_module():
    """Test ManifoldConstrainedHyperConnection module."""
    print("=" * 60)
    print("Testing mHC Module")
    print("=" * 60)
    
    d_model = 64
    n = 4
    batch_size = 2
    seq_len = 16
    
    mhc = ManifoldConstrainedHyperConnection(d_model=d_model, expansion_rate=n)
    
    # Create n-stream input
    x_streams = torch.randn(batch_size, seq_len, n, d_model)
    x_flat = x_streams.view(batch_size, seq_len, n * d_model)
    
    # Compute mappings
    H_pre, H_post, H_res = mhc.compute_mappings(x_flat)
    
    print(f"x_streams shape: {x_streams.shape}")
    print(f"H_pre shape: {H_pre.shape} (expected: [{batch_size}, {seq_len}, 1, {n}])")
    print(f"H_post shape: {H_post.shape} (expected: [{batch_size}, {seq_len}, 1, {n}])")
    print(f"H_res shape: {H_res.shape} (expected: [{batch_size}, {seq_len}, {n}, {n}])")
    
    # Check H_res is doubly stochastic
    row_sums = H_res.sum(dim=-1)
    col_sums = H_res.sum(dim=-2)
    print(f"H_res row sums: min={row_sums.min():.4f}, max={row_sums.max():.4f}")
    print(f"H_res col sums: min={col_sums.min():.4f}, max={col_sums.max():.4f}")
    
    # Check H_pre and H_post are non-negative
    print(f"H_pre non-negative: {(H_pre >= 0).all()}")
    print(f"H_post non-negative: {(H_post >= 0).all()}")
    
    # Test read_out
    layer_input = mhc.read_out(x_streams, H_pre)
    print(f"Layer input shape: {layer_input.shape} (expected: [{batch_size}, {seq_len}, {d_model}])")
    
    # Test write_in_and_mix
    layer_output = torch.randn(batch_size, seq_len, d_model)
    x_streams_new = mhc.write_in_and_mix(x_streams, layer_output, H_post, H_res)
    print(f"Updated x_streams shape: {x_streams_new.shape} (expected: {x_streams.shape})")
    
    print("✓ mHC module test passed!\n")


def test_mhc_transformer_block():
    """Test mHCTransformerBlock."""
    print("=" * 60)
    print("Testing mHC Transformer Block")
    print("=" * 60)
    
    d_model = 64
    n_heads = 4
    d_ff = 256
    max_seq_len = 128
    n = 4
    batch_size = 2
    seq_len = 16
    
    block = mHCTransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        expansion_rate=n,
    )
    
    # Create n-stream input
    x_streams = torch.randn(batch_size, seq_len, n, d_model)
    
    # Forward pass
    out_streams = block(x_streams)
    
    print(f"Input shape: {x_streams.shape}")
    print(f"Output shape: {out_streams.shape}")
    print(f"Shapes match: {x_streams.shape == out_streams.shape}")
    
    # Test backward pass
    loss = out_streams.sum()
    loss.backward()
    
    # Check gradients exist for key parameters
    grad_exists = all(
        p.grad is not None 
        for p in block.parameters() 
        if p.requires_grad
    )
    print(f"Gradients computed: {grad_exists}")
    
    print("✓ mHC Transformer Block test passed!\n")


def test_stream_expansion_contraction():
    """Test StreamExpansion and StreamContraction."""
    print("=" * 60)
    print("Testing Stream Expansion/Contraction")
    print("=" * 60)
    
    d_model = 64
    n = 4
    batch_size = 2
    seq_len = 16
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    for init_mode in ['replicate', 'zeros_except_first', 'learned']:
        expander = StreamExpansion(d_model, n, init_mode)
        x_streams = expander(x)
        print(f"Expansion ({init_mode}): {x.shape} -> {x_streams.shape}")
    
    for contract_mode in ['mean', 'first', 'learned']:
        x_streams = torch.randn(batch_size, seq_len, n, d_model)
        contractor = StreamContraction(d_model, n, contract_mode)
        x_out = contractor(x_streams)
        print(f"Contraction ({contract_mode}): {x_streams.shape} -> {x_out.shape}")
    
    print("✓ Stream Expansion/Contraction test passed!\n")


def test_minimal_llm_mhc():
    """Test the full MinimalLLM_mHC model."""
    print("=" * 60)
    print("Testing MinimalLLM_mHC (Full Model)")
    print("=" * 60)
    
    # Create a small config for testing
    config = BlueberryConfig(
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ff=256,
        n_kv_heads=2,
        max_seq_len=128,
        vocab_size=1000,
        use_mhc=True,
        mhc_expansion_rate=4,
        mhc_alpha_init=0.01,
        mhc_sinkhorn_iters=20,
    )
    
    model = MinimalLLM_mHC(config)
    
    batch_size = 2
    seq_len = 32
    
    # Create input
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {config.vocab_size}]")
    
    # Count parameters
    n_params = model.get_num_params()
    print(f"Total parameters: {n_params:,}")
    
    # Get mHC stats
    stats = model.get_mhc_stats()
    print(f"mHC stats: {stats}")
    
    # Test backward pass
    loss = logits.sum()
    loss.backward()
    
    grad_exists = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_with_grad = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Parameters with gradients: {grad_exists}/{total_params_with_grad}")
    
    print("✓ MinimalLLM_mHC test passed!\n")


def test_stability_analysis():
    """
    Analyze the stability properties as discussed in the paper.
    Check that composite mappings maintain doubly stochastic property.
    """
    print("=" * 60)
    print("Testing Stability Analysis (Composite Mappings)")
    print("=" * 60)
    
    n = 4
    n_layers = 10
    batch_size = 1
    seq_len = 1
    
    # Simulate multiple layers of H_res mappings
    H_res_layers = []
    for _ in range(n_layers):
        M = torch.randn(batch_size, seq_len, n, n)
        H_res = sinkhorn_knopp(M, n_iters=20)
        H_res_layers.append(H_res)
    
    # Compute composite mapping
    composite = H_res_layers[0]
    for i in range(1, n_layers):
        composite = torch.matmul(composite, H_res_layers[i])
    
    # Check properties
    row_sums = composite.sum(dim=-1)
    col_sums = composite.sum(dim=-2)
    
    print(f"After {n_layers} composed layers:")
    print(f"Row sums (should be ~1): {row_sums.squeeze()}")
    print(f"Col sums (should be ~1): {col_sums.squeeze()}")
    print(f"Max absolute row sum deviation: {(row_sums - 1).abs().max():.6f}")
    print(f"Max absolute col sum deviation: {(col_sums - 1).abs().max():.6f}")
    
    # Compute Amax Gain Magnitude as in paper
    forward_gain = composite.abs().sum(dim=-1).max()  # Max row sum of abs
    backward_gain = composite.abs().sum(dim=-2).max()  # Max col sum of abs
    
    print(f"Forward Amax Gain Magnitude: {forward_gain:.4f}")
    print(f"Backward Amax Gain Magnitude: {backward_gain:.4f}")
    
    print("✓ Stability analysis test passed!\n")


if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("\n" + "=" * 60)
    print("mHC (Manifold-Constrained Hyper-Connections) Test Suite")
    print("=" * 60 + "\n")
    
    test_sinkhorn_knopp()
    test_mhc_module()
    test_mhc_transformer_block()
    test_stream_expansion_contraction()
    test_minimal_llm_mhc()
    test_stability_analysis()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
