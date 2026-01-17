# Manifold-Constrained Hyper-Connections (mHC)

This document describes the implementation of **mHC** (Manifold-Constrained Hyper-Connections) based on the paper ["mHC: Manifold-Constrained Hyper-Connections" by Xie et al. (DeepSeek-AI)](https://arxiv.org/abs/2512.24880).

## Overview

mHC extends the standard residual connection paradigm by:

1. **Expanding the residual stream width**: Instead of a single residual stream, mHC maintains `n` parallel streams (default `n=4`)
2. **Learning dynamic routing**: Three learnable mappings control information flow:
   - `H_pre`: Aggregates n streams → single layer input
   - `H_post`: Maps layer output → n streams
   - `H_res`: Mixes features within the residual stream
3. **Ensuring stability via manifold constraint**: Projects `H_res` onto the Birkhoff polytope (doubly stochastic matrices) using the Sinkhorn-Knopp algorithm

## Key Insight

Standard Hyper-Connections (HC) provide performance gains but suffer from training instability at scale because the composite mapping across layers diverges from identity, causing signal explosion/vanishing.

**mHC solves this** by constraining `H_res` to be doubly stochastic:
- Row sums = 1, Column sums = 1
- All entries ≥ 0
- Spectral norm ≤ 1 (non-expansive)
- Closed under matrix multiplication

This ensures stable signal propagation across arbitrary depths.

## Architecture

```
Standard Residual:
    x_{l+1} = x_l + F(x_l)

mHC Residual:
    x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
    
where x_l ∈ R^{n×C} is an n-stream residual
```

## Files

### Core Implementation

| File | Description |
|------|-------------|
| `models/mhc.py` | Core mHC module with Sinkhorn-Knopp algorithm |
| `models/llm_mhc.py` | LLM architectures using mHC |
| `configs/llm_config.py` | Configuration parameters for mHC |

### Key Components

```python
# Sinkhorn-Knopp algorithm for doubly stochastic projection
from models.mhc import sinkhorn_knopp

# Core mHC module
from models.mhc import ManifoldConstrainedHyperConnection

# mHC-enabled transformer block  
from models.mhc import mHCTransformerBlock

# Stream expansion/contraction
from models.mhc import StreamExpansion, StreamContraction

# Complete mHC LLM
from models.llm_mhc import MinimalLLM_mHC
```

## Usage

### Training with mHC

```bash
# Train with mHC enabled
python train_llm.py --use_mhc --train_tokens 8000000

# Customize mHC parameters
python train_llm.py --use_mhc \
    --mhc_expansion_rate 4 \
    --mhc_alpha_init 0.01 \
    --mhc_sinkhorn_iters 20
```

### Programmatic Usage

```python
from configs.llm_config import BlueberryConfig
from models.llm_mhc import MinimalLLM_mHC

# Create config with mHC enabled
config = BlueberryConfig(
    use_mhc=True,
    mhc_expansion_rate=4,      # Number of streams (n)
    mhc_alpha_init=0.01,       # Gating factor initialization
    mhc_sinkhorn_iters=20,     # Sinkhorn-Knopp iterations
)

# Create model
model = MinimalLLM_mHC(config)

# Forward pass
logits = model(input_ids)  # [batch, seq_len, vocab_size]
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_mhc` | `False` | Enable mHC instead of standard residual |
| `mhc_expansion_rate` | `4` | Number of streams (n in paper) |
| `mhc_alpha_init` | `0.01` | Initial gating factor for dynamic mappings |
| `mhc_sinkhorn_iters` | `20` | Sinkhorn-Knopp iterations for doubly stochastic |
| `mhc_leading_dense_layers` | `1` | Number of initial layers using standard residual |
| `mhc_stream_init` | `"replicate"` | How to initialize streams: `replicate`, `zeros_except_first`, `learned` |
| `mhc_stream_contract` | `"mean"` | How to contract streams: `mean`, `first`, `learned` |

## Mathematical Details

### Sinkhorn-Knopp Algorithm

The algorithm iteratively normalizes rows and columns to produce a doubly stochastic matrix:

```
M^(0) = exp(H̃_res)  # Ensure positivity

for t = 1 to t_max:
    M^(t) = T_r(T_c(M^(t-1)))  # Row then column normalization
    
H_res = M^(t_max)
```

### Properties of Doubly Stochastic Matrices

1. **Norm Preservation**: `||H_res||_2 ≤ 1` (non-expansive)
2. **Compositional Closure**: Product of doubly stochastic matrices is doubly stochastic
3. **Birkhoff Polytope**: The set forms the convex hull of permutation matrices

## Testing

Run the test suite to verify the implementation:

```bash
python test_mhc.py
```

This tests:
- Sinkhorn-Knopp produces doubly stochastic matrices
- mHC module dimensions are correct
- Forward/backward passes work correctly
- Full LLM with mHC works end-to-end
- Stability analysis of composite mappings

## Performance Notes

### Memory Overhead

mHC increases memory usage due to the n-stream residual:
- Activations: `n×` more memory for residual stream
- Parameters: Additional mappings (H_pre, H_post, H_res) per layer

### Computational Overhead

The paper reports only 6.7% additional time overhead with n=4 when properly optimized with:
- Kernel fusion
- Selective recomputation  
- Communication overlapping (for distributed training)

## References

- [mHC Paper (arXiv:2512.24880)](https://arxiv.org/abs/2512.24880)
- [Hyper-Connections (HC) Paper (arXiv:2409.19606)](https://arxiv.org/abs/2409.19606)
- [Identity Mappings in Deep Residual Networks (He et al., 2016)](https://arxiv.org/abs/1603.05027)
