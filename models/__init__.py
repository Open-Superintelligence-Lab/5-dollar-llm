from .layers import (
    Rotary,
    MultiHeadAttention,
    TransformerBlock,
)
from .llm import MinimalLLM
from .mhc import (
    sinkhorn_knopp,
    ManifoldConstrainedHyperConnection,
    mHCTransformerBlock,
    StreamExpansion,
    StreamContraction,
)
from .llm_mhc import MinimalLLM_mHC, MinimalLLM_mHC_Hybrid

__all__ = [
    "Rotary",
    "MultiHeadAttention",
    "TransformerBlock",
    "MinimalLLM",
    # mHC components
    "sinkhorn_knopp",
    "ManifoldConstrainedHyperConnection",
    "mHCTransformerBlock",
    "StreamExpansion",
    "StreamContraction",
    "MinimalLLM_mHC",
    "MinimalLLM_mHC_Hybrid",
]

