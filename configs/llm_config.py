from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BlueberryConfig:
    # Model architecture (88M Params - Blueberry-Nano)
    d_model: int = 512       
    n_heads: int = 8         
    n_layers: int = 22
    d_ff: int = 2048         
    
    # GQA parameters
    n_kv_heads: int = 4      
    
    # mHC (Manifold-Constrained Hyper-Connections) parameters
    use_mhc: bool = False                    # Enable mHC instead of standard residual
    mhc_expansion_rate: int = 4              # Number of streams (n in paper)
    mhc_alpha_init: float = 0.01             # Initial gating factor for dynamic mappings
    mhc_sinkhorn_iters: int = 20             # Sinkhorn-Knopp iterations for doubly stochastic
    mhc_leading_dense_layers: int = 1        # Number of initial layers using standard residual
    mhc_stream_init: str = "replicate"       # How to initialize streams: replicate, zeros_except_first, learned
    mhc_stream_contract: str = "mean"        # How to contract streams: mean, first, learned
    
    # Data params
    # ⚠️ WARNING: For simplicity, I recomend not changing max_seq_len
    # If you change max_seq_len, you MUST re-run data preparation!
    # The data preparation script chunks data at this exact length, and the RoPE
    # cache is initialized with this value. Mismatches will cause runtime errors.
    # Run: python data/prepare_mix_data.py --target_tokens 25_000_000
    # you may change the number of tokens
    max_seq_len: int = 2048  # check the warning above
    vocab_size: int = 49152  
    
    # Base Training Defaults
    compile_model: bool = True
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    train_tokens: int = 8000000
    
    # Learning Rate (Aggressive for pre-training)
    muon_lr: float = 0.024
    muon_momentum: float = 0.95
    adamw_lr: float = 0.006
    warmup_ratio: float = 0.0
    schedule_type: str = "constant"

    # Evaluation
    eval_every: Optional[int] = None
    eval_steps: int = 100
    eval_milestones: Optional[Tuple[int, ...]] = None
    
    # Regularization
    weight_decay: float = 0.2
    dropout: float = 0.0
    grad_clip: float = 1.0
    use_amp: bool = True
    
    # Logging
    log_milestones: Tuple[int, ...] = (100, 500, 1000)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

