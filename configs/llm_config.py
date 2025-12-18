from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BlueberryConfig:
    # Model architecture (151M Params - Blueberry-Nano)
    d_model: int = 512       
    n_heads: int = 8         
    n_layers: int = 32       
    d_ff: int = 2048         
    
    # GQA parameters
    n_kv_heads: int = 4      
    
    # Dense model settings (MoE disabled by default)
    use_moe: bool = False
    
    # RoPE / Attention defaults (inherited from previous Base but assumed used)
    qk_rope_dim: int | None = 32
    qk_nope_dim: int | None = 128
    kv_lora_rank: int | None = 64
    v_dim: int | None = 128
    
    # Data params
    max_seq_len: int = 2048  
    vocab_size: int = 49152  
    
    # Base Training Defaults
    compile_model: bool = True
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    train_tokens: int = 1_000_000_000
    
    # Learning Rate (Aggressive for pre-training)
    muon_lr: float = 0.015
    muon_momentum: float = 0.95
    adamw_lr: float = 0.001
    warmup_ratio: float = 0.0
    schedule_type: str = "constant"

    # Evaluation
    eval_every: int = 2000
    eval_steps: int = 100
    
    # Regularization
    weight_decay: float = 0.2
    dropout: float = 0.1
    grad_clip: float = 1.0
    use_amp: bool = True
    
    # Logging
    log_milestones: Tuple[int, ...] = (2000, 5000, 10000)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


@dataclass
class Blueberry24GBConfig(BlueberryConfig):
    # Optimized for RTX 4090 (24GB)
    pass


@dataclass
class Blueberry80GBConfig(BlueberryConfig):
    # Optimized for H100 (80GB)
    batch_size: int = 128
    gradient_accumulation_steps: int = 2


@dataclass
class DebugConfig(BlueberryConfig):
    # Tiny architecture for fast debugging
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    
    # Standard settings (Dense)
    use_moe: bool = False
    
    # Reduced resources
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 128
    
    # Shorter training
    train_tokens: int = 100_000 # ~100 steps
    
    log_milestones: Tuple[int, ...] = (10, 50, 80)
    muon_lr: float = 0.01
    adamw_lr: float = 0.001

    def __post_init__(self):
        super().__post_init__()

