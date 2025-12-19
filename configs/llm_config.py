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
    
    # Data params
    max_seq_len: int = 2048  
    vocab_size: int = 49152  
    
    # Base Training Defaults
    compile_model: bool = True
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    train_tokens: int = 20000000
    
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
    dropout: float = 0.0
    grad_clip: float = 1.0
    use_amp: bool = True
    z_loss_weight: float = 1e-4  # Z-Loss for logit stability (0 to disable)

    # Token Smearing (arXiv research - improves token representations)
    use_token_smearing: bool = False  # Enable token smearing
    smear_lambda: float = 0.07  # Smearing strength
    smear_gate_dim: int = 12  # Input dims for smear gate

    # Logging
    log_milestones: Tuple[int, ...] = (100, 500, 1000)

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