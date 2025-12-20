from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BlueberryConfig:
    """
    Explanation for choosing this config:
    https://github.com/Open-Superintelligence-Lab/5-dollar-llm/issues/58
    # need to possibly adjust layer number and do learning rate search
    """
    # Model architecture (96M Params - Optimized for Batch 8 throughput)
    d_model: int = 512       
    n_heads: int = 8         
    n_layers: int = 20     
    d_ff: int = 2048         
    
    # GQA parameters
    n_kv_heads: int = 4      
    
    # Data params
    max_seq_len: int = 2048  
    vocab_size: int = 49152  
    
    # Base Training Defaults (Optimized for Batch 8)
    compile_model: bool = True
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    train_tokens: int = 20000000
    
    # Learning Rate (Aggressive for pre-training with Batch 8)
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    adamw_lr: float = 0.0012
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
    
    # Logging
    log_milestones: Tuple[int, ...] = (100, 500, 1000)

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"