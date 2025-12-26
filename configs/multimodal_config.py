from dataclasses import dataclass
from typing import Optional, Tuple
from configs.llm_config import BlueberryConfig

@dataclass
class MultimodalConfig(BlueberryConfig):
    # Extension for Multimodal
    text_vocab_size: int = 49152
    image_vocab_size: int = 1024 # Match VQ-VAE
    
    # Special tokens
    # We add 2 special tokens: <seg_start> and <seg_end>
    # vocab_size = text_vocab_size + 2 + image_vocab_size
    vocab_size: int = 49152 + 2 + 1024
    
    seg_start_id: int = 49152
    seg_end_id: int = 49153
    image_token_offset: int = 49154
    
    # Image resolution
    image_size: int = 128
    # 128x128 -> 16x16 grid = 256 tokens (with stride 2 + stride 2 downsampling in VQ-VAE)
    # My VQ-VAE has 2 downsampling layers (stride 2 each), so 128 -> 64 -> 32. 
    # Wait, my VQ-VAE Encoder has: conv1(s2), conv2(s2), conv3(s1).
    # 128 / 2 = 64
    # 64 / 2 = 32
    # So 32x32 = 1024 tokens.
    # If I want 16x16, I need one more stride 2 layer or start with 64x64 images.
    # Let's stick with 32x32 = 1024 tokens for 128x128 image.
    num_image_tokens: int = 1024 
    
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8 # Shallow for faster learning from scratch
    d_ff: int = 1024
    max_seq_len: int = 2024
    
    # Training overrides
    batch_size: int = 20 # Increased to push memory usage towards 80-90%
    muon_lr: float = 0.02 # Found to be optimal via LR search
