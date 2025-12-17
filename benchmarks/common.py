
import torch
from pathlib import Path
from transformers import AutoTokenizer
import sys
import os

# Add root directory to path
# benchmarks/common.py -> benchmarks/ -> root/
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
sys.path.append(str(root_dir))

from models.llm import MoEMinimalLLM
from configs.llm_config import MoEModelConfig, GPU24GBMoEModelConfig

def load_model_from_checkpoint(checkpoint_path, device='cuda', dtype=torch.bfloat16):
    """
    Load a model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        dtype: Data type
    
    Returns:
        model, config, tokenizer
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    # Safe globals to allow loading custom config class
    torch.serialization.add_safe_globals([MoEModelConfig, GPU24GBMoEModelConfig])
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise e

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("Warning: Config not found in checkpoint, using default MoEModelConfig")
        config = MoEModelConfig()

    # Ensure max_seq_len is large enough for benchmarks
    # RoPE can handle extrapolation/longer sequences to some extent, but cache must be initialized
    if hasattr(config, 'max_seq_len') and config.max_seq_len < 2048:
        print(f"Adjusting max_seq_len from {config.max_seq_len} to 2048 for evaluation")
        config.max_seq_len = 2048

    # Create model
    # Note: the config object should already have the right parameters
    model = MoEMinimalLLM(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Fallback if checkpoint is just the state dict
        model.load_state_dict(checkpoint)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    # Load tokenizer - assuming SmolLM-135M as used in training
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    print(f"✅ Model loaded successfully!")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Precision: {dtype}")
    
    return model, config, tokenizer

def get_device_and_dtype():
    """Get appropriate device and dtype for evaluation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Using bfloat16 precision")
        else:
            dtype = torch.float16
            print("Using float16 precision")
    else:
        dtype = torch.float32
        print("Using float32 precision (CPU mode)")
    
    return device, dtype
