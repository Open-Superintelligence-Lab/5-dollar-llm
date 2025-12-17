import argparse
import time
import os
import torch
import logging
from torch.utils.data import DataLoader

# Fix tokenizer parallelism warning when using DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# On CPU with multiprocessing, torch.compile needs Python fallback to avoid C++ compiler issues
if not torch.cuda.is_available():
    os.environ["TORCH_INDUCTOR_SKIP_CPP_WRAPPER"] = "1"
    # Suppress torch.compile errors on CPU and fall back to eager mode
    torch._dynamo.config.suppress_errors = True

from configs.llm_config import DebugMoEConfig
from configs.dataset_config import DataConfig
from training.trainer import train_moe_model
from utils.helpers import set_seed
from utils.logger import setup_logging
from train_llm import prepare_datasets

def main():
    logger = setup_logging(log_dir="./logs_debug")
    logger.info("Starting MoE DEBUG training")

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    
    set_seed(42)
    
    # Use DebugMoEConfig
    config = DebugMoEConfig()
    
    print("Loading dataset with Hugging Face Datasets API...")
    # For debug, we just want a small number of docs
    num_docs = 100
    data_cfg = DataConfig(
        seq_length=config.max_seq_len,
        num_samples=num_docs,
        cache_dir="./hf_cache_debug",
    )

    from data.loader import setup_tokenizer
    
    # Setup tokenizer first to get vocab size
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size

    # Prepare datasets (handles caching automatically)
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer, cache_dir="./processed_data_debug")
    
    logger.info(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    if device == "CPU":
        loader_args["pin_memory"] = False
        loader_args["num_workers"] = os.cpu_count() // 2 # Avoid multiprocessing issues on some setups or low resource
        
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    print("\nModel configuration")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"experts: {config.num_experts}, top‑k: {config.expert_top_k}")
    print(f"steps: {config.max_steps}, batch size: {config.batch_size}")
    print(f"vocab size: {config.vocab_size}\n")
    logger.info(f"Model configuration: {vars(config)}")

    print("Starting DEBUG training...")
    print("-" * 70)
    start = time.time()
    
    # Override output dir for debugging
    output_dir = "./checkpoints/debug_run"
    experiment_name = "debug_experiment"

    model, metrics = train_moe_model(config, train_loader, val_loader, output_dir=output_dir, experiment_name=experiment_name)
    elapsed = (time.time() - start) / 60
    logger.info("Training complete")

    print("\nResults")
    print("-" * 70)
    print(f"Training time: {elapsed:.2f} min")
    print(f"Val loss:       {metrics['val_loss']:.4f}")
    
    # Don't save full checkpoint for debug run, just maybe minimal or none
    print("Debug run complete. No heavy checkpoint saved.")

if __name__ == "__main__":
    main()
