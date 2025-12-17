import argparse
import time
import os
import torch
import logging
from torch.utils.data import DataLoader

# Fix tokenizer parallelism warning when using DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.llm_config import MoEModelConfig, GPU24GBMoEModelConfig
from configs.dataset_config import DataConfig
from training.trainer import train_moe_model
from utils.helpers import set_seed
from utils.logger import setup_logging


def print_system_info():
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}\n")


def prepare_datasets(data_cfg, tokenizer, cache_dir="./processed_data"):
    import json
    import shutil
    from datasets import load_from_disk, load_dataset, Dataset
    from data.loader import tokenize_and_chunk, finalize_dataset

    # cache_dir provided via argument
    train_cache = os.path.join(cache_dir, "train")
    val_cache = os.path.join(cache_dir, "val")
    info_path = os.path.join(cache_dir, "dataset_info.json")

    # Define what config parameters invalidate the cache
    config_state = {
        "dataset_path": data_cfg.dataset_path,
        "dataset_name": data_cfg.dataset_name,
        "tokenizer_name": data_cfg.tokenizer_name,
        "seq_length": data_cfg.seq_length,
        "num_samples": data_cfg.num_samples,
    }

    # 1. Try to load valid cache
    if os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                if json.load(f) == config_state:
                    print(f"Loading cached datasets from {cache_dir}...")
                    return load_from_disk(train_cache), load_from_disk(val_cache)
            print("Cache configuration mismatch. Rebuilding...")
        except Exception as e:
            print(f"Cache check failed ({e}). Rebuilding...")
    
    # 2. Rebuild cache
    if os.path.exists(cache_dir):
        print(f"Cleaning old cache at {cache_dir}...")
        shutil.rmtree(cache_dir)
    
    # Ensure directory exists immediately
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load and split
    print("Loading raw dataset and splitting documents...")
    raw_dataset = load_dataset(
        data_cfg.dataset_path,
        data_cfg.dataset_name,
        split=data_cfg.split,
        cache_dir=data_cfg.cache_dir,
        streaming=True,
    )
    
    raw_samples = list(raw_dataset.take(data_cfg.num_samples))
    num_val = int(len(raw_samples) * 0.1)
    num_train = len(raw_samples) - num_val
    
    raw_train = Dataset.from_list(raw_samples[:num_train])
    raw_val = Dataset.from_list(raw_samples[num_train:])
    print(f"Split into {len(raw_train):,} train docs and {len(raw_val):,} val docs")
    
    # Tokenize and save
    print("Tokenizing train set...")
    train_ds = finalize_dataset(tokenize_and_chunk(raw_train, tokenizer, data_cfg), data_cfg)
    train_ds.save_to_disk(train_cache)
    
    print("Tokenizing validation set...")
    val_ds = finalize_dataset(tokenize_and_chunk(raw_val, tokenizer, data_cfg), data_cfg)
    val_ds.save_to_disk(val_cache)

    # Save cache info
    os.makedirs(cache_dir, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(config_state, f, indent=2)
    print("Saved dataset cache info.")

    return train_ds, val_ds


def main():
    logger = setup_logging(log_dir="./logs")
    logger.info("Starting MoE training")

    print_system_info()
    set_seed(42)
    parser = argparse.ArgumentParser(description="Train MoE Model")
    parser.add_argument("--muon_lr", type=float, help="Override Muon learning rate")
    parser.add_argument("--adamw_lr", type=float, help="Override AdamW learning rate")
    parser.add_argument("--max_steps", type=int, help="Override max_steps")
    parser.add_argument("--experiment_name", type=str, default="moe_training", help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    args = parser.parse_args()

    # For H100 uncomment MoEModelConfig, for small GPU uncomment GPU24GBMoEModelConfig
    # config = MoEModelConfig()
    config = GPU24GBMoEModelConfig()

    # Override config with args
    if args.muon_lr is not None:
        config.muon_lr = args.muon_lr
    if args.adamw_lr is not None:
        config.adamw_lr = args.adamw_lr
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    
    experiment_name = args.experiment_name
    output_dir = os.path.join(args.output_dir, experiment_name)

    # Calculate required documents dynamically
    # Assume avg 1000 tokens per doc (conservative estimate)
    # Safety factor 2.0 to ensure enough data
    avg_tokens_per_doc = 1000
    safety_factor = 2.0
    total_tokens_needed = config.max_steps * config.batch_size * config.max_seq_len
    calc_num_docs = int((total_tokens_needed / avg_tokens_per_doc) * safety_factor)
    
    # For very short runs (debugging), we verify we have at least some docs.
    if calc_num_docs < 100:
        calc_num_docs = 100
        
    print(f"Dynamic Data Calculation:")
    print(f"  Steps: {config.max_steps}, Batch: {config.batch_size}, Seq: {config.max_seq_len}")
    print(f"  Total tokens needed: {total_tokens_needed:,}")
    print(f"  Est. docs needed (factor {safety_factor}): {calc_num_docs:,}")
    
    # config.num_documents = calc_num_docs # Removed from config
    num_docs = calc_num_docs

    print("Loading dataset with Hugging Face Datasets API...")
    data_cfg = DataConfig(
        seq_length=config.max_seq_len,
        num_samples=num_docs,
        cache_dir="./hf_cache",
    )

    from data.loader import setup_tokenizer
    
    # Setup tokenizer first to get vocab size
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size

    # Prepare datasets (handles caching automatically)
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)
    
    logger.info(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    # Check for sufficient data
    total_needed = config.max_steps * config.batch_size
    if len(train_ds) < total_needed:
        msg = (
            f"Insufficient training data! "
            f"Need {total_needed} sequences (max_steps={config.max_steps} * batch_size={config.batch_size}) "
            f"but only have {len(train_ds)} sequences. "
            f"The model will overfit if data repeats. "
            f"To fix: increase inferred num_documents (currently {num_docs}) "
            f"or reduce max_steps."
        )
        logger.error(msg)
        raise ValueError(msg)

    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
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

    print("Starting training...")
    print("-" * 70)
    start = time.time()

    model, metrics = train_moe_model(config, train_loader, val_loader, output_dir=output_dir, experiment_name=experiment_name)
    elapsed = (time.time() - start) / 60
    logger.info("Training complete")

    print("\nResults")
    print("-" * 70)
    print(f"Training time: {elapsed:.2f} min")
    print(f"Val loss:       {metrics['val_loss']:.4f}")
    print(f"Val accuracy:   {metrics['val_accuracy']:.4f}")
    print(f"Val perplexity: {metrics['val_perplexity']:.2f}")
    logger.info(f"Final metrics: {metrics}")

    ckpt_path = os.path.join(output_dir, "final_model.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(),
         "config": config,
         "metrics": metrics},
        ckpt_path,
    )
    print(f"Model checkpoint saved to {ckpt_path}")
    logger.info(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
