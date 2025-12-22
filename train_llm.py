import argparse
import time
import os
import torch
import logging
import random
import numpy as np
from torch.utils.data import DataLoader

# Fix tokenizer parallelism warning when using DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.llm_config import BlueberryConfig
from configs.dataset_config import DataConfig
from training.trainer import train_minimal_llm
from utils.helpers import set_seed, format_time
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

    # CASE 0: Dataset path is already a processed on-disk dataset
    # We check if the path passed in data_cfg.dataset_path is a directory containing a dataset dict
    if os.path.isdir(data_cfg.dataset_path):
        # Heuristic: check if it has dataset_dict.json or state.json or just load it
        try:
            print(f"Loading pre-processed dataset from {data_cfg.dataset_path}...")
            # We assume it's a dataset with "input_ids" and "labels"
            ds = load_from_disk(data_cfg.dataset_path)
            
            # Set format to torch for preprocessed datasets
            if hasattr(ds, 'set_format'):
                # Single dataset
                if "input_ids" in ds.column_names and "labels" in ds.column_names:
                    ds.set_format(type="torch", columns=["input_ids", "labels"])
            
            # If it's a DatasetDict (train, val), return it
            if isinstance(ds, dict) or hasattr(ds, "keys"):
                if "train" in ds and "val" in ds:
                    # Set format for both splits
                    if hasattr(ds["train"], 'set_format'):
                        ds["train"].set_format(type="torch", columns=["input_ids", "labels"])
                        ds["val"].set_format(type="torch", columns=["input_ids", "labels"])
                    return ds["train"], ds["val"]
                elif "train" in ds:
                    # Splitting manually if only train exists
                    print("Found only 'train' split. Creating validation split...")
                    splitted = ds["train"].train_test_split(test_size=0.1, seed=42)
                    # Set format for both splits
                    splitted["train"].set_format(type="torch", columns=["input_ids", "labels"])
                    splitted["test"].set_format(type="torch", columns=["input_ids", "labels"])
                    return splitted["train"], splitted["test"]
            
            # If it's a single Dataset (just rows)
            print("Loaded single dataset. Splitting into train/val...")
            splitted = ds.train_test_split(test_size=0.1, seed=42)
            # Set format for both splits
            splitted["train"].set_format(type="torch", columns=["input_ids", "labels"])
            splitted["test"].set_format(type="torch", columns=["input_ids", "labels"])
            return splitted["train"], splitted["test"]

        except Exception as e:

            # Fallback: try loading "train" and "val" subdirectories directly
            try:
                train_path = os.path.join(data_cfg.dataset_path, "train")
                val_path = os.path.join(data_cfg.dataset_path, "val")
                if os.path.exists(train_path) and os.path.exists(val_path):
                    print(f"Loading separate train/val datasets from {data_cfg.dataset_path}...")
                    train_ds = load_from_disk(train_path)
                    val_ds = load_from_disk(val_path)
                    
                    if hasattr(train_ds, 'set_format'):
                        train_ds.set_format(type="torch", columns=["input_ids", "labels"])
                    if hasattr(val_ds, 'set_format'):
                        val_ds.set_format(type="torch", columns=["input_ids", "labels"])
                    return train_ds, val_ds
            except Exception as e2:
                print(f"Sub-directory load failed: {e2}")

            print(f"Could not load as direct dataset ({e}). Falling back to HF loading...")

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
    
    # Streaming requires taking samples explicitly
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
    logger.info("Starting training")

    print_system_info()
    set_seed(42)
    parser = argparse.ArgumentParser(description="Train MoE Model")
    parser.add_argument("--muon_lr", type=float, help="Override Muon learning rate")
    parser.add_argument("--adamw_lr", type=float, help="Override AdamW learning rate")
    parser.add_argument("--train_tokens", type=int, default=8000000, help="Override train_tokens")
    parser.add_argument("--experiment_name", type=str, default="speedrun_4.5", help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--config_class", type=str, help="Python path to config class (e.g., configs.llm_config.BlueberryConfig)")
    parser.add_argument("--load_checkpoint", type=str, help="Path to checkpoint file to load weights from")
    parser.add_argument("--compile", type=str, help="Whether to compile the model (true/false)")
    parser.add_argument("--dataset_path", type=str, help="Path to preprocessed dataset directory")
    parser.add_argument("--eval_every", type=int, help="Override eval_every steps")
    parser.add_argument("--save_every", type=int, help="Override save_every steps")
    parser.add_argument("--batch_size", type=int, help="Override batch_size")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Override gradient_accumulation_steps")
    parser.add_argument("--target_train_loss", type=float, default=0.0, help="Stop training when training loss reaches this value")
    parser.add_argument("--log_every", type=int, default=100, help="Logging frequency in steps")
    parser.add_argument("--warmup", type=str, default="true", help="Whether to perform untimed compilation warmup (true/false)")

    args = parser.parse_args()

    # Load Config
    if args.config_class:
        import importlib
        try:
            module_name, class_name = args.config_class.rsplit(".", 1)
            module = importlib.import_module(module_name)
            ConfigClass = getattr(module, class_name)
            print(f"Loading config from {args.config_class}")
            config = ConfigClass()
        except Exception as e:
            print(f"Error loading config class {args.config_class}: {e}")
            raise e
    else:
        # Default to the optimized Pow2 config
        config = BlueberryConfig()

    # Override config with args
    if args.muon_lr is not None:
        config.muon_lr = args.muon_lr
    if args.adamw_lr is not None:
        config.adamw_lr = args.adamw_lr
    if args.train_tokens is not None:
        config.train_tokens = args.train_tokens
    if args.compile is not None:
        config.compile_model = (args.compile.lower() == "true")
    if args.eval_every is not None:
        config.eval_every = args.eval_every
    if args.save_every is not None:
        config.save_every = args.save_every
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.log_every is not None:
        config.log_every = args.log_every
    
    use_warmup = (args.warmup.lower() == "true")

    
    experiment_name = args.experiment_name
    output_dir = os.path.join(args.output_dir, experiment_name)

    # Calculate required documents dynamically
    # Assume avg 1000 tokens per doc (conservative estimate)
    # Safety factor 2.0 to ensure enough data
    avg_tokens_per_doc = 1000
    safety_factor = 2.0
    total_tokens_needed = config.train_tokens
    calc_num_docs = int((total_tokens_needed / avg_tokens_per_doc) * safety_factor)
    
    # For very short runs (debugging), we verify we have at least some docs.
    if calc_num_docs < 100:
        calc_num_docs = 100
        
    print(f"Dynamic Data Calculation:")
    print(f"  Batch: {config.batch_size}, Seq: {config.max_seq_len}, Accumulation: {config.gradient_accumulation_steps}")
    print(f"  Target tokens: {total_tokens_needed:,}")
    print(f"  Est. docs needed (factor {safety_factor}): {calc_num_docs:,}")
    
    num_docs = calc_num_docs

    print("Loading dataset with Hugging Face Datasets API...")
    data_cfg = DataConfig(
        dataset_path=args.dataset_path if args.dataset_path else DataConfig.dataset_path,
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

    # Worker init function to ensure each worker has a deterministic seed
    def worker_init_fn(worker_id):
        worker_seed = 42 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(42)

    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)


    print("\nModel configuration")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"train tokens: {config.train_tokens:,}")
    print(f"batch size: {config.batch_size}")
    print(f"vocab size: {config.vocab_size}\n")
    logger.info(f"Model configuration: {vars(config)}")

    # Train the model
    train_minimal_llm(
        config, 
        train_loader, 
        val_loader, 
        output_dir=output_dir, 
        experiment_name=experiment_name,
        load_weights_path=args.load_checkpoint,
        target_train_loss=args.target_train_loss,
    )


if __name__ == "__main__":
    main()
