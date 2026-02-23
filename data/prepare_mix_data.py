import argparse
import os
import sys
import random
import json
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Add parent directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.llm_config import BlueberryConfig

def process_stream(name, dataset, tokenizer, target_tokens, f_handle, chunk_size):
    print(f"  Processing {name} for {target_tokens:,} tokens...")
    iterator = iter(dataset)
    current_tokens = 0
    token_buffer = []
    
    while current_tokens < target_tokens:
        try:
            text = next(iterator)['text']
            ids = tokenizer.encode(text, add_special_tokens=True)
            token_buffer.extend(ids)
            current_tokens += len(ids)
            
            # Flush complete chunks
            while len(token_buffer) >= chunk_size:
                chunk = token_buffer[:chunk_size]
                json.dump({"input_ids": chunk, "labels": chunk}, f_handle)
                f_handle.write("\n")
                token_buffer = token_buffer[chunk_size:]
                
            if current_tokens % 1_000_000 < 5_000:
                print(f"    {name}: {current_tokens:,} / {target_tokens:,}", end='\r')
                
        except StopIteration:
            print(f"    {name} ran out of data!")
            break
            
    print(f"    {name}: Collected {current_tokens:,} tokens.")
    return current_tokens

def prepare_pretraining_data(args):
    # Get max_seq_len from config (can be overridden by args)
    config = BlueberryConfig()
    chunk_size = args.max_seq_len if args.max_seq_len else config.max_seq_len
    
    print(f"🚀 Preparing pre-training data (Target: {args.target_tokens:,} tokens) [Sequential + Shuffle]...")
    print(f"📏 Using chunk_size (max_seq_len): {chunk_size}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    output_path = os.path.join(args.output_dir, f"pretrain_mix_{args.target_tokens}")
    jsonl_path = os.path.join(args.output_dir, "temp_mix.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Ratios - 70% FineWeb-Edu, 30% Cosmopedia
    target_fw = int(args.target_tokens * 0.7)
    target_cosmo = int(args.target_tokens * 0.3)
    
    with open(jsonl_path, "w") as f:
        # 1. FineWeb-Edu
        ds_fineweb = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
        process_stream("FineWeb-Edu", ds_fineweb, tokenizer, target_fw, f, chunk_size)
        
        # 2. Cosmopedia
        ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
        process_stream("Cosmopedia", ds_cosmo, tokenizer, target_cosmo, f, chunk_size)
        
        # 3. Python-Edu removed as requested
        
    print("\nLoading JSONL...")
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    
    print("Global Shuffling...")
    ds = ds.shuffle(seed=42)
    
    print(f"Saving to {output_path}...")
    ds.save_to_disk(output_path)
    
    # Save metadata about preparation parameters
    metadata_path = os.path.join(output_path, "prep_metadata.json")
    metadata = {
        "max_seq_len": chunk_size,
        "target_tokens": args.target_tokens,
        "tokenizer_name": args.tokenizer_name,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Saved preparation metadata: max_seq_len={chunk_size}")

    if args.push_to_hub:
        if not args.repo_id:
            print("❌ Error: --repo_id is required when --push_to_hub is used.")
        else:
            print(f"📤 Pushing to Hugging Face Hub: {args.repo_id}...")
            ds.push_to_hub(args.repo_id, private=args.private)
            print("✅ Push complete!")
    
    print("Cleaning up...")
    os.remove(jsonl_path)
    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_tokens", type=int, default=22_000_000, help="Number of tokens to prepare (e.g., 8M, 20M, 100M, 1B, 2B)")
    parser.add_argument("--output_dir", type=str, default="./processed_data", help="Output directory")
    parser.add_argument("--tokenizer_name", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Tokenizer")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Max sequence length (defaults to config value)")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the processed dataset to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, help="Hugging Face repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--private", action="store_true", help="Make the Hugging Face repository private")
    
    args = parser.parse_args()
    prepare_pretraining_data(args)
