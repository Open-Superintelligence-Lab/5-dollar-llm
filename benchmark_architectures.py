import subprocess
import os
import json
import time
import torch
import importlib
from pathlib import Path
import pandas as pd
from models.llm import MinimalLLM

def count_parameters(config_path):
    try:
        module_name, class_name = config_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        ConfigClass = getattr(module, class_name)
        config = ConfigClass()
        # Initialize model on CPU just for param counting
        model = MinimalLLM(config)
        return sum(p.numel() for p in model.parameters()) / 1e6
    except Exception as e:
        print(f"Error counting parameters for {config_path}: {e}")
        return 0

def run_benchmark():
    configs = [
        "configs.llm_config.BlueberryConfig",
    ]
    
    target_loss = 3.5
    results = []
    
    os.makedirs("benchmarks", exist_ok=True)
    
    for config_path in configs:
        config_name = config_path.split(".")[-1]
        param_count = count_parameters(config_path)
        
        print(f"\n" + "="*50)
        print(f"🚀 Benchmarking {config_name} ({param_count:.1f}M params)")
        print("="*50)
        
        output_dir = f"logs/{config_name}"
        
        cmd = [
            "python", "train_llm.py",
            "--config_class", config_path,
            "--target_train_loss", str(target_loss),
            "--experiment_name", f"bench_{config_name}_loss3.5",
            "--output_dir", output_dir,
            "--train_tokens", "500000000", 
            "--eval_every", "1000",
            "--log_every", "100",
            "--warmup", "false", 
        ]
        
        start_time = time.time()
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error training {config_name}: {e}")
            continue
        
        total_time = time.time() - start_time
        
        # Load metrics
        metrics_file = Path(output_dir) / f"bench_{config_name}_loss3.5" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                data = json.load(f)
                
            results.append({
                "Architecture": config_name,
                "Params (M)": round(param_count, 1),
                "Max VRAM (GB)": round(data.get("max_memory_reserved_gb", 0), 2),
                "Target Loss": target_loss,
                "Final Loss": data["final_metrics"].get("val_loss", "N/A"),
                "Steps": data.get("actual_steps", "N/A"),
                "Training Time (s)": data.get("active_training_time_seconds", "N/A"),
                "Total Time (s)": total_time,
            })
        else:
            print(f"⚠️ Metrics file not found for {config_name}")

    # Display results
    if results:
        df = pd.DataFrame(results)
        print("\n" + "#"*50)
        print(f"📊 BENCHMARK SUMMARY (Target Loss: {target_loss})")
        print("#"*50)
        print(df.to_string(index=False))
        
        df.to_csv("benchmarks/summary_final_b8.csv", index=False)
        print(f"\nSummary saved to benchmarks/summary_final_b8.csv")
    else:
        print("No results to display.")

if __name__ == "__main__":
    run_benchmark()
