# this script makes sure the speedrun training is reproducible
# by running training NUM_RUNS times
# training time should be similar for all runs

import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import json
import time
import statistics
from utils.helpers import format_time

# Configuration
NUM_RUNS = 3
TARGET_TOKENS = 8000000

def run_training(run_id):
    print(f"\n🚀 Starting Run {run_id}...")
    experiment_name = f"repro_run_{run_id}"
    cmd = [
        "python", "train_llm.py",
        "--experiment_name", experiment_name,
        "--compile", "true",
        "--dataset_path", "processed_data/speedrun_40M",
        "--train_tokens", str(TARGET_TOKENS)
    ]
    
    # Small delay between runs to allow GPU to reach consistent state
    if run_id > 1:
        time.sleep(1)
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    
    # Load metrics from json
    metrics_path = f"checkpoints/{experiment_name}/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
            res = {
                "run_id": run_id,
                "duration_s": data.get("active_training_time_seconds", 0),
                "steps": data.get("actual_steps", 0),
                "val_loss": data.get("final_metrics", {}).get("val_loss", 0),
                "train_loss": data.get("final_metrics", {}).get("train_loss", 0),
                "success": True
            }
            print(f"✅ Run {run_id} complete: {format_time(res['duration_s'])} | {res['steps']} steps | Train Loss: {res['train_loss']:.4f} | Val Loss: {res['val_loss']:.4f}")
            return res
    else:
        print(f"❌ Run {run_id} failed to produce metrics.json")
        return {"run_id": run_id, "success": False}

def main():
    results = []
    
    print(f"=== Reproducing Speedrun (Target: {TARGET_TOKENS:,} Tokens) ===")
    print(f"This script will train the model {NUM_RUNS} times to verify consistency.")
    
    for i in range(1, NUM_RUNS + 1):
        res = run_training(i)
        if res["success"]:
            results.append(res)
            
    if not results:
        print("No successful runs.")
        return
        
    durations = [r["duration_s"] for r in results]
    steps = [r["steps"] for r in results]
    val_losses = [r["val_loss"] for r in results]
    train_losses = [r["train_loss"] for r in results]
    
    print("\n" + "="*50)
    print("📈 STATISTICAL SUMMARY")
    print("="*50)
    print(f"Number of runs: {len(results)}")
    
    if len(durations) > 1:
        mean_dur = statistics.mean(durations)
        std_dur = statistics.stdev(durations)
        print(f"Time:        Mean = {format_time(mean_dur)}, StdDev = {std_dur:.2f}s")
        print(f"Steps:       Mean = {statistics.mean(steps):.2f}, StdDev = {statistics.stdev(steps):.2f}")
        print(f"Train Loss:  Mean = {statistics.mean(train_losses):.4f}, StdDev = {statistics.stdev(train_losses):.4f}")
        print(f"Val Loss:    Mean = {statistics.mean(val_losses):.4f}, StdDev = {statistics.stdev(val_losses):.4f}")
    else:
        print(f"Time:        {format_time(durations[0])}")
        print(f"Steps:       {steps[0]}")
        print(f"Train Loss:  {train_losses[0]:.4f}")
        print(f"Val Loss:    {val_losses[0]:.4f}")
        
    print("\nDetailed Results:")
    for r in results:
        print(f"  Run {r['run_id']}: {format_time(r['duration_s'])}, {r['steps']} steps, Train Loss: {r['train_loss']:.4f}, Val Loss: {r['val_loss']:.4f}")

if __name__ == "__main__":
    main()
