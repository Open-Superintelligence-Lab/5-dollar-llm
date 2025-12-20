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
TARGET_LOSS = 4.5
NUM_RUNS = 3

def run_training(run_id):
    print(f"\n🚀 Starting Run {run_id}...")
    experiment_name = f"repro_run_{run_id}"
    cmd = [
        "python", "train_llm.py",
        "--target_train_loss", str(TARGET_LOSS),
        "--experiment_name", experiment_name,
        "--compile", "true",
        "--dataset_path", "processed_data/speedrun_40M"
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
            return {
                "run_id": run_id,
                "duration_s": data.get("active_training_time_seconds", 0),
                "steps": data.get("actual_steps", 0),
                "success": True
            }
    else:
        print(f"❌ Run {run_id} failed to produce metrics.json")
        return {"run_id": run_id, "success": False}

def main():
    results = []
    
    print(f"=== Reproducing Speedrun (Target Loss: {TARGET_LOSS}) ===")
    print(f"This script will train the model {NUM_RUNS} times to this loss so you can verify consistency.")
    
    for i in range(1, NUM_RUNS + 1):
        res = run_training(i)
        if res["success"]:
            results.append(res)
            
    if not results:
        print("No successful runs.")
        return
        
    durations = [r["duration_s"] for r in results]
    steps = [r["steps"] for r in results]
    
    print("\n" + "="*50)
    print("📈 STATISTICAL SUMMARY")
    print("="*50)
    print(f"Number of runs: {len(results)}")
    
    if len(durations) > 1:
        mean_dur = statistics.mean(durations)
        std_dur = statistics.stdev(durations)
        print(f"Time:        Mean = {format_time(mean_dur)}, StdDev = {std_dur:.2f}s")
        print(f"Steps:       Mean = {statistics.mean(steps):.2f}, StdDev = {statistics.stdev(steps):.2f}")
    else:
        print(f"Time:        {format_time(durations[0])}")
        print(f"Steps:       {steps[0]}")
        
    print("\nDetailed Results:")
    for r in results:
        print(f"  Run {r['run_id']}: {format_time(r['duration_s'])}, {r['steps']} steps")

if __name__ == "__main__":
    main()
