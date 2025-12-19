# Benchmarking script for FFN types
import os

import subprocess
import json
import time
import statistics
from utils.helpers import format_time

def run_training(run_id, ff_type, d_ff, experiment_prefix):
    print(f"\n🚀 Starting Run {run_id} ({ff_type}, d_ff={d_ff})...")
    experiment_name = f"{experiment_prefix}_run_{run_id}"
    cmd = [
        "python", "train_llm.py",
        "--target_train_loss", "4.5",
        "--experiment_name", experiment_name,
        "--ff_type", ff_type,
        "--d_ff", str(d_ff),
        "--compile", "true",
        "--dataset_path", "processed_data/speedrun_40M"
    ]
    
    # Small delay between runs
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
    runs_per_config = 1
    
    configs = [
        {"name": "SwiGLU (Baseline)", "ff_type": "swiglu", "d_ff": 2048, "prefix": "baseline_swiglu"},
        {"name": "Squared ReLU (Primer)", "ff_type": "squared_relu", "d_ff": 3072, "prefix": "primer_sqrelu"}
    ]
    
    all_results = {}
    
    print("=== 🔬 FFN Type Comparison Benchmark ===")
    print(f"Goal: Reach 4.5 loss as fast as possible ({runs_per_config} runs each)")
    
    for config in configs:
        print(f"\n--- Benchmarking {config['name']} ---")
        results = []
        for i in range(1, runs_per_config + 1):
            res = run_training(i, config["ff_type"], config["d_ff"], config["prefix"])
            if res["success"]:
                results.append(res)
        all_results[config["name"]] = results

    print("\n" + "="*70)
    print("📊 FINAL BENCHMARK COMPARISON")
    print("="*70)
    
    summary_data = []
    
    for name, results in all_results.items():
        if not results:
            print(f"{name}: No successful runs.")
            continue
            
        durations = [r["duration_s"] for r in results]
        steps = [r["steps"] for r in results]
        mean_dur = statistics.mean(durations)
        mean_steps = statistics.mean(steps)
        
        summary_data.append({
            "name": name,
            "mean_dur": mean_dur,
            "mean_steps": mean_steps
        })
        
        if len(durations) > 1:
            std_dur = statistics.stdev(durations)
            print(f"{name: <25} | Time: {format_time(mean_dur)} (±{std_dur:.2f}s) | Steps: {mean_steps:.2f}")
        else:
            print(f"{name: <25} | Time: {format_time(durations[0])} | Steps: {steps[0]}")

    if len(summary_data) == 2:
        baseline = summary_data[0]
        experiment = summary_data[1]
        speedup = (baseline["mean_dur"] - experiment["mean_dur"]) / baseline["mean_dur"] * 100
        print("-" * 70)
        print(f"🚀 {experiment['name']} is {speedup:.1f}% faster than {baseline['name']}")
        print("="*70)

if __name__ == "__main__":
    main()
