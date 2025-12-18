# this script makes sure the speedrun training is reproducible


import subprocess
import json
import os
import time
import statistics

def run_training(run_id):
    print(f"\n🚀 Starting Run {run_id}...")
    experiment_name = f"repro_run_{run_id}"
    cmd = [
        "python", "train_llm.py",
        "--target_train_loss", "3.5",
        "--experiment_name", experiment_name,
        "--compile", "true",
        "--dataset_path", "processed_data/speedrun_40M"
    ]
    
    start_time = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    last_loss = None
    for line in process.stdout:
        print(line, end="")
        if "Loss:" in line:
            try:
                # Extract loss for logging purposes if needed
                parts = line.split("Loss: ")
                if len(parts) > 1:
                    last_loss = float(parts[1].split("|")[0].strip())
            except:
                pass
                
    process.wait()
    end_time = time.time()
    duration = end_time - start_time
    
    # Load metrics from json
    metrics_path = f"checkpoints/{experiment_name}/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
            # The trainer saves actual_steps and total_time_minutes
            # We can also calculate tokens-per-second etc.
            return {
                "run_id": run_id,
                "duration_ms": data.get("total_time_seconds", 0) * 1000 or data.get("total_time_minutes", 0) * 60000,
                "steps": data.get("actual_steps", 0),
                "success": True
            }
    else:
        print(f"❌ Run {run_id} failed to produce metrics.json")
        return {"run_id": run_id, "success": False}

def main():
    num_runs = 2
    results = []
    
    print(f"=== Reproducing Speedrun 1 ({num_runs} runs) ===")
    
    for i in range(1, num_runs + 1):
        res = run_training(i)
        if res["success"]:
            results.append(res)
            
    if not results:
        print("No successful runs.")
        return
        
    durations = [r["duration_ms"] for r in results]
    steps = [r["steps"] for r in results]
    
    print("\n" + "="*50)
    print("📈 STATISTICAL SUMMARY")
    print("="*50)
    print(f"Number of runs: {len(results)}")
    
    if len(durations) > 1:
        print(f"Time (ms):   Mean = {statistics.mean(durations):.0f}ms, StdDev = {statistics.stdev(durations):.2f}ms")
        print(f"Steps:       Mean = {statistics.mean(steps):.2f}, StdDev = {statistics.stdev(steps):.2f}")
    else:
        print(f"Time (ms):   {durations[0]:.0f}ms")
        print(f"Steps:       {steps[0]}")
        
    print("\nDetailed Results:")
    for r in results:
        print(f"  Run {r['run_id']}: {r['duration_ms']:.0f}ms, {r['steps']} steps")

if __name__ == "__main__":
    main()
