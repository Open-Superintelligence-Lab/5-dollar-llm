import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_combined_loss():
    log_dir = Path("logs")
    configs = [
        "BlueberryConfig",
        "BlueberryWideConfig",
        "BlueberryFFNConfig",
        "BlueberryB8Config",
        "BlueberryB8WideConfig",
        "BlueberryB8TallConfig",
    ]
    
    plt.figure(figsize=(14, 10))
    
    found_data = False
    for config in configs:
        # Search for metrics files in different experiment subdirectories
        metrics_files = list(log_dir.glob(f"{config}/bench_{config}_loss3.5/metrics.json"))
        metrics_files += list(log_dir.glob(f"{config}/bench_{config}_loss5/metrics.json"))
        metrics_files += list(log_dir.glob(f"{config}/bench_{config}_loss9/metrics.json"))
        
        for metrics_file in metrics_files:
            # Create a more readable label
            exp_type = ""
            if "loss3.5" in str(metrics_file): exp_type = " (Target 3.5)"
            elif "loss5" in str(metrics_file): exp_type = " (Target 5.0)"
            elif "loss9" in str(metrics_file): exp_type = " (Target 9.0)"
            
            experiment_label = f"{config}{exp_type}"
            
            try:
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                
                history = data.get("history", {})
                # Use train_losses and train_elapsed_times if available
                losses = history.get("train_losses", [])
                times = history.get("train_elapsed_times", [])
                
                if not losses:
                    # Fallback to val_losses if train_losses not found
                    losses = history.get("val_losses", [])
                    times = [t * 60 for t in history.get("elapsed_times", [])] # Convert mins to secs
                    
                if losses:
                    # Line styles based on target loss
                    style = '-'
                    if "loss5" in str(metrics_file): style = '--'
                    if "loss9" in str(metrics_file): style = ':'
                    
                    plt.plot(times, losses, label=experiment_label, linewidth=2, linestyle=style, alpha=0.8)
                    found_data = True
                else:
                    print(f"No loss data found in {metrics_file}")
                    
            except Exception as e:
                print(f"Error loading {metrics_file}: {e}")

    if not found_data:
        print("No metrics found to plot. Wait for the benchmark to finish!")
        return

    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title("LLM Training Loss vs Time Comparison (All Experiments)", fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.yscale("log")
    
    # Adjust y-axis to focus on the interesting range
    plt.ylim(3.0, 11)
    
    output_path = "benchmarks/combined_loss_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Combined loss plot saved to {output_path}")

if __name__ == "__main__":
    plot_combined_loss()
