import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_combined_losses(checkpoint_dirs, output_file, window=20):
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e'] # Blue and Orange
    
    for i, (label, d) in enumerate(checkpoint_dirs.items()):
        metrics_path = Path(d) / "metrics.json"
        if not metrics_path.exists():
            print(f"Warning: {metrics_path} not found")
            continue
            
        with open(metrics_path, "r") as f:
            data = json.load(f)
            history = data.get("history", {})
            train_losses = history.get("train_losses", [])
            train_steps = history.get("train_steps", [])
            
            if train_losses:
                color = colors[i % len(colors)]
                # Plot raw data (faint)
                plt.plot(train_steps, train_losses, color=color, alpha=0.2, linewidth=1)
                
                # Plot smoothed data (bold)
                if len(train_losses) > window:
                    smoothed = moving_average(train_losses, window)
                    smoothed_steps = train_steps[window-1:]
                    plt.plot(smoothed_steps, smoothed, color=color, label=f"{label} (smoothed)", linewidth=2)
                else:
                    plt.plot(train_steps, train_losses, color=color, label=label, linewidth=2)
            else:
                print(f"Warning: No train_losses in {metrics_path}")

    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title("Training Loss Comparison (Smoothed Window=20)", fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Smoothed plot saved to {output_file}")

if __name__ == "__main__":
    configs = {
        "SwiGLU (Baseline)": "checkpoints/baseline_swiglu_run_1",
        "Squared ReLU (Primer)": "checkpoints/primer_sqrelu_run_1"
    }
    plot_combined_losses(configs, "comparison_loss_plot.png")
