import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_combined_losses(checkpoint_dirs, output_file):
    plt.figure(figsize=(10, 6))
    
    for label, d in checkpoint_dirs.items():
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
                plt.plot(train_steps, train_losses, label=label, alpha=0.8)
            else:
                print(f"Warning: No train_losses in {metrics_path}")

    plt.xlabel("Steps")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison: SwiGLU vs Squared ReLU")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log') # Losses often look better on log scale
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    configs = {
        "SwiGLU (Baseline)": "checkpoints/baseline_swiglu_run_1",
        "Squared ReLU (Primer)": "checkpoints/primer_sqrelu_run_1"
    }
    plot_combined_losses(configs, "comparison_loss_plot.png")
