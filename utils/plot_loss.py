import matplotlib.pyplot as plt
import json
import argparse
import os

def plot_loss(metrics_file, output_file, title="Validation Loss Curve"):
    if not os.path.exists(metrics_file):
        print(f"Error: File not found {metrics_file}")
        return

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    if 'history' not in data:
        print("Error: 'history' key not found in metrics file")
        return
    
    history = data['history']
    steps = history.get('steps', [])
    val_losses = history.get('val_losses', [])
    
    if not steps or not val_losses:
        print("Error: incomplete data in history")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, val_losses, marker='o', linestyle='-', color='b', label='Val Loss')
    
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Annotate points
    for i, (x, y) in enumerate(zip(steps, val_losses)):
        plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot validation loss curve")
    parser.add_argument("metrics_file", help="Path to metrics.json")
    parser.add_argument("output_file", help="Path to save the plot image")
    parser.add_argument("--title", default="Validation Loss - 8M Tokens Benchmark", help="Plot title")
    
    args = parser.parse_args()
    
    plot_loss(args.metrics_file, args.output_file, args.title)
