import matplotlib.pyplot as plt
import json
import argparse
import os
import glob

def plot_combined_losses(input_paths, output_file, labels=None, title="Sweep Results Comparison"):
    """
    Plots validation loss from multiple metrics.json files.
    Two subplots: vs Steps and vs Time.
    """
    if not labels:
        labels = [os.path.basename(os.path.dirname(p)) for p in input_paths]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
    
    # Use a nice color cycle
    colors = plt.cm.tab10.colors
    
    for i, path in enumerate(input_paths):
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        history = data.get('history', {})
        steps = history.get('steps', [])
        times = history.get('elapsed_times', [])
        losses = history.get('val_losses', [])
        
        if not steps or not losses:
            print(f"Skipping file with no history: {path}")
            continue
            
        label = labels[i]
        color = colors[i % len(colors)]
        
        # Plot vs Steps
        ax1.plot(steps, losses, label=label, marker='o', markersize=4, color=color, linewidth=2, alpha=0.8)
        
        # Plot vs Time (if available)
        if times:
            ax2.plot(times, losses, label=label, marker='o', markersize=4, color=color, linewidth=2, alpha=0.8)

        # Plot zoomed (steps > 200, loss < 5.5)
        zoom_indices = [idx for idx, s in enumerate(steps) if s >= 200]
        if zoom_indices:
            z_steps = [steps[idx] for idx in zoom_indices]
            z_losses = [losses[idx] for idx in zoom_indices]
            ax3.plot(z_steps, z_losses, label=label, marker='o', markersize=5, color=color, linewidth=2.5, alpha=0.9)
    
    # Configure Subplot 1 (Steps)
    ax1.set_title("Validation Loss vs Steps")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.legend(fontsize='small', loc='upper right')
    
    # Configure Subplot 2 (Time)
    ax2.set_title("Validation Loss vs Time")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Loss")
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.legend(fontsize='small', loc='upper right')

    # Configure Subplot 3 (Zoomed)
    ax3.set_title("Validation Loss vs Steps (Zoomed Final)")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Loss")
    ax3.grid(True, which="both", ls="-", alpha=0.3)
    # Set limits for zoom
    ax3.set_ylim(4.7, 5.3) 
    ax3.legend(fontsize='small', loc='upper right')
    
    # Add final loss annotations if not too crowded
    for i, path in enumerate(input_paths):
         with open(path, 'r') as f:
            history = json.load(f).get('history', {})
            steps = history.get('steps', [])
            losses = history.get('val_losses', [])
            if steps and losses:
                ax1.annotate(f"{losses[-1]:.3f}", (steps[-1], losses[-1]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color=colors[i % len(colors)])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=150)
    print(f"Combined plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate combined plots for multiple experiments")
    parser.add_argument("--files", nargs='+', help="Paths to metrics.json files", required=True)
    parser.add_argument("--output", default="combined_comparison.png", help="Output image file")
    parser.add_argument("--labels", nargs='+', help="Custom labels for the legend")
    parser.add_argument("--title", default="LR/WD Sweep Results Comparison (8M Tokens)", help="Overall plot title")
    
    args = parser.parse_args()
    
    plot_combined_losses(args.files, args.output, args.labels, args.title)
