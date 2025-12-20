import subprocess
import os

def run_ablation():
    muon_lr = 0.03
    # Fine-grained AdamW sweep around the high-speed 0.0054-0.0063 range
    adamw_lrs = [0.0051, 0.0054, 0.0057, 0.0060, 0.0063, 0.0066]
    
    target_loss = 4.5
    output_base_dir = "./checkpoints/adam_ablation"
    
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for alr in adamw_lrs:
        experiment_name = f"muon_0.03_adam_{alr:.5f}"
        print("\n" + "="*50)
        print(f"Starting ablation: {experiment_name}")
        print(f"Muon LR: {muon_lr}, AdamW LR: {alr:.6f}")
        print("="*50 + "\n")
        
        cmd = [
            "python", "train_llm.py",
            "--muon_lr", str(muon_lr),
            "--adamw_lr", f"{alr:.6f}",
            "--target_train_loss", str(target_loss),
            "--experiment_name", experiment_name,
            "--output_dir", output_base_dir,
            "--log_every", "10"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Experiment {experiment_name} failed with error: {e}")
            continue

if __name__ == "__main__":
    run_ablation()
