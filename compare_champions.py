import subprocess
import os

def run_comparison():
    experiments = [
        {"name": "muon_0.03_adam_0.0063", "muon_lr": 0.03, "adamw_lr": 0.0063},
        {"name": "muon_0.024_adam_0.0060", "muon_lr": 0.024, "adamw_lr": 0.0060},
    ]
    
    target_loss = 3.5
    # Increase train_tokens to ensure we reach the target loss
    train_tokens = 200000000  # 200M tokens
    output_base_dir = "./checkpoints/champion_comparison"
    
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for exp in experiments:
        experiment_name = exp["name"]
        print("\n" + "="*50)
        print(f"Starting comparison: {experiment_name}")
        print(f"Muon LR: {exp['muon_lr']}, AdamW LR: {exp['adamw_lr']}")
        print("="*50 + "\n")
        
        cmd = [
            "python", "train_llm.py",
            "--muon_lr", str(exp["muon_lr"]),
            "--adamw_lr", str(exp["adamw_lr"]),
            "--target_train_loss", str(target_loss),
            "--train_tokens", str(train_tokens),
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
    run_comparison()
