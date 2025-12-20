import subprocess
import os

def run_search():
    muon_base_lr = 0.03
    adam_base_lr = 0.0063
    ratio = adam_base_lr / muon_base_lr  # 0.21

    # Fine-grained Muon search around 0.03
    muon_lrs = [0.024, 0.027, 0.030, 0.033, 0.036]
    
    target_loss = 4.5
    output_base_dir = "./checkpoints/lr_search"
    
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for lr in muon_lrs:
        adam_lr = lr * ratio
        experiment_name = f"muon_lr_{lr}_adam_lr_{adam_lr:.5f}"
        print("\n" + "="*50)
        print(f"Starting experiment: {experiment_name}")
        print(f"Muon LR: {lr}, AdamW LR: {adam_lr:.6f}")
        print("="*50 + "\n")
        
        cmd = [
            "python", "train_llm.py",
            "--muon_lr", str(lr),
            "--adamw_lr", f"{adam_lr:.6f}",
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
    run_search()
