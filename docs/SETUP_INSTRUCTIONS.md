# 🚀 5-dollar-llm: Setup & Speedrun Guide

Welcome to the **5-dollar-llm** repository! This project is dedicated to pushing the limits of training efficiency for 151M parameter models on consumer hardware.

Whether you are a human researcher or an AI agent, this guide will help you set up the environment and start competing on the leaderboard.

---

## 🛠️ 1. Environment Setup

We recommend using **Python 3.10+**.

### Clone the Repository
```bash
git clone https://github.com/Open-Superintelligence-Lab/5-dollar-llm
cd 5-dollar-llm
```

### Install Dependencies
```bash
pip install -r requirements.txt
pip install huggingface_hub  # Required for data download
```

### Option A: Quick Start (40M Tokens) - Recommended for Speedruns
Perfect for hitting the 4.5 and 3.5 loss milestones. Downloads in seconds.
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 40M Token Subset...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain', split='train[:20000]')
os.makedirs('processed_data/speedrun_40M', exist_ok=True)
ds.save_to_disk('processed_data/speedrun_40M')
print('✅ Speedrun Data Ready!')
"
```

### Option B: Full Dataset (1B Tokens) - Recommended for the Marathon
Required for the 1B token world record challenge.
```bash
python3 -c "
from datasets import load_dataset
import os
print('Downloading 1B Pretraining Data...')
ds = load_dataset('vukrosic/blueberry-1B-pretrain')
os.makedirs('processed_data/pretrain_1B', exist_ok=True)
ds.save_to_disk('processed_data/pretrain_1B')
print('✅ Full Data Ready!')
"
```

---

## 🏎️ 2. The Speedrun Leaderboard

Our community competes to reach specific training loss milestones in the shortest time possible on a single **NVIDIA RTX 4090**.

### ⚡ Speedrun 1: The 4.5 Loss Challenge
*   **Goal:** Reach a training loss of **≤ 4.5** as quickly as possible.
*   **Purpose:** Ideal for **quick architecture tests**, testing new optimizers, or rapid hyperparameter searches.
*   **Rules:** Must follow the [Official Speedrun Rules](LEADERBOARD.md#📜-official-rules).
*   *   **Expected Time:** ~2-3 minutes.
*   **Command:**
    ```bash
    python train_llm.py \
        --dataset_path processed_data/speedrun_40M \
        --target_train_loss 4.5 \
        --experiment_name arch_test_v1
    ```

### ⚡ The 3.5 Loss Speedrun
*   **Goal:** Reach a training loss of **≤ 3.5**.
*   **Purpose:** Used for **deeper research**. Smaller gains in the 4.5 speedrun are verified here to ensure they don't collapse or plateau early.
*   **Expected Time:** ~9 minutes.
*   **Command:**
    ```bash
    python train_llm.py \
        --dataset_path processed_data/speedrun_40M \
        --target_train_loss 3.5 \
        --experiment_name deep_research_v1
    ```

---

## 📊 3. Iterating & Research

- **Configs:** Modify `configs/llm_config.py` to change configs (keep the parameter size around 151M), learning rates, or optimization schedules.
- **Model:** Edit `models/llm.py` to experiment with new attention mechanisms or layer types.
- **Logs:** Check the `logs/` directory for detailed training metrics.
- **Leaderboard:** See `docs/LEADERBOARD.md` for current world records and submission instructions.


