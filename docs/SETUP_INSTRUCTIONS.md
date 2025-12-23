# 🚀 5-dollar-llm: Setup & Speedrun Guide

Welcome to the **5-dollar-llm** repository! This project is dedicated to pushing the limits of training efficiency for a 88M parameter model on 1 billion tokens (GPT-1 level model).

---

## Step by step instructions

### Step 1: Select a GPU to train on.

If you don't have a GPU, you may use a cloud GPU.

#### Free GPUs:
- **Lightning AI**: You can use the free **L4 GPU**.
- **Google Colab**: Use the free T4 or paid A100.
- **Tip**: If the model doesn't fit in your GPU memory, you can **reduce the model size** (e.g., reduce `batch_size`, `n_layer`, or `n_embd` in `configs/llm_config.py`).

#### Paid GPUs:
- **You may rent a GPU affordably at**
[Salad](https://salad.com/pricing) | [Novita](https://novita.ai/pricing?gpu=1) [(or use our affiliate to help us get more compute ❤️)](https://novita.ai/?ref=mjqyndm&utm_source=affiliate) | [VastAI](https://vast.ai/pricing) - A lot of GPU providers give 50% off on spot billing.

You may watch our tutorial on the [AI Research Setup](https://youtu.be/FXUMISiMOTE).



### 🛠️ 1. Environment Setup

We recommend using **Python 3.10+**.

### Clone the Repository
```bash
git clone https://github.com/Open-Superintelligence-Lab/5-dollar-llm
cd 5-dollar-llm
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Option A: Quick Start (40M Tokens) - Most of the time you will just need to download this dataset
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

### Option B: If you train on 100M or 1B tokens, first read below
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


### Step 2: Measure the baseline

You need to know how our (current) code performs on your hardware before changing it, so you can measure the impact of your changes.

- This is done by simply running `python train_llm.py`
- After it finishes running, please run it again.
- Keep note of `Training Time (⏱️ Speedrun):` and final `Final Val Loss` from the second run.
- You may notice that these 2 runs give different training time, even though they execute the exact same code. This is normal, and it is because the first run will build / compile the model, the second run is what you need to beat. If you can solve this issue so it compiles graphs if needed with just a single run, and doesn't add that to the training time, please make a pull request.

### Step 3:
Now that you have the exact time you need to beat, you can start making changes.

If you ran `python train_llm.py` as mentioned above, you trained the model on 8 million tokens (default).

Currently we have 4 benchmarks:
- 8,000,000 Tokens
- 20,000,000 Tokens
- 100,000,000 Tokens
- 1,000,000,000 Tokens

Just an improvement on 1 benchmark is enough to submit, but you may measure multiple.

### Step 4 (optional):

If you wish to try 20M tokens, please run `python train_llm.py --train_tokens 20000000`.

W are not yet sure if you need to rerun it 2 times after you have already built the graphs with 8M tokens. We are working on this. As a safe bet, we recommend running the baseline on 20M 2 times as well and checking the last results.

Same goes for 100M and 1B tokens but make sure you have the full 1 billion token dataset downloaded.

### Step 5:

Add your code changes.

- Only make a single change at a time and train the model to measure the impact of it. If the resulting time is a lot slower than the baseline, your changes may have broken the torch graph so you will have to run it a second time to get the real results.
- Do not combine multiple experiments into one (eg. learning rate, fused adam, attention heads, etc.) because you will not know what caused improvement and what caused regression.

### Step 6:

Confirm that your changes ourperform baseline - check the `Training Time (⏱️ Speedrun):` & final `Final Val Loss`.

Create a pull request on GitHub into main branch.

Once you submit your changes, we will mesures them ourselves, and if they improve performance, we will add you to the [leaderboard](LEADERBOARD.md#📜-official-rules) - you can leave your X / LinkedIn / GitHub / etc. in the pull request.



## 📊 Iterating & Research

- **Configs:** Modify `configs/llm_config.py` to change configs (keep the parameter size around 88M), learning rates, or optimization schedules.
- **Model:** Edit `models/llm.py` to experiment with new attention mechanisms or layer types.
- **GPU Memory:** If the model doesn't fit on your GPU, you can reduce the model size (e.g., `batch_size` or `n_layer`) for faster local iteration.


