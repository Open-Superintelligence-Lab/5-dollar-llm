# 🏆 Blueberry-Nano Speedrun Leaderboard

> Training is run on 1x4090 RTX.

Usually we will do research together to be able to beat records, but you may also do it alone.

## 📜 Official Rules

To qualify for the **Speedrun** (4.5 loss / 3.5 loss / 1B tokens) leaderboard, your run must follow these rules:

1.  Surpass the record (training loss of **≤ 4.5**, training loss of **≤ 3.5**, or fastest training time on **8M tokens** / **1B tokens**).
2.  Use the data mentioned in the [SETUP_INTRUCTIONS](docs/SETUP_INSTRUCTIONS.md)
3.  The official metric is **Active Training Time**. Setup and compilation overhead (`Setup & Compilation Time`) is excluded.
4.  Measure your baseline (current code on your hardware) and compare your improvements against that baseline. Explain it to the PR description concisely.
5.  Keep the added code minimal, clean and readable.

## ⚡ 8M Tokens Speedrun
*Goal: Fastest Time to train 8M tokens*

| # | Date | Train Loss | Val Loss | Time | Tokens Used | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2025-12-21 | 4.7487 | 4.8466 | 1m 44s 79ms | 8,011,776 | [Vuk Rosić](https://x.com/VukRosic99) | Hyperparam search: batch size doubled 4 to 8, n_layers 32 to 22 to fit into memory, muon lr 0.015 to 0.024 and adamw_lr from 0.001 to 0.006 |
| 2 | 2025-12-22 | 4.7479 | 4.8467 | 1m 29s 209ms | 8,011,776 | [Vuk Rosić](https://x.com/VukRosic99) | Squared ReLU instead of SwiGLU, one less linear layer in feedforward |
| 3 | 2025-12-22 | 4.7286 | 4.8363 | 1m 28s 664ms | 8,011,776 | [GitHub](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/56) [ToheedAkhtar01](https://x.com/ToheedAkhtar01) | Polar Muon - it replaces Muon’s Newton-Schulz iteration with a fixed-coefficient iterative scheme for faster, numerically stable orthogonalization. |

> **Record Repeatability / Noise**:
- Run 1: 1m 28s 664ms, 489 steps, Train Loss: 4.7286, Val Loss: 4.8363
- Run 2: 1m 28s 312ms, 489 steps, Train Loss: 4.7172, Val Loss: 4.8320
- Run 3: 1m 28s 175ms, 489 steps, Train Loss: 4.7314, Val Loss: 4.8397
- Run 4: 1m 28s 546ms, 489 steps, Train Loss: 4.7347, Val Loss: 4.8377
- Run 5: 1m 28s 458ms, 489 steps, Train Loss: 4.7325, Val Loss: 4.8373

⚠️ If you are unable to reproduce our results on RTX 4090, you may have different CPU, PCIe Bandwidth, or Thermal Throttling. We always recommend measuring your baseline first then comparing against your changes. We measure on Novita AI 4090 with Intel(R) Xeon(R) Platinum 8473C CPU. The CPU selection is random so it requires multiple tries.


## ⚡ 20M Tokens Speedrun
*Goal: Fastest Time to train 20M tokens*

| # | Date | Train Loss | Val Loss | Time | Tokens Used | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2025-12-22 | 4.2004 | 4.2021 | 4m 8s 168ms | 20,004,864 | [Vuk Rosić](https://x.com/VukRosic99) | Hyperparam search: batch size doubled 4 to 8, n_layers 32 to 22 to fit into memory, muon lr 0.015 to 0.024 and adamw_lr from 0.001 to 0.006 |
| 2 | 2025-12-22 | 4.2118 | 4.2087 | 3m 32s 156ms | 20,004,864 | [Vuk Rosić](https://x.com/VukRosic99) | Squared ReLU instead of SwiGLU, one less linear layer in feedforward |
| 3 | 2025-12-22 | 4.1952 | 4.2056 | 3m 29s 308ms | 20,004,864 | [ToheedAkhtar01](https://x.com/ToheedAkhtar01) [GitHub](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/56) | Polar Muon - it replaces Muon’s Newton-Schulz iteration with a fixed-coefficient iterative scheme for faster, numerically stable orthogonalization. |

## ⚡ 100M Tokens Speedrun
*Goal: Fastest Time to train 100M tokens*

| # | Date | Train Loss | Val Loss | Time | Tokens Used | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2025-12-22 | 3.7212 | 3.7492 | 20m 27s 988ms | 100,007,936 | User | Hyperparam search: batch size doubled 4 to 8, n_layers 32 to 22 to fit into memory, muon lr 0.015 to 0.024 and adamw_lr from 0.001 to 0.006 |
| 2 | 2025-12-22 | 3.7370 | 3.7526 | 17m 27s 59ms | 100,007,936 | User | Squared ReLU instead of SwiGLU, one less linear layer in feedforward |


## 🏅 The 1B Marathon (World Record)
*Goal: Best Model @ 1B Tokens (Time < 4h)*

| # | Date | Val Loss | Time | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| - | - | - | - | - | - |


## 🤝 GPUs: Free & Paid
**You may rent 4090 affordably at**
[Salad](https://salad.com/pricing) | [Novita](https://novita.ai/pricing?gpu=1) [(or use our affiliate to help us get more compute ❤️)](https://novita.ai/?ref=mjqyndm&utm_source=affiliate) | [VastAI](https://vast.ai/pricing) - A lot of GPU providers also give 50% off on spot billing.

**Free GPU Alternatives:**
- **Lightning AI**: You can use the free **L4 GPU**.
- **Google Colab**: Use the free T4 or paid A100.
- **Tip**: If the model doesn't fit in your GPU memory, you can **reduce the model size** (e.g., reduce `batch_size`, `n_layer`, or `n_embd` in `configs/llm_config.py`).

Once you create improvement, we will measure it on 4090.