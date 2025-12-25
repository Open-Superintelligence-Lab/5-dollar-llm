# 🏆 Blueberry-Nano Speedrun Leaderboard

## 📜 Official Rules

Please read [SETUP_INTRUCTIONS](docs/SETUP_INSTRUCTIONS.md) for detailed guide.

## ⚡ 8M Tokens Speedrun
*Goal: Fastest Time to train 8M tokens*

| # | Date | Train Loss | Val Loss | Time | Tokens Used | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2025-12-21 | 4.7487 | 4.8466 | 1m 44s 79ms | 8,011,776 | [Vuk Rosić](https://x.com/VukRosic99) | Hyperparam search: batch size doubled 4 to 8, n_layers 32 to 22 to fit into memory, muon lr 0.015 to 0.024 and adamw_lr from 0.001 to 0.006 |
| 2 | 2025-12-22 | 4.7479 | 4.8467 | 1m 29s 209ms | 8,011,776 | [Vuk Rosić](https://x.com/VukRosic99) | Squared ReLU instead of SwiGLU, one less linear layer in feedforward |
| 3 | 2025-12-22 | 4.7286 | 4.8363 | 1m 28s 664ms | 8,011,776 | [GitHub](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/56) [ToheedAkhtar01](https://x.com/ToheedAkhtar01) | Polar Muon - it replaces Muon’s Newton-Schulz iteration with a fixed-coefficient iterative scheme for faster, numerically stable orthogonalization. |
| 4 | 2025-12-23 | 4.7333 | 4.8366 | 1m 27s 856ms | 8,011,776 | [GitHub](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/67/) | Fused AdamW |
| 5 | 2025-12-23 | 4.7409 | 4.8403 | 1m 26s 178ms | 8,011,776 | [bigwolfeman](https://github.com/bigwolfeman) | Cast model into bf16 - model = model.to(device, dtype=torch.bfloat16), Note: Optimizers might require higher precision for longer runs |
| 5 (new eval) | 2025-12-24 | 4.7408 | 4.8387 | 1m 59s 44ms | 8,011,776 | [bigwolfeman](https://github.com/bigwolfeman) | Included evaluations during training to plot loss curve. Training setup unchanged from #5. |

> **Record Repeatability / Noise**:
- Run 1: 1m 27s 856ms, Train Loss: 4.7333, Val Loss: 4.8366
- Run 2: 1m 28s 275ms, Train Loss: 4.7397, Val Loss: 4.8373

⚠️ If you are unable to reproduce our results on RTX 4090, you may have different CPU, PCIe Bandwidth, or Thermal Throttling. We always recommend measuring your baseline first then comparing against your changes. We measure on Novita AI 4090 with Intel(R) Xeon(R) Platinum 8473C CPU. The CPU selection is random so it requires multiple tries.


## ⚡ 20M Tokens Speedrun
*Goal: Fastest Time to train 20M tokens*

| # | Date | Train Loss | Val Loss | Time | Tokens Used | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2025-12-22 | 4.2004 | 4.2021 | 4m 8s 168ms | 20,004,864 | [Vuk Rosić](https://x.com/VukRosic99) | Hyperparam search: batch size doubled 4 to 8, n_layers 32 to 22 to fit into memory, muon lr 0.015 to 0.024 and adamw_lr from 0.001 to 0.006 |
| 2 | 2025-12-22 | 4.2118 | 4.2087 | 3m 32s 156ms | 20,004,864 | [Vuk Rosić](https://x.com/VukRosic99) | Squared ReLU instead of SwiGLU, one less linear layer in feedforward |
| 3 | 2025-12-22 | 4.1952 | 4.2056 | 3m 29s 308ms | 20,004,864 | [ToheedAkhtar01](https://x.com/ToheedAkhtar01) [GitHub](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/56) | Polar Muon - it replaces Muon’s Newton-Schulz iteration with a fixed-coefficient iterative scheme for faster, numerically stable orthogonalization. |
| 4 | 2025-12-23 | 4.2049 | 4.2075 | 3m 28s 591ms | 20,004,864 | [GitHub](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/67/) | Fused AdamW |
| 5 | 2025-12-23 | 4.1701 | 4.1791 | 3m 19s 165ms | 20,004,864 | [bigwolfeman](https://github.com/bigwolfeman) | Cast model into bf16 - model = model.to(device, dtype=torch.bfloat16), Note: Optimizers might require higher precision for longer runs |
| 5 (new eval) | 2025-12-24 | 4.1631 | 4.1756 | 3m 50s 276ms | 20,004,864 | [bigwolfeman](https://github.com/bigwolfeman) | Included evaluations during training to plot loss curve. Training setup unchanged from #5. |

> **Record Repeatability / Noise**:
- Run 1: 3m 28s 591ms, Train Loss: 4.2049, Val Loss: 4.2075
- Run 2: 3m 28s 871ms, Train Loss: 4.2049, Val Loss: 4.2075

## ⚡ 100M Tokens Speedrun
*Goal: Fastest Time to train 100M tokens*

| # | Date | Train Loss | Val Loss | Time | Tokens Used | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2025-12-22 | 3.7212 | 3.7492 | 20m 27s 988ms | 100,007,936 | [Vuk Rosić](https://x.com/VukRosic99) | Hyperparam search: batch size doubled 4 to 8, n_layers 32 to 22 to fit into memory, muon lr 0.015 to 0.024 and adamw_lr from 0.001 to 0.006 |
| 2 | 2025-12-22 | 3.7370 | 3.7526 | 17m 27s 59ms | 100,007,936 | [Vuk Rosić](https://x.com/VukRosic99) | Squared ReLU instead of SwiGLU, one less linear layer in feedforward |
| 3 | 2025-12-22 | 3.7439 | 3.7609 | 17m 8s 637ms | 100,007,936 | [ToheedAkhtar01](https://x.com/ToheedAkhtar01) [GitHub Polar](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/56); [GitHub AdamW](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/67/) | Fused AdamW; Polar Muon - it replaces Muon’s Newton-Schulz iteration with a fixed-coefficient iterative scheme for faster, numerically stable orthogonalization. |
| 4 | 2025-12-23 | 3.6700 | 3.7094 | 16m 17s 221ms | 100,007,936 | [bigwolfeman](https://github.com/bigwolfeman) | Cast model into bf16 - model = model.to(device, dtype=torch.bfloat16), Note: Optimizers might require higher precision for longer runs |
| 4 (new eval) | 2025-12-24 | 3.6568 | 3.7108 | 16m 44s 139ms | 100,007,936 | [bigwolfeman](https://github.com/bigwolfeman) | Included evaluations during training to plot loss curve. Training setup unchanged from #4. |

## 🏅 The 1B Marathon (World Record GPT-1)
*Goal: Best Model @ 1B Tokens (GPT-1)*

| # | Date | Train Loss | Val Loss | Time | Tokens Used | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 2025-12-23 | 3.4747 | 3.3580 | 2h 51m 31s | 1,000,007,680 | [Vuk Rosić](https://x.com/VukRosic99), [ToheedAkhtar01](https://x.com/ToheedAkhtar01), [GitHub #67](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/67/), [GitHub #56](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/pull/56) | n_layers 32→22, optimized LRs (Muon 0.024, AdamW 0.006), Squared ReLU, Fused AdamW, Polar Muon|
| 2 | 2025-12-23 | 3.4946 | 3.3583 | 2h 42m 49s | 1,000,007,680 | [bigwolfeman](https://github.com/bigwolfeman) | Cast model into bf16 - model = model.to(device, dtype=torch.bfloat16), Note: For 1B tokens higher precision in optimizer might be better. |