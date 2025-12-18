# 🏆 Blueberry-Nano Speedrun Leaderboard

> Training is run on 1x4090 RTX.

## 📜 Official Rules

To qualify for the **Speedrun** (4.5 loss / 3.5 loss / 1B tokens) leaderboard, your run must follow these rules:

1.  **Target:** Surpass the record (training loss of **≤ 4.5** on the first speedrun, training loss of **≤ 3.5** on the second speedrun, fastest training time to **1B tokens** on the third speedrun).
2.  **Model Size:** Your model must have **151M ± 5%** total parameters (143.4M to 158.5M).
3.  **Data:** Use the `processed_data/speedrun_40M` dataset (generated from the first 20,000 samples of `vukrosic/blueberry-1B-pretrain`).
4.  **Hardware:** Records are officially verified on a single **NVIDIA RTX 4090**.
5.  **Timing:** The official metric is **Active Training Time**. Setup and compilation overhead (~85s-150s) can be excluded by using the `--warmup true` flag, which performs an untimed warmup with dummy data.
6.  Keep the added code minimal, clean and readable.

> [!TIP]
> `torch.compile` is highly recommended for speed, but adds initial latency. Your results will show `Setup & Compilation` separately from `Active Training` to ensure fair benchmarking.



## ⚡ Fastest To 4.5 Train Loss
*Goal: Fastest Time to Reach Loss ≤ 4.5*
> First benchmark is faster to experiment on. We can later find what transfers to the longer training.

| # | Date | Time | Tokens Used | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 2025-12-18 | **2m 04s** | **5,472,256** | [Vuk Rosić](https://x.com/VukRosic99) | Optimized Config (LR 0.015, Warmup 0, Constant, GradAcc 1) + [Per-step check] |

> **Noise**: New record should be at least 1 second fater or it could be randomness.




## ⚡ Fastest To 3.5 Train Loss
*Goal: Fastest Time to Reach Loss ≤ 3.5*

| # | Date | Time | Tokens Used | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 2025-12-18 | **6m 47s** | **17,539,072** | [Vuk Rosić](https://x.com/VukRosic99) | Optimized Config (LR 0.015, Warmup 0, Constant, GradAcc 1) + [Per-step check] |

## 🗂️ More categories coming soon
- You may suggest: goal is to interpolate between fast experimentation and confirming it works on big models.

## 🏅 The 1B Marathon (World Record)
*Goal: Best Model @ 1B Tokens (Time < 4h)*

| # | Date | Val Loss | Time | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 2025-12-17 | 3.1940 | 2h 37m | [Vuk Rosić](https://x.com/VukRosic99) | Final Milestone Record |


## 🤝 Compute Sponsorship & Verification
**You may rent 4090 affordably at**
[Salad](https://salad.com/pricing) | [Novita](https://novita.ai/pricing?gpu=1) | [VastAI](https://vast.ai/pricing) - A lot of GPU providers also give 50% off on spot billing.

You may also use free L4 at [LightningAI](https://lightning.ai/) - just make sure to measure your baseline as well and then compare. Once you break the record, we will measure it on 4090.

**Can't access a GPU? We've got you.**

1.  **Verification**: If you optimize the code but can't verify the exact time, submit a Pull Request. We will run your code on our 4090 to confirm the record!
2.  **Sponsorship**: If you have a great idea (e.g., a new architecture) but no GPU, open an Issue/Ticket. If the idea looks promising, we will run the experiment for you and credit you.