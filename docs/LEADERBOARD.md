# 🏆 Blueberry-Nano Speedrun Leaderboard

> Training is run on 1x4090 RTX.

## ⚡ Fastest To 6.7 Train Loss
*Goal: Fastest Time to Reach Loss ≤ 6.7*
> Everyone is GPU poor, let's make every FLOP count.

| # | Date | Time | Tokens Used | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 2025-12-18 | **22.24s** | **385,024** | [Vuk Rosić](https://x.com/VukRosic99) | Optimized Config (LR 0.015, Warmup 0, Constant, GradAcc 1) + [Per-step check] |
| **2** | 2025-12-17 | 1m 4s | 2,457,600 | [Vuk Rosić](https://x.com/VukRosic99) | Optimized Baseline (LR 0.02, GradAcc 1) |

## 🎯 Best Loss @ 67M Tokens
*Goal: Lowest Validation Loss after 67,000,000 Training Tokens*
> Once you go big, data is the bottleneck.

| # | Date | Val Loss | Time | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **-** | 2025-12-17 | TBD | TBD | [Vuk Rosić](https://x.com/VukRosic99) | Baseline (In Progress) |

## 🏅 The 1B Marathon (World Record)
*Goal: Best Model @ 1B Tokens (Time < 4h)*

| # | Date | Val Loss | Time | User | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 2025-12-17 | 3.1940 | 2h 37m | [Vuk Rosić](https://x.com/VukRosic99) | Final Milestone Record |


## 🤝 Compute Sponsorship & Verification
**Don't have a 4090? We've got you.**

1.  **Verification**: If you optimize the code but can't verify the exact time, submit a Pull Request. We will run your code on our standardized hardware to confirm the record!
2.  **Sponsorship**: If you have a great idea (e.g., a new architecture) but no GPU, open an Issue/Ticket. If the idea looks promising, we will run the experiment for you and credit you.