#!/bin/bash

# run_math_sweep.sh
# Automates the verification of the article's mathematical derivations (Eq 16 & 17).

TOKENS=8000000
BASE_DIR="./checkpoints/sweep"
mkdir -p $BASE_DIR

echo "🚀 Starting Mathematical Sweep (8M Tokens)..."

# Experiment 1: Inverse Sqrt (Coupled) - The predicted "Optimal"
echo "--- Running Exp 1: Inverse Sqrt (Coupled) ---"
python train_llm.py --schedule_type inverse_sqrt --coupled_wd true --train_tokens $TOKENS --output_dir "$BASE_DIR/exp1_sqrt_coupled"

# Experiment 2: Inverse Sqrt (Constant WD) - Ablation of coupling
echo "--- Running Exp 2: Inverse Sqrt (Constant WD) ---"
python train_llm.py --schedule_type inverse_sqrt --coupled_wd false --train_tokens $TOKENS --output_dir "$BASE_DIR/exp2_sqrt_constant"

# Experiment 3: Inverse Time (Constant WD) - Eq 16
echo "--- Running Exp 3: Inverse Time (Constant WD) ---"
python train_llm.py --schedule_type inverse_time --coupled_wd false --train_tokens $TOKENS --output_dir "$BASE_DIR/exp3_time_constant"

# Experiment 4: Inverse Time (Coupled) - Coupling variation for linear decay
echo "--- Running Exp 4: Inverse Time (Coupled) ---"
python train_llm.py --schedule_type inverse_time --coupled_wd true --train_tokens $TOKENS --output_dir "$BASE_DIR/exp4_time_coupled"

# Experiment 5: Inverse Sqrt (Coupled) with Lambda = 0.1
echo "--- Running Exp 5: Inverse Sqrt (Coupled, WD=0.1) ---"
python train_llm.py --schedule_type inverse_sqrt --coupled_wd true --weight_decay 0.1 --train_tokens $TOKENS --output_dir "$BASE_DIR/exp5_sqrt_wd0.1"

# Experiment 6: Inverse Sqrt (Coupled) with Lambda = 0.4
echo "--- Running Exp 6: Inverse Sqrt (Coupled, WD=0.4) ---"
python train_llm.py --schedule_type inverse_sqrt --coupled_wd true --weight_decay 0.4 --train_tokens $TOKENS --output_dir "$BASE_DIR/exp6_sqrt_wd0.4"

echo "✅ Sweep Complete. Check $BASE_DIR for results."
