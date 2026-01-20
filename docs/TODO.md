# 📝 Research Ideas

> **Note**: These tasks are designed to be completed independently. Pick one that interests you and dive deep!

Check if it's properly implememnted in accordance to the paper - Hyper-Connections: https://arxiv.org/pdf/2409.19606 and DeepSeek-mHC: https://arxiv.org/pdf/2512.24880

---

## 🔬 mHC (Manifold-Constrained Hyper-Connections) Research

### Task 1: Sinkhorn-Knopp Convergence Analysis
**Assignee**: _____________  
**Priority**: High

- [ ] Analyze how the number of Sinkhorn-Knopp iterations (`mhc_sinkhorn_iters`) affects:
  - Training stability (gradient norms over time)
  - Final model performance (loss/perplexity)
  - Computational overhead (time per step)
- [ ] Test with iterations: 5, 10, 20, 50, 100
- [ ] Document findings with graphs
- [ ] **Research Question**: Is 20 iterations optimal, or can we reduce it for efficiency without hurting stability?

---

### Task 2: Expansion Rate Ablation
**Assignee**: _____________  
**Priority**: High

- [ ] Test different expansion rates (`mhc_expansion_rate`): 2, 4, 6, 8
- [ ] Measure parameter count increase vs. performance gain
- [ ] Compare memory usage and training throughput
- [ ] **Research Question**: What's the sweet spot between model capacity and efficiency for our scale?

---

### Task 3: Stream Initialization Strategies
**Assignee**: _____________  
**Priority**: Medium

- [ ] Compare the three stream initialization modes in `StreamExpansion`:
  - `replicate`: Copy input to all streams
  - `zeros_except_first`: Only first stream gets input
  - `learned`: Learned projection
- [ ] Test each on a 20M token run
- [ ] **Research Question**: Does initialization strategy matter for final performance, or does the model learn to overcome initial differences?

---

### Task 4: Alpha Gating Factor Dynamics
**Assignee**: _____________  
**Priority**: Medium

- [ ] Track how `alpha_pre`, `alpha_post`, `alpha_res` evolve during training
- [ ] Log these values every 500 steps and plot over time
- [ ] Compare static vs. dynamic mappings contribution (set alpha_init to 0 vs 0.01 vs 0.1)
- [ ] **Research Question**: How important are the dynamic (input-dependent) mappings vs. static biases?

---

### Task 5: Stability Comparison: mHC vs Standard Residual
**Assignee**: _____________  
**Priority**: High

- [ ] Train two models on 100M tokens:
  - Standard `MinimalLLM` (baseline)
  - `MinimalLLM_mHC` with default settings
- [ ] Compare:
  - Gradient norm stability over training
  - Loss curves (smoothness, convergence speed)
  - Final validation metrics
- [ ] **Research Question**: Does mHC provide measurable stability benefits at our model scale (88M params)?

---

## 🧠 Code Review & Verification

### Task 6: Verify Doubly Stochastic Properties
**Assignee**: _____________  
**Priority**: High

- [ ] Review `sinkhorn_knopp()` implementation in `models/mhc.py`
- [ ] Add assertions/logging during training to verify:
  - Row sums ≈ 1.0 (tolerance check)
  - Column sums ≈ 1.0
  - All entries ≥ 0
- [ ] Check for numerical stability issues (NaN, inf) with extreme inputs
- [ ] **Research Question**: Are there edge cases where Sinkhorn-Knopp fails to converge?

---

### Task 7: Gradient Flow Analysis
**Assignee**: _____________  
**Priority**: Medium

- [ ] Implement gradient magnitude tracking per layer
- [ ] Compare gradient flow between:
  - Standard residual connections
  - mHC with doubly stochastic constraint
- [ ] Visualize as heatmaps across layers and training steps
- [ ] **Research Question**: Does mHC truly preserve gradient magnitude as claimed by the paper?

---

### Task 8: Memory Profiling
**Assignee**: _____________  
**Priority**: Medium

- [ ] Profile peak memory usage with and without mHC
- [ ] Break down memory by component:
  - n-stream activation storage
  - H_pre, H_post, H_res parameter overhead
  - Intermediate buffers
- [ ] **Research Question**: Can we reduce memory overhead through selective recomputation (as mentioned in the paper)?

---

## 📚 Literature & Theory

### Task 9: Birkhoff Polytope Deep Dive
**Assignee**: _____________  
**Priority**: Low

- [ ] Research the mathematical properties of the Birkhoff polytope
- [ ] Understand why doubly stochastic matrices form a convex set
- [ ] Explore if there are alternative manifold constraints worth exploring
- [ ] Write a 1-page summary for the team
- [ ] **Research Question**: Are there other manifold constraints that could provide similar stability benefits?

---

### Task 10: Related Work Survey
**Assignee**: _____________  
**Priority**: Low

- [ ] Read and summarize the original Hyper-Connections paper (arXiv:2409.19606)
- [ ] Review DenseFormer, MUDDFormer, and other macro-architecture innovations
- [ ] Identify potential improvements or hybrid approaches
- [ ] **Research Question**: What other architectural innovations could complement mHC?

---

## 🏃 Baseline Training Tasks

### Task 11: Establish mHC Baseline
**Assignee**: _____________  
**Priority**: High

- [ ] Train on 100M tokens with mHC enabled (default settings)
- [ ] Record in LEADERBOARD.md
- [ ] Compare against existing baselines

### Task 12: Large-Scale Validation
**Assignee**: _____________  
**Priority**: High

- [ ] Train on 1B tokens with mHC
- [ ] Monitor for any stability issues at scale
- [ ] Document training dynamics

---

## 📋 Task Status

| Task | Assignee | Status | Notes |
|------|----------|--------|-------|
| 1. Sinkhorn Convergence | | ⬜ Not Started | |
| 2. Expansion Rate | | ⬜ Not Started | |
| 3. Stream Init | | ⬜ Not Started | |
| 4. Alpha Dynamics | | ⬜ Not Started | |
| 5. Stability Comparison | | ⬜ Not Started | |
| 6. Doubly Stochastic Verify | | ⬜ Not Started | |
| 7. Gradient Flow | | ⬜ Not Started | |
| 8. Memory Profiling | | ⬜ Not Started | |
| 9. Birkhoff Theory | | ⬜ Not Started | |
| 10. Related Work | | ⬜ Not Started | |
| 11. mHC Baseline | | ⬜ Not Started | |
| 12. 1B Validation | | ⬜ Not Started | |

---

## 🚀 Quick Start Commands

```bash
# Run mHC tests
python test_mhc.py

# Train with mHC (quick test)
python train_llm.py --use_mhc --train_tokens 8000000

# Train with custom mHC params
python train_llm.py --use_mhc \
    --mhc_expansion_rate 4 \
    --mhc_alpha_init 0.01 \
    --mhc_sinkhorn_iters 20 \
    --train_tokens 100000000
```

---

*Last updated: 2026-01-17*
