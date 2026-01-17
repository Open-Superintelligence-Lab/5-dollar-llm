# 📝 Research TODOs

**Paper**: [Conditional Memory via Scalable Lookup (Cheng et al., 2025)](https://www.arxiv.org/pdf/2601.07372)

### 1. Implementation Audit & Correctness 
- [ ] **Audit `models/engram.py` against Equations 1-4**: Verify the exact mathematical alignment of the implementation (specifically the context-aware gating and `MultiHeadNgramHash`) with the paper's definitions. Document any authorized deviations.
- [ ] **Verify `TokenizerCompression`**: Ensure the text normalization (NFKC, lowercasing) matches the paper's preprocessing exactly to avoid semantic mismatches.
- [ ] **Check `engram_vocab_size` Scaling**: In `configs/llm_config.py`, we set `engram_vocab_size=200000`. The paper suggests this should scale with model size/tokens. Calculate the optimal size for our 88M model (vs the 27B in the paper) to avoid excessive collisions or memory waste.

### 2. Systems & Efficiency 
- [ ] **N-gram Hashing Collision Analysis**: Run a script over a subset of the Cosmopedia dataset to measure the collision rate of `MultiHeadNgramHash` with current table sizes. Plot collision rate vs table size.
- [ ] **Kernel Optimization**: Investigate if `torch.compile` correctly fuses the Engram gathering operations. If not, propose a custom Triton kernel for the `MultiHeadNgramHash` lookup.

### 3. Training Dynamics & Optimizer 
- [ ] **Implement Split-Optimization**: The paper expects Engram embeddings to be trained with **high LR** and **0 weight decay**. Currently `training/trainer.py` groups them with AdamW. Create a dedicated parameter group for `engram` params with `lr=10x_base` and `wd=0.0`, ablate different values.
- [ ] **Gradient Flow Analysis**: Monitor the gradient norm of the Engram embedding table (`embeddings.weight`) vs the content gating (`wk`, `wv`). Ensure the sparse updates are actually engaging the memory.
- [ ] **Ablation: Gating Mechanism**: Experiment with the gating activation. Current is `sigmoid`. Test if `softmax` (if applicable) or a learned bias helps stability during early training.

### 4. Interpretability & Analysis 
- [ ] **Gate Activation Logging**: Add logging to track the average value of the fusion gate $\alpha_t$. Does it saturate? Does it start low and increase?
- [ ] **Memory Utilization Map**: Visualize which N-grams (2-gram vs 3-gram) trigger higher gate values. Does the model prefer short-range or long-range patterns? 

You may choose any of these or suggest your own, make sure your research add novel insigts. You may contact us on discord to discuss it.