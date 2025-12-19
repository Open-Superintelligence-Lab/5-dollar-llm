# 🚀 TASK: Squared ReLU Research

We found a **speedup** by switching from SwiGLU to **Squared ReLU** (from the "[Primer](https://arxiv.org/pdf/2109.08668)" paper) - but this needs to be verified.

### 📊 Results (151M Params)
| Activation | FFN Width ($d_{ff}$) | Time to 4.5 Loss |
| :--- | :--- | :--- |
| SwiGLU | 2048 | 1m 59s |
| **Squared ReLU** | **3072** | **1m 52s** |

### 🏎️ Run Benchmark
Test it yourself:
```bash
python compare_ffn_types.py
```

---

### 🛠️ Open Tasks (Contribute!)
1. **Optimize LR**: Do learning rate search for squared relu.
2. Your own improvements

Make sure to 

- Submit a PR into this branch.
