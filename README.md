
> ## **🚀 THIS BRANCH IS FOR SQUARED RELU RESEARCH**. [VIEW TASK](DO_THIS_TASK.md).

# 5-Dollar LLM (Blueberry 151M)

> Training the best possible LLM from scratch for $5.

- 151M paramters dense LLM.

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

## 📺 YouTube Video

[![Watch the video](https://img.youtube.com/vi/dayc3y34XXs/maxresdefault.jpg)](https://youtu.be/dayc3y34XXs)

🎥 **[Watch our introduction video](https://youtu.be/dayc3y34XXs)** to learn more about the project!

> Check out our speedrun [leaderboard](docs/LEADERBOARD.md)!

## 🗺️ Roadmap

**Our goals:**
1. **GPT-1** Level by Dec 20 2025
2. **GPT-2** Level by Jan 20 2026
3. **GPT-3** Level by Feb 20 2026
4. **Top 150** in LMArena (GPT-4o-mini level) by April 2026
5. **Top 50** by Dec 2026
6. **Top 10** by April 2027
7. We could aim for **Top 1** by 2028, TBD

**Likely architecture for our first LLM (Top 150, April 2026):**
- 8 Billion Parameters
- 15 Trillion Tokens

This requires 300,000 H100 hours or equivalent.

We will partner with one or multiple partners for this compute while keeping all research / engineering / code FULLY open source (and making daily videos on everything we do), for the sake of open science that benefits everyone.

**Potential partners include:**
Hugging Face, NVIDIA, Microsoft, Google, Amazon, Meta, IBM, Oracle, Alibaba, Tencent, Huawei, Baidu, CoreWeave, Lambda Labs, Hyperbolic, Stability AI, OpenAI, Anthropic, xAI, Cohere, Mistral AI, Graphcore, Tenstorrent, Intel, AMD, Dell Technologies, ai2, a16z, Sequoia Capital, and more.

As a community, we will find ways to get the compute.

Currently LLMs are the most useful AI models, so it's a clear way for us to do useful research. As we gain more experience, we will expand towards more speculative research that could lead to better AI models.

**If you or someone you know has extensive research experience and can offer advisory or leadership support, please contact me.**

---

## 🏎️ The Speedrun Challenge

Can you train a model to **4.5 loss** in under 3 minutes?

### ⚡ Speedrun 1 (The 3-Minute Challenge)
Reach **4.5 loss** as fast as possible.
- **Rules:** 151M params ± 5%, must use speedrun data. See [Official Rules](docs/LEADERBOARD.md#📜-official-rules).
- **Timing:** Uses untimed compilation warmup for fair benchmarking.

```bash
git clone https://github.com/Open-Superintelligence-Lab/5-dollar-llm
cd 5-dollar-llm
pip install -r requirements.txt
python data/download_hf_data.py   # Downloads 40M token subset
python train_llm.py --target_train_loss 4.5
```

### 📚 Full Documentation
For detailed environment setup, data options (1B tokens), and leaderboard rules, see:
👉 **[Full Setup & Speedrun Guide](docs/SETUP_INSTRUCTIONS.md)**
👉 **[Leaderboard](docs/LEADERBOARD.md)**


## 🚀 Getting Started & Contributing

We welcome all contributions! Follow this workflow to get started:

### 1. **Pick a task**
Check the [Tasks](https://github.com/Open-Superintelligence-Lab/5-dollar-llm/issues) and carefully read and understand the task you want to work on. Leave a comment or send a Discord message when you begin working on it. If you want to do something that is not listed, please tell us on Discord to ensure you do it in a way that aligns with our goal.

### 2. **Set up your environment**

1. **Fork this repository** - Click the "Fork" button at the top right to create your own copy.
2. **Clone your fork**:
   ```bash
   git clone FORK_URL_HERE
   cd 5-dollar-llm
   ```
   *(You may also clone it with your IDE)*
   
   > **Note:** If you have already forked/cloned, please ensure you sync your fork with this repo & pull the latest changes to your local before starting - we make frequent changes.

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### 3. **Implement your changes**
Write your code following the setup instructions above and our coding standards.

## 🧪 Running Experiments

To run a new experiment without overwriting the baseline, simply provide a unique experiment name:

```bash
python train_llm.py --experiment_name my_new_experiment
```

Results (checkpoints and logs) will be saved to `checkpoints/my_new_experiment/` for easy comparison.

> **Performance Test (optional):** We will run the experiments anyways, but you may also run it yourself. Make sure to specify a new name so you don't overwrite the baseline.

### 4. **Submission**
Once finished, create a Pull Request into the `main` branch. Please notify us on [Discord](https://discord.gg/6AbXGpKTwN).

> Please read `CONTRIBUTING.md` for detailed guidelines.
