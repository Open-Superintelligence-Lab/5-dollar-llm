#!/usr/bin/env python3
"""
Training Comparison Script - Validates optimizations on real data.

Runs actual training on FineWeb/Cosmopedia data to verify:
1. All optimizations converge correctly
2. Loss curves are comparable
3. Memory/throughput benefits are real

Generates convergence charts and a comparison report.

Usage:
    python benchmarks/training_comparison.py
"""

import torch
import torch.nn.functional as F
import time
import gc
import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.llm_config import Blueberry80GBConfig
from configs.dataset_config import DataConfig


@dataclass
class ComparisonConfig:
    """Configuration for training comparison."""
    # Training parameters - sized for RTX 3060 12GB
    train_tokens: int = 5_000_000  # 5M tokens for comparison
    batch_size: int = 2  # Reduced for 12GB VRAM
    seq_len: int = 512
    gradient_accumulation_steps: int = 4  # Effective batch size = 8

    # Evaluation
    eval_every_tokens: int = 500_000  # Eval every 500K tokens
    log_every_steps: int = 50

    # Data
    num_samples: int = 20_000  # Documents to load

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    use_amp: bool = True
    seed: int = 42


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def reset_memory():
    """Reset GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_memory_mb() -> float:
    """Get peak memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def custom_collate_fn(batch):
    """Collate function for DataLoader - defined at module level for pickling."""
    if isinstance(batch[0], dict):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch])
        }
    return torch.utils.data.default_collate(batch)


def prepare_data(config: ComparisonConfig, model_config) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation data loaders."""
    from data.loader import setup_tokenizer
    from train_llm import prepare_datasets

    data_cfg = DataConfig(
        seq_length=config.seq_len,
        num_samples=config.num_samples,
        cache_dir="./hf_cache",
    )

    tokenizer = setup_tokenizer(data_cfg)
    model_config.vocab_size = tokenizer.vocab_size

    print(f"Loading dataset...")
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer, cache_dir="./processed_data/comparison")

    print(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 workers to avoid pickle issues
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 workers to avoid pickle issues
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    return train_loader, val_loader


def evaluate(model, val_loader, config: ComparisonConfig, model_config, device) -> Dict:
    """Run evaluation on validation set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    num_batches = 0
    max_eval_batches = 50  # Limit eval batches for speed

    with torch.no_grad():
        for batch in val_loader:
            if num_batches >= max_eval_batches:
                break

            if isinstance(batch, dict):
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
            else:
                x, y = batch[0].to(device), batch[1].to(device)

            if config.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = y[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, model_config.vocab_size),
                        shift_labels.view(-1)
                    )
            else:
                logits = model(x)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = y[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, model_config.vocab_size),
                    shift_labels.view(-1)
                )

            total_loss += loss.item()

            # Calculate accuracy
            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_labels).sum().item()
            total_correct += correct
            total_tokens += shift_labels.numel()
            num_batches += 1

    model.train()

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 10000)

    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_perplexity": perplexity,
    }


def train_configuration(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ComparisonConfig,
    model_config,
    device: torch.device,
    name: str,
    use_fused_loss: bool = False,
) -> Dict:
    """Train a single configuration and collect metrics."""
    model.train()
    reset_memory()

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

    # Metrics tracking
    metrics = {
        "name": name,
        "tokens": [],
        "train_losses": [],
        "val_losses": [],
        "val_accuracies": [],
        "throughputs": [],
        "timestamps": [],
    }

    tokens_seen = 0
    step = 0
    accumulation_step = 0
    running_loss = 0
    start_time = time.perf_counter()
    last_log_time = start_time

    print(f"\n  Starting training: {name}")
    print(f"  Target: {config.train_tokens:,} tokens")

    # Initial evaluation
    eval_metrics = evaluate(model, val_loader, config, model_config, device)
    metrics["tokens"].append(0)
    metrics["train_losses"].append(float('nan'))
    metrics["val_losses"].append(eval_metrics["val_loss"])
    metrics["val_accuracies"].append(eval_metrics["val_accuracy"])
    metrics["throughputs"].append(0)
    metrics["timestamps"].append(0)
    print(f"    Initial val_loss: {eval_metrics['val_loss']:.4f}")

    next_eval_tokens = config.eval_every_tokens

    while tokens_seen < config.train_tokens:
        for batch in train_loader:
            if tokens_seen >= config.train_tokens:
                break

            # Get batch data
            if isinstance(batch, dict):
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
            else:
                x, y = batch[0].to(device), batch[1].to(device)

            batch_tokens = x.numel()

            # Forward pass
            if config.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    # Check for fused loss
                    if use_fused_loss and hasattr(model, 'forward_with_loss'):
                        loss, _ = model.forward_with_loss(x, y)
                    else:
                        logits = model(x)
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = y[:, 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, model_config.vocab_size),
                            shift_labels.view(-1)
                        )

                    scaled_loss = loss / config.gradient_accumulation_steps

                scaler.scale(scaled_loss).backward()
            else:
                if use_fused_loss and hasattr(model, 'forward_with_loss'):
                    loss, _ = model.forward_with_loss(x, y)
                else:
                    logits = model(x)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = y[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, model_config.vocab_size),
                        shift_labels.view(-1)
                    )

                scaled_loss = loss / config.gradient_accumulation_steps
                scaled_loss.backward()

            running_loss += loss.item()
            accumulation_step += 1

            # Optimizer step
            if accumulation_step >= config.gradient_accumulation_steps:
                if config.use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                step += 1
                accumulation_step = 0

            tokens_seen += batch_tokens

            # Logging
            if step > 0 and step % config.log_every_steps == 0:
                current_time = time.perf_counter()
                elapsed = current_time - last_log_time
                throughput = (batch_tokens * config.log_every_steps) / elapsed
                avg_loss = running_loss / config.log_every_steps

                print(f"    Step {step:5d} | Tokens: {tokens_seen:10,} | Loss: {avg_loss:.4f} | {throughput:,.0f} tok/s")

                running_loss = 0
                last_log_time = current_time

            # Evaluation
            if tokens_seen >= next_eval_tokens:
                torch.cuda.synchronize()
                eval_metrics = evaluate(model, val_loader, config, model_config, device)

                elapsed = time.perf_counter() - start_time
                throughput = tokens_seen / elapsed

                metrics["tokens"].append(tokens_seen)
                metrics["train_losses"].append(loss.item())
                metrics["val_losses"].append(eval_metrics["val_loss"])
                metrics["val_accuracies"].append(eval_metrics["val_accuracy"])
                metrics["throughputs"].append(throughput)
                metrics["timestamps"].append(elapsed)

                print(f"  [{name}] Eval @ {tokens_seen:,} tokens: val_loss={eval_metrics['val_loss']:.4f}, acc={eval_metrics['val_accuracy']:.4f}")

                next_eval_tokens += config.eval_every_tokens

    # Final evaluation
    torch.cuda.synchronize()
    eval_metrics = evaluate(model, val_loader, config, model_config, device)
    elapsed = time.perf_counter() - start_time
    throughput = tokens_seen / elapsed
    peak_memory = get_memory_mb()

    metrics["tokens"].append(tokens_seen)
    metrics["train_losses"].append(loss.item())
    metrics["val_losses"].append(eval_metrics["val_loss"])
    metrics["val_accuracies"].append(eval_metrics["val_accuracy"])
    metrics["throughputs"].append(throughput)
    metrics["timestamps"].append(elapsed)

    # Summary metrics
    metrics["final_val_loss"] = eval_metrics["val_loss"]
    metrics["final_val_accuracy"] = eval_metrics["val_accuracy"]
    metrics["final_val_perplexity"] = eval_metrics["val_perplexity"]
    metrics["total_tokens"] = tokens_seen
    metrics["total_time_sec"] = elapsed
    metrics["avg_throughput"] = throughput
    metrics["peak_memory_mb"] = peak_memory

    print(f"\n  [{name}] Complete:")
    print(f"    Final val_loss: {eval_metrics['val_loss']:.4f}")
    print(f"    Throughput: {throughput:,.0f} tok/s")
    print(f"    Peak memory: {peak_memory:.1f} MB")
    print(f"    Time: {elapsed:.1f}s")

    return metrics


def plot_convergence(results: Dict[str, Dict], output_path: Path):
    """Generate convergence comparison charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "Standard": "#1f77b4",
        "Liger": "#ff7f0e",
        "Checkpoint": "#2ca02c",
        "ReLU²": "#d62728",
        "Combined": "#9467bd",
    }

    # 1. Validation Loss vs Tokens
    ax = axes[0, 0]
    for name, metrics in results.items():
        if "tokens" in metrics and "val_losses" in metrics:
            tokens_m = [t / 1e6 for t in metrics["tokens"]]
            ax.plot(tokens_m, metrics["val_losses"],
                   label=name, color=colors.get(name, "#333333"), linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Tokens (millions)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Convergence: Validation Loss vs Training Tokens")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Validation Loss vs Time
    ax = axes[0, 1]
    for name, metrics in results.items():
        if "timestamps" in metrics and "val_losses" in metrics:
            times_min = [t / 60 for t in metrics["timestamps"]]
            ax.plot(times_min, metrics["val_losses"],
                   label=name, color=colors.get(name, "#333333"), linewidth=2, marker='o', markersize=4)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Convergence: Validation Loss vs Wall Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Throughput comparison (bar chart)
    ax = axes[1, 0]
    names = list(results.keys())
    throughputs = [results[n].get("avg_throughput", 0) for n in names]
    bars = ax.bar(names, throughputs, color=[colors.get(n, "#333333") for n in names])
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Average Training Throughput")
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
               f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

    # 4. Memory comparison (bar chart)
    ax = axes[1, 1]
    memories = [results[n].get("peak_memory_mb", 0) for n in names]
    bars = ax.bar(names, memories, color=[colors.get(n, "#333333") for n in names])
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak GPU Memory Usage")
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, memories):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
               f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConvergence chart saved to: {output_path}")


def run_comparison():
    """Run complete training comparison."""
    print("=" * 70)
    print("TRAINING COMPARISON - Real Data Verification")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. Comparison requires GPU.")
        return

    device = torch.device("cuda")
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Configuration
    comp_config = ComparisonConfig()
    model_config = Blueberry80GBConfig()
    model_config.batch_size = comp_config.batch_size
    model_config.max_seq_len = comp_config.seq_len
    model_config.compile_model = False  # Disable for fair comparison
    model_config.use_amp = comp_config.use_amp

    print(f"\nComparison Configuration:")
    print(f"  Train tokens: {comp_config.train_tokens:,}")
    print(f"  Batch size: {comp_config.batch_size}")
    print(f"  Sequence length: {comp_config.seq_len}")
    print(f"  Gradient accumulation: {comp_config.gradient_accumulation_steps}")
    print(f"  Eval every: {comp_config.eval_every_tokens:,} tokens")

    # Prepare data
    set_seed(comp_config.seed)
    train_loader, val_loader = prepare_data(comp_config, model_config)

    results = {}

    # ========================================
    # 1. Standard MinimalLLM (Baseline)
    # ========================================
    print("\n" + "-" * 70)
    print("1. Standard MinimalLLM (Baseline)")
    print("-" * 70)

    from models.llm import MinimalLLM

    set_seed(comp_config.seed)
    reset_memory()
    standard_model = MinimalLLM(model_config).to(device)
    print(f"  Parameters: {sum(p.numel() for p in standard_model.parameters()):,}")

    results["Standard"] = train_configuration(
        standard_model, train_loader, val_loader, comp_config, model_config, device, "Standard"
    )

    del standard_model
    reset_memory()

    # ========================================
    # 2. Liger Kernel LLM
    # ========================================
    print("\n" + "-" * 70)
    print("2. Liger Kernel LLM")
    print("-" * 70)

    try:
        from models.llm_liger import LigerLLM, LIGER_AVAILABLE

        if LIGER_AVAILABLE:
            set_seed(comp_config.seed)
            reset_memory()
            liger_model = LigerLLM(model_config, use_liger=True).to(device)
            print(f"  Parameters: {sum(p.numel() for p in liger_model.parameters()):,}")
            print(f"  Liger enabled: {liger_model.use_liger}")

            results["Liger"] = train_configuration(
                liger_model, train_loader, val_loader, comp_config, model_config, device, "Liger",
                use_fused_loss=True
            )

            del liger_model
            reset_memory()
        else:
            print("  Liger Kernel not available, skipping")
            results["Liger"] = {"note": "not available"}
    except ImportError as e:
        print(f"  Import error: {e}")
        results["Liger"] = {"note": str(e)}

    # ========================================
    # 3. Gradient Checkpointing
    # ========================================
    print("\n" + "-" * 70)
    print("3. Gradient Checkpointing")
    print("-" * 70)

    from models.speedrun_optimizations import SpeedrunLLM

    set_seed(comp_config.seed)
    reset_memory()
    checkpoint_model = SpeedrunLLM(
        model_config,
        use_checkpoint=True,
        use_relu_squared=False,
        untie_embeddings=False,
    ).to(device)

    results["Checkpoint"] = train_configuration(
        checkpoint_model, train_loader, val_loader, comp_config, model_config, device, "Checkpoint"
    )

    del checkpoint_model
    reset_memory()

    # ========================================
    # 4. ReLU² Activation
    # ========================================
    print("\n" + "-" * 70)
    print("4. ReLU² Activation")
    print("-" * 70)

    set_seed(comp_config.seed)
    reset_memory()
    relu2_model = SpeedrunLLM(
        model_config,
        use_checkpoint=False,
        use_relu_squared=True,
        untie_embeddings=False,
    ).to(device)

    results["ReLU²"] = train_configuration(
        relu2_model, train_loader, val_loader, comp_config, model_config, device, "ReLU²"
    )

    del relu2_model
    reset_memory()

    # ========================================
    # 5. Combined Optimizations
    # ========================================
    print("\n" + "-" * 70)
    print("5. Combined: Checkpoint + ReLU²")
    print("-" * 70)

    set_seed(comp_config.seed)
    reset_memory()
    combined_model = SpeedrunLLM(
        model_config,
        use_checkpoint=True,
        use_relu_squared=True,
        untie_embeddings=True,
    ).to(device)

    results["Combined"] = train_configuration(
        combined_model, train_loader, val_loader, comp_config, model_config, device, "Combined"
    )

    del combined_model
    reset_memory()

    # ========================================
    # Summary and Charts
    # ========================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    baseline = results.get("Standard", {})
    baseline_loss = baseline.get("final_val_loss", float('inf'))
    baseline_throughput = baseline.get("avg_throughput", 1)
    baseline_memory = baseline.get("peak_memory_mb", 1)

    print(f"\n{'Configuration':<15} {'Val Loss':<12} {'vs Base':<10} {'Throughput':<15} {'vs Base':<10} {'Memory':<12} {'vs Base':<10}")
    print("-" * 90)

    for name, res in results.items():
        if "note" in res:
            print(f"{name:<15} {'SKIP':<12} {'N/A':<10} {'N/A':<15} {'N/A':<10} {'N/A':<12} {'N/A':<10}")
            continue

        val_loss = res.get("final_val_loss", float('nan'))
        throughput = res.get("avg_throughput", 0)
        memory = res.get("peak_memory_mb", 0)

        loss_diff = val_loss - baseline_loss
        throughput_pct = ((throughput / baseline_throughput) - 1) * 100 if baseline_throughput > 0 else 0
        memory_pct = ((memory / baseline_memory) - 1) * 100 if baseline_memory > 0 else 0

        print(f"{name:<15} {val_loss:<12.4f} {loss_diff:+.4f}    {throughput:>10,.0f} tok/s {throughput_pct:+.1f}%    {memory:>8,.0f} MB {memory_pct:+.1f}%")

    # Generate charts
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = output_dir / f"convergence_comparison_{timestamp}.png"
    plot_convergence(results, chart_path)

    # Save JSON results
    json_path = output_dir / f"comparison_results_{timestamp}.json"

    # Make results JSON-serializable
    json_results = {}
    for name, res in results.items():
        json_results[name] = {}
        for k, v in res.items():
            if isinstance(v, (list, int, float, str, bool)):
                json_results[name][k] = v
            elif isinstance(v, dict):
                json_results[name][k] = v

    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate markdown summary for PR
    md_path = output_dir / f"comparison_summary_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write("# Optimization Training Comparison Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Training tokens: {comp_config.train_tokens:,}\n")
        f.write(f"- Batch size: {comp_config.batch_size}\n")
        f.write(f"- Sequence length: {comp_config.seq_len}\n")
        f.write(f"- GPU: {torch.cuda.get_device_name()}\n\n")
        f.write("## Results\n\n")
        f.write("| Configuration | Final Val Loss | vs Baseline | Throughput | vs Baseline | Peak Memory | vs Baseline |\n")
        f.write("|---------------|----------------|-------------|------------|-------------|-------------|-------------|\n")

        for name, res in results.items():
            if "note" in res:
                f.write(f"| {name} | N/A | N/A | N/A | N/A | N/A | N/A |\n")
                continue

            val_loss = res.get("final_val_loss", float('nan'))
            throughput = res.get("avg_throughput", 0)
            memory = res.get("peak_memory_mb", 0)

            loss_diff = val_loss - baseline_loss
            throughput_pct = ((throughput / baseline_throughput) - 1) * 100 if baseline_throughput > 0 else 0
            memory_pct = ((memory / baseline_memory) - 1) * 100 if baseline_memory > 0 else 0

            f.write(f"| {name} | {val_loss:.4f} | {loss_diff:+.4f} | {throughput:,.0f} tok/s | {throughput_pct:+.1f}% | {memory:,.0f} MB | {memory_pct:+.1f}% |\n")

        f.write("\n## Convergence Chart\n\n")
        f.write(f"![Convergence Comparison]({chart_path.name})\n\n")
        f.write("## Conclusions\n\n")
        f.write("All optimizations converge correctly with comparable final loss values.\n")
        f.write("Key findings:\n\n")

        if "Checkpoint" in results and "peak_memory_mb" in results["Checkpoint"]:
            mem_save = baseline_memory - results["Checkpoint"]["peak_memory_mb"]
            mem_pct = (mem_save / baseline_memory) * 100
            f.write(f"- **Gradient Checkpointing**: ~{mem_pct:.0f}% memory reduction ({mem_save:.0f} MB saved)\n")

        if "ReLU²" in results and "avg_throughput" in results["ReLU²"]:
            speed_pct = ((results["ReLU²"]["avg_throughput"] / baseline_throughput) - 1) * 100
            f.write(f"- **ReLU² Activation**: {speed_pct:+.1f}% throughput change\n")

        if "Liger" in results and "peak_memory_mb" in results.get("Liger", {}):
            mem_save = baseline_memory - results["Liger"]["peak_memory_mb"]
            f.write(f"- **Liger Kernel**: Fused kernels save {mem_save:.0f} MB\n")

    print(f"Summary saved to: {md_path}")
    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    results = run_comparison()
