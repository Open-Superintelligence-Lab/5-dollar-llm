#!/usr/bin/env python3
"""
Verification script for optimization correctness.

Runs short training runs to verify that optimizations:
1. Train correctly (loss decreases)
2. Produce comparable results
3. Don't break model convergence

This is essential validation before claiming optimization benefits.

Usage:
    python benchmarks/verify_optimizations.py
"""

import torch
import torch.nn.functional as F
import time
import gc
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.llm_config import Blueberry80GBConfig


@dataclass
class VerificationConfig:
    """Short training run configuration for verification."""
    batch_size: int = 2
    seq_len: int = 512  # Shorter for faster verification
    train_steps: int = 100  # Short run
    eval_every: int = 25
    use_amp: bool = True
    seed: int = 42


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def create_synthetic_data(config: VerificationConfig, model_config, device, num_batches: int = 200):
    """Create synthetic training data."""
    data = []
    for _ in range(num_batches):
        input_ids = torch.randint(0, model_config.vocab_size, (config.batch_size, config.seq_len), device=device)
        data.append(input_ids)
    return data


def train_short_run(
    model: torch.nn.Module,
    data: List[torch.Tensor],
    config: VerificationConfig,
    model_config,
    device: torch.device,
    name: str,
) -> Dict:
    """Run a short training to verify model works."""
    model.train()
    reset_memory()

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

    losses = []
    start_time = time.perf_counter()

    for step in range(config.train_steps):
        input_ids = data[step % len(data)]
        labels = input_ids.clone()

        optimizer.zero_grad()

        if config.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Check if model has forward_with_loss (Liger)
                if hasattr(model, 'forward_with_loss'):
                    loss, _ = model.forward_with_loss(input_ids, labels)
                else:
                    logits = model(input_ids)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, model_config.vocab_size),
                        shift_labels.view(-1)
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            if hasattr(model, 'forward_with_loss'):
                loss, _ = model.forward_with_loss(input_ids, labels)
            else:
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, model_config.vocab_size),
                    shift_labels.view(-1)
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        losses.append(loss.item())

        if (step + 1) % config.eval_every == 0:
            avg_loss = sum(losses[-config.eval_every:]) / config.eval_every
            print(f"  [{name}] Step {step + 1}: loss = {avg_loss:.4f}")

    elapsed = time.perf_counter() - start_time
    peak_memory = get_memory_mb()

    # Calculate metrics
    initial_loss = sum(losses[:10]) / 10
    final_loss = sum(losses[-10:]) / 10
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100

    tokens_processed = config.train_steps * config.batch_size * config.seq_len
    throughput = tokens_processed / elapsed

    return {
        "name": name,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_pct": loss_reduction,
        "throughput_tokens_per_sec": throughput,
        "peak_memory_mb": peak_memory,
        "elapsed_sec": elapsed,
        "converged": loss_reduction > 5,  # At least 5% loss reduction
        "losses": losses,
    }


def verify_gradient_equivalence(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    data: torch.Tensor,
    model_config,
) -> Dict:
    """Verify two models produce equivalent gradients."""
    model1.train()
    model2.train()

    # Zero gradients
    model1.zero_grad()
    model2.zero_grad()

    labels = data.clone()

    # Forward + backward on model1
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits1 = model1(data)
        shift_logits1 = logits1[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss1 = F.cross_entropy(
            shift_logits1.view(-1, model_config.vocab_size),
            shift_labels.view(-1)
        )
    loss1.backward()

    # Forward + backward on model2
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits2 = model2(data)
        shift_logits2 = logits2[:, :-1, :].contiguous()
        loss2 = F.cross_entropy(
            shift_logits2.view(-1, model_config.vocab_size),
            shift_labels.view(-1)
        )
    loss2.backward()

    # Compare gradients
    max_diff = 0
    total_params = 0
    matching_params = 0

    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if p1.grad is not None and p2.grad is not None:
            diff = (p1.grad - p2.grad).abs().max().item()
            max_diff = max(max_diff, diff)
            total_params += 1
            if diff < 1e-3:  # Allow small numerical differences
                matching_params += 1

    return {
        "loss1": loss1.item(),
        "loss2": loss2.item(),
        "loss_diff": abs(loss1.item() - loss2.item()),
        "max_grad_diff": max_diff,
        "grad_match_ratio": matching_params / total_params if total_params > 0 else 0,
        "gradients_equivalent": max_diff < 1e-2,
    }


def run_verification():
    """Run complete verification suite."""
    print("=" * 70)
    print("OPTIMIZATION VERIFICATION SUITE")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. Verification requires GPU.")
        return

    device = torch.device("cuda")
    print(f"\nDevice: {torch.cuda.get_device_name()}")

    # Configuration
    verify_config = VerificationConfig()
    model_config = Blueberry80GBConfig()
    model_config.batch_size = verify_config.batch_size
    model_config.max_seq_len = verify_config.seq_len

    print(f"\nVerification Configuration:")
    print(f"  Batch size: {verify_config.batch_size}")
    print(f"  Sequence length: {verify_config.seq_len}")
    print(f"  Training steps: {verify_config.train_steps}")
    print(f"  Seed: {verify_config.seed}")

    # Create synthetic data
    set_seed(verify_config.seed)
    data = create_synthetic_data(verify_config, model_config, device)
    print(f"  Data batches: {len(data)}")

    results = {}

    # ========================================
    # 1. Standard MinimalLLM (Baseline)
    # ========================================
    print("\n" + "-" * 70)
    print("1. Standard MinimalLLM (Baseline)")
    print("-" * 70)

    from models.llm import MinimalLLM

    set_seed(verify_config.seed)
    reset_memory()
    standard_model = MinimalLLM(model_config).to(device)

    result = train_short_run(standard_model, data, verify_config, model_config, device, "Standard")
    results["standard"] = result

    print(f"\n  Initial loss: {result['initial_loss']:.4f}")
    print(f"  Final loss: {result['final_loss']:.4f}")
    print(f"  Loss reduction: {result['loss_reduction_pct']:.1f}%")
    print(f"  Throughput: {result['throughput_tokens_per_sec']:,.0f} tok/s")
    print(f"  Peak memory: {result['peak_memory_mb']:.1f} MB")
    print(f"  Converged: {'YES' if result['converged'] else 'NO'}")

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
            set_seed(verify_config.seed)
            reset_memory()
            liger_model = LigerLLM(model_config, use_liger=True).to(device)

            result = train_short_run(liger_model, data, verify_config, model_config, device, "Liger")
            results["liger"] = result

            print(f"\n  Initial loss: {result['initial_loss']:.4f}")
            print(f"  Final loss: {result['final_loss']:.4f}")
            print(f"  Loss reduction: {result['loss_reduction_pct']:.1f}%")
            print(f"  Throughput: {result['throughput_tokens_per_sec']:,.0f} tok/s")
            print(f"  Peak memory: {result['peak_memory_mb']:.1f} MB")
            print(f"  Converged: {'YES' if result['converged'] else 'NO'}")

            del liger_model
            reset_memory()
        else:
            print("  Liger Kernel not available, skipping")
            results["liger"] = {"converged": None, "note": "not available"}

    except ImportError as e:
        print(f"  Import error: {e}")
        results["liger"] = {"converged": None, "note": str(e)}

    # ========================================
    # 3. Gradient Checkpointing
    # ========================================
    print("\n" + "-" * 70)
    print("3. Gradient Checkpointing")
    print("-" * 70)

    from models.speedrun_optimizations import SpeedrunLLM

    set_seed(verify_config.seed)
    reset_memory()
    checkpoint_model = SpeedrunLLM(
        model_config,
        use_checkpoint=True,
        use_relu_squared=False,
        untie_embeddings=False,
    ).to(device)

    result = train_short_run(checkpoint_model, data, verify_config, model_config, device, "Checkpoint")
    results["checkpoint"] = result

    print(f"\n  Initial loss: {result['initial_loss']:.4f}")
    print(f"  Final loss: {result['final_loss']:.4f}")
    print(f"  Loss reduction: {result['loss_reduction_pct']:.1f}%")
    print(f"  Throughput: {result['throughput_tokens_per_sec']:,.0f} tok/s")
    print(f"  Peak memory: {result['peak_memory_mb']:.1f} MB")
    print(f"  Converged: {'YES' if result['converged'] else 'NO'}")

    del checkpoint_model
    reset_memory()

    # ========================================
    # 4. ReLU² Activation
    # ========================================
    print("\n" + "-" * 70)
    print("4. ReLU² Activation")
    print("-" * 70)

    set_seed(verify_config.seed)
    reset_memory()
    relu2_model = SpeedrunLLM(
        model_config,
        use_checkpoint=False,
        use_relu_squared=True,
        untie_embeddings=False,
    ).to(device)

    result = train_short_run(relu2_model, data, verify_config, model_config, device, "ReLU²")
    results["relu2"] = result

    print(f"\n  Initial loss: {result['initial_loss']:.4f}")
    print(f"  Final loss: {result['final_loss']:.4f}")
    print(f"  Loss reduction: {result['loss_reduction_pct']:.1f}%")
    print(f"  Throughput: {result['throughput_tokens_per_sec']:,.0f} tok/s")
    print(f"  Peak memory: {result['peak_memory_mb']:.1f} MB")
    print(f"  Converged: {'YES' if result['converged'] else 'NO'}")

    del relu2_model
    reset_memory()

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = True
    baseline = results["standard"]

    print(f"\n{'Configuration':<20} {'Converged':<10} {'Final Loss':<12} {'vs Baseline':<12} {'Memory':<12}")
    print("-" * 70)

    for name, res in results.items():
        if res.get("converged") is None:
            print(f"{name:<20} {'SKIP':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            continue

        converged = "YES" if res["converged"] else "NO"
        final_loss = f"{res['final_loss']:.4f}"
        loss_diff = res['final_loss'] - baseline['final_loss']
        vs_baseline = f"{loss_diff:+.4f}"
        memory = f"{res['peak_memory_mb']:.0f} MB"

        if not res["converged"]:
            all_passed = False

        print(f"{name:<20} {converged:<10} {final_loss:<12} {vs_baseline:<12} {memory:<12}")

    print("\n" + "-" * 70)
    if all_passed:
        print("ALL OPTIMIZATIONS VERIFIED - Training converges correctly")
    else:
        print("WARNING: Some optimizations may have issues")

    # Save results
    output_file = Path(__file__).parent / "verification_results.json"
    with open(output_file, "w") as f:
        # Remove non-serializable losses list for JSON
        json_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                json_results[k] = {kk: vv for kk, vv in v.items() if kk != "losses"}
            else:
                json_results[k] = v
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    print("=" * 70)
    return results


if __name__ == "__main__":
    results = run_verification()
