#!/usr/bin/env python3
"""
Benchmark script for speedrun optimizations.

Compares:
1. Standard MinimalLLM
2. Gradient Checkpointing (memory savings)
3. ReLU² activation (faster convergence)
4. Combined optimizations

Usage:
    python benchmarks/speedrun_benchmark.py
"""

import torch
import torch.nn.functional as F
import time
import gc
import sys
from pathlib import Path
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.llm_config import Blueberry80GBConfig
from models.llm import MinimalLLM
from models.speedrun_optimizations import SpeedrunLLM, get_speedrun_status


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    batch_size: int = 2
    seq_len: int = 1024
    warmup_steps: int = 3
    benchmark_steps: int = 10
    use_amp: bool = True


def get_memory_stats() -> dict:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}

    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "peak_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def reset_memory():
    """Reset memory tracking."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_forward_backward(
    model: torch.nn.Module,
    config: BenchmarkConfig,
    model_config: Blueberry80GBConfig,
    device: torch.device,
) -> dict:
    """Benchmark forward + backward pass."""
    model.train()
    reset_memory()

    # Create dummy input/labels
    input_ids = torch.randint(0, model_config.vocab_size, (config.batch_size, config.seq_len), device=device)
    labels = torch.randint(0, model_config.vocab_size, (config.batch_size, config.seq_len), device=device)

    # Warmup
    for _ in range(config.warmup_steps):
        if config.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, model_config.vocab_size),
                    shift_labels.view(-1)
                )
            loss.backward()
        else:
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, model_config.vocab_size),
                shift_labels.view(-1)
            )
            loss.backward()

        model.zero_grad()

    torch.cuda.synchronize()
    reset_memory()

    # Benchmark
    start_time = time.perf_counter()

    for _ in range(config.benchmark_steps):
        if config.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, model_config.vocab_size),
                    shift_labels.view(-1)
                )
            loss.backward()
        else:
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, model_config.vocab_size),
                shift_labels.view(-1)
            )
            loss.backward()

        model.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    tokens_processed = config.batch_size * config.seq_len * config.benchmark_steps
    throughput = tokens_processed / elapsed

    memory_stats = get_memory_stats()

    return {
        "throughput_tokens_per_sec": throughput,
        "time_per_step_ms": (elapsed / config.benchmark_steps) * 1000,
        "memory_peak_mb": memory_stats["peak_mb"],
        "memory_allocated_mb": memory_stats["allocated_mb"],
    }


def run_benchmark():
    """Run speedrun optimization benchmark."""
    print("=" * 70)
    print("Speedrun Optimizations Benchmark")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available. Benchmarks require GPU.")
        return

    device = torch.device("cuda")
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Configuration
    bench_config = BenchmarkConfig()
    model_config = Blueberry80GBConfig()
    model_config.batch_size = bench_config.batch_size
    model_config.max_seq_len = bench_config.seq_len
    model_config.compile_model = False  # Disable for fair comparison
    model_config.use_amp = bench_config.use_amp

    print(f"\nBenchmark Configuration:")
    print(f"  Batch size: {bench_config.batch_size}")
    print(f"  Sequence length: {bench_config.seq_len}")
    print(f"  Warmup steps: {bench_config.warmup_steps}")
    print(f"  Benchmark steps: {bench_config.benchmark_steps}")
    print(f"  AMP (BFloat16): {bench_config.use_amp}")

    results = {}

    # ========================================
    # Benchmark 1: Standard MinimalLLM
    # ========================================
    print("\n" + "-" * 70)
    print("1. Standard MinimalLLM (baseline)")
    print("-" * 70)

    reset_memory()
    standard_model = MinimalLLM(model_config).to(device)
    total_params = sum(p.numel() for p in standard_model.parameters())
    print(f"   Parameters: {total_params:,}")

    result = benchmark_forward_backward(standard_model, bench_config, model_config, device)
    print(f"   Throughput: {result['throughput_tokens_per_sec']:,.0f} tokens/sec")
    print(f"   Time/step: {result['time_per_step_ms']:.2f} ms")
    print(f"   Peak memory: {result['memory_peak_mb']:.1f} MB")

    results["standard"] = result

    del standard_model
    reset_memory()

    # ========================================
    # Benchmark 2: Gradient Checkpointing Only
    # ========================================
    print("\n" + "-" * 70)
    print("2. Gradient Checkpointing (memory optimization)")
    print("-" * 70)

    reset_memory()
    checkpoint_model = SpeedrunLLM(
        model_config,
        use_checkpoint=True,
        use_relu_squared=False,
        untie_embeddings=False,
    ).to(device)

    result = benchmark_forward_backward(checkpoint_model, bench_config, model_config, device)
    print(f"   Throughput: {result['throughput_tokens_per_sec']:,.0f} tokens/sec")
    print(f"   Time/step: {result['time_per_step_ms']:.2f} ms")
    print(f"   Peak memory: {result['memory_peak_mb']:.1f} MB")

    results["checkpoint"] = result

    del checkpoint_model
    reset_memory()

    # ========================================
    # Benchmark 3: ReLU² Only
    # ========================================
    print("\n" + "-" * 70)
    print("3. ReLU² Activation (faster convergence)")
    print("-" * 70)

    reset_memory()
    relu2_model = SpeedrunLLM(
        model_config,
        use_checkpoint=False,
        use_relu_squared=True,
        untie_embeddings=False,
    ).to(device)

    result = benchmark_forward_backward(relu2_model, bench_config, model_config, device)
    print(f"   Throughput: {result['throughput_tokens_per_sec']:,.0f} tokens/sec")
    print(f"   Time/step: {result['time_per_step_ms']:.2f} ms")
    print(f"   Peak memory: {result['memory_peak_mb']:.1f} MB")

    results["relu2"] = result

    del relu2_model
    reset_memory()

    # ========================================
    # Benchmark 4: Combined Optimizations
    # ========================================
    print("\n" + "-" * 70)
    print("4. Combined: Checkpoint + ReLU² + Untied Embeddings")
    print("-" * 70)

    reset_memory()
    combined_model = SpeedrunLLM(
        model_config,
        use_checkpoint=True,
        use_relu_squared=True,
        untie_embeddings=True,
    ).to(device)

    result = benchmark_forward_backward(combined_model, bench_config, model_config, device)
    print(f"   Throughput: {result['throughput_tokens_per_sec']:,.0f} tokens/sec")
    print(f"   Time/step: {result['time_per_step_ms']:.2f} ms")
    print(f"   Peak memory: {result['memory_peak_mb']:.1f} MB")

    results["combined"] = result

    del combined_model
    reset_memory()

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    baseline = results["standard"]
    print(f"\nBaseline (Standard MinimalLLM):")
    print(f"  Throughput: {baseline['throughput_tokens_per_sec']:,.0f} tok/s")
    print(f"  Peak Memory: {baseline['memory_peak_mb']:.1f} MB")

    print(f"\nImprovements vs Baseline:")

    for name, res in results.items():
        if name == "standard":
            continue

        throughput_change = ((res['throughput_tokens_per_sec'] / baseline['throughput_tokens_per_sec']) - 1) * 100
        memory_change = ((res['memory_peak_mb'] / baseline['memory_peak_mb']) - 1) * 100
        memory_saved = baseline['memory_peak_mb'] - res['memory_peak_mb']

        print(f"\n  {name.title()}:")
        print(f"    Throughput: {throughput_change:+.1f}%")
        print(f"    Memory: {memory_change:+.1f}% ({memory_saved:+.1f} MB)")

    print("\n" + "=" * 70)
    return results


if __name__ == "__main__":
    results = run_benchmark()
