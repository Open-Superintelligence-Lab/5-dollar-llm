#!/usr/bin/env python3
"""
Benchmark script comparing standard vs Liger Kernel optimized LLM training.

This script measures:
1. Forward pass throughput (tokens/second)
2. Forward+backward pass throughput
3. Peak memory usage
4. Loss computation time

Usage:
    python benchmarks/liger_benchmark.py

Expected improvements with Liger Kernel:
- Throughput: +20%
- Memory: -60%
- FusedLinearCrossEntropy: -80% memory on loss computation
"""

import torch
import torch.nn.functional as F
import time
import gc
import argparse
from dataclasses import dataclass
from typing import Optional

# Add project root to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.llm_config import Blueberry80GBConfig
from models.llm import MinimalLLM

# Try to import Liger model
try:
    from models.llm_liger import LigerLLM, LIGER_AVAILABLE
except ImportError:
    LIGER_AVAILABLE = False
    LigerLLM = None


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    batch_size: int = 2  # Reduced for RTX 3060 12GB
    seq_len: int = 1024  # Reduced for memory
    warmup_steps: int = 3
    benchmark_steps: int = 10
    use_amp: bool = True
    compile_model: bool = False  # Disable for fair comparison


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


def benchmark_forward(
    model: torch.nn.Module,
    config: BenchmarkConfig,
    model_config: Blueberry80GBConfig,
    device: torch.device,
) -> dict:
    """Benchmark forward pass only."""
    model.eval()
    reset_memory()

    # Create dummy input
    input_ids = torch.randint(0, model_config.vocab_size, (config.batch_size, config.seq_len), device=device)

    # Warmup
    for _ in range(config.warmup_steps):
        with torch.no_grad():
            if config.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = model(input_ids)
            else:
                _ = model(input_ids)

    torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()

    for _ in range(config.benchmark_steps):
        with torch.no_grad():
            if config.use_amp:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    _ = model(input_ids)
            else:
                _ = model(input_ids)

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


def benchmark_forward_backward(
    model: torch.nn.Module,
    config: BenchmarkConfig,
    model_config: Blueberry80GBConfig,
    device: torch.device,
    use_fused_loss: bool = False,
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
                if use_fused_loss and hasattr(model, 'forward_with_loss'):
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
        else:
            if use_fused_loss and hasattr(model, 'forward_with_loss'):
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

        model.zero_grad()

    torch.cuda.synchronize()
    reset_memory()

    # Benchmark
    start_time = time.perf_counter()

    for _ in range(config.benchmark_steps):
        if config.use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if use_fused_loss and hasattr(model, 'forward_with_loss'):
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
        else:
            if use_fused_loss and hasattr(model, 'forward_with_loss'):
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
    """Run complete benchmark comparison."""
    print("=" * 70)
    print("🔬 Liger Kernel Benchmark: Standard vs Optimized LLM Training")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA not available. Benchmarks require GPU.")
        return

    device = torch.device("cuda")
    print(f"\n📊 Device: {torch.cuda.get_device_name()}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Configuration
    bench_config = BenchmarkConfig()
    model_config = Blueberry80GBConfig()
    model_config.batch_size = bench_config.batch_size
    model_config.max_seq_len = bench_config.seq_len
    model_config.compile_model = bench_config.compile_model
    model_config.use_amp = bench_config.use_amp

    print(f"\n⚙️  Benchmark Configuration:")
    print(f"   Batch size: {bench_config.batch_size}")
    print(f"   Sequence length: {bench_config.seq_len}")
    print(f"   Warmup steps: {bench_config.warmup_steps}")
    print(f"   Benchmark steps: {bench_config.benchmark_steps}")
    print(f"   AMP (BFloat16): {bench_config.use_amp}")
    print(f"   Vocab size: {model_config.vocab_size}")

    # Calculate theoretical logits size
    logits_size_mb = (bench_config.batch_size * bench_config.seq_len * model_config.vocab_size * 2) / 1024 / 1024
    print(f"   Logits tensor size: {logits_size_mb:.1f} MB (eliminated by FusedLinearCrossEntropy)")

    results = {}

    # ========================================
    # Benchmark 1: Standard MinimalLLM
    # ========================================
    print("\n" + "-" * 70)
    print("📦 Benchmark 1: Standard MinimalLLM")
    print("-" * 70)

    reset_memory()
    standard_model = MinimalLLM(model_config).to(device)
    total_params = sum(p.numel() for p in standard_model.parameters())
    print(f"   Parameters: {total_params:,}")

    print("\n   Forward pass only:")
    fwd_results = benchmark_forward(standard_model, bench_config, model_config, device)
    print(f"   - Throughput: {fwd_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
    print(f"   - Time/step: {fwd_results['time_per_step_ms']:.2f} ms")
    print(f"   - Peak memory: {fwd_results['memory_peak_mb']:.1f} MB")

    print("\n   Forward + Backward (standard loss):")
    fwd_bwd_results = benchmark_forward_backward(standard_model, bench_config, model_config, device)
    print(f"   - Throughput: {fwd_bwd_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
    print(f"   - Time/step: {fwd_bwd_results['time_per_step_ms']:.2f} ms")
    print(f"   - Peak memory: {fwd_bwd_results['memory_peak_mb']:.1f} MB")

    results["standard"] = {
        "forward": fwd_results,
        "forward_backward": fwd_bwd_results,
    }

    del standard_model
    reset_memory()

    # ========================================
    # Benchmark 2: Liger LLM (if available)
    # ========================================
    if LIGER_AVAILABLE and LigerLLM is not None:
        print("\n" + "-" * 70)
        print("🚀 Benchmark 2: Liger Kernel Optimized LLM")
        print("-" * 70)

        reset_memory()
        liger_model = LigerLLM(model_config, use_liger=True).to(device)
        print(f"   Parameters: {sum(p.numel() for p in liger_model.parameters()):,}")
        print(f"   Liger enabled: {liger_model.use_liger}")

        print("\n   Forward pass only:")
        fwd_results = benchmark_forward(liger_model, bench_config, model_config, device)
        print(f"   - Throughput: {fwd_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
        print(f"   - Time/step: {fwd_results['time_per_step_ms']:.2f} ms")
        print(f"   - Peak memory: {fwd_results['memory_peak_mb']:.1f} MB")

        print("\n   Forward + Backward (standard loss):")
        fwd_bwd_results = benchmark_forward_backward(liger_model, bench_config, model_config, device, use_fused_loss=False)
        print(f"   - Throughput: {fwd_bwd_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
        print(f"   - Time/step: {fwd_bwd_results['time_per_step_ms']:.2f} ms")
        print(f"   - Peak memory: {fwd_bwd_results['memory_peak_mb']:.1f} MB")

        print("\n   Forward + Backward (FUSED loss - no logits materialization!):")
        fwd_bwd_fused_results = benchmark_forward_backward(liger_model, bench_config, model_config, device, use_fused_loss=True)
        print(f"   - Throughput: {fwd_bwd_fused_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
        print(f"   - Time/step: {fwd_bwd_fused_results['time_per_step_ms']:.2f} ms")
        print(f"   - Peak memory: {fwd_bwd_fused_results['memory_peak_mb']:.1f} MB")

        results["liger"] = {
            "forward": fwd_results,
            "forward_backward": fwd_bwd_results,
            "forward_backward_fused": fwd_bwd_fused_results,
        }

        del liger_model
        reset_memory()
    else:
        print("\n⚠️  Liger Kernel not available, skipping Liger benchmarks")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 70)

    if "standard" in results:
        std = results["standard"]
        print(f"\n🔵 Standard MinimalLLM:")
        print(f"   Forward throughput: {std['forward']['throughput_tokens_per_sec']:,.0f} tok/s")
        print(f"   Fwd+Bwd throughput: {std['forward_backward']['throughput_tokens_per_sec']:,.0f} tok/s")
        print(f"   Fwd+Bwd peak memory: {std['forward_backward']['memory_peak_mb']:.1f} MB")

    if "liger" in results:
        lig = results["liger"]
        print(f"\n🟢 Liger Optimized LLM:")
        print(f"   Forward throughput: {lig['forward']['throughput_tokens_per_sec']:,.0f} tok/s")
        print(f"   Fwd+Bwd throughput: {lig['forward_backward']['throughput_tokens_per_sec']:,.0f} tok/s")
        print(f"   Fwd+Bwd peak memory: {lig['forward_backward']['memory_peak_mb']:.1f} MB")
        print(f"\n🟡 Liger with FusedLinearCrossEntropy:")
        print(f"   Fwd+Bwd throughput: {lig['forward_backward_fused']['throughput_tokens_per_sec']:,.0f} tok/s")
        print(f"   Fwd+Bwd peak memory: {lig['forward_backward_fused']['memory_peak_mb']:.1f} MB")

        # Calculate improvements
        if "standard" in results:
            std_mem = std['forward_backward']['memory_peak_mb']
            lig_mem = lig['forward_backward']['memory_peak_mb']
            fused_mem = lig['forward_backward_fused']['memory_peak_mb']

            std_throughput = std['forward_backward']['throughput_tokens_per_sec']
            lig_throughput = lig['forward_backward']['throughput_tokens_per_sec']
            fused_throughput = lig['forward_backward_fused']['throughput_tokens_per_sec']

            print(f"\n📈 IMPROVEMENTS:")
            print(f"   Liger vs Standard:")
            print(f"   - Throughput: {((lig_throughput / std_throughput) - 1) * 100:+.1f}%")
            print(f"   - Memory: {((lig_mem / std_mem) - 1) * 100:+.1f}%")
            print(f"\n   Liger+Fused vs Standard:")
            print(f"   - Throughput: {((fused_throughput / std_throughput) - 1) * 100:+.1f}%")
            print(f"   - Memory: {((fused_mem / std_mem) - 1) * 100:+.1f}%")
            print(f"   - Memory saved: {std_mem - fused_mem:.1f} MB")

    print("\n" + "=" * 70)
    return results


if __name__ == "__main__":
    results = run_benchmark()
