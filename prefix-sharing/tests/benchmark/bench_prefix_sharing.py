"""Performance benchmark for prefix-sharing attention backend.

Measures:
1. End-to-end time for prefix-sharing forward vs independent forward
2. Build KV time (provider store + reuser load)
3. Attention compute time
4. Speedup ratio = independent_time / prefix_sharing_time

Usage:
    PYTHONPATH=prefix-sharing python -m tests.benchmark.bench_prefix_sharing
    # Or on GPU:
    PYTHONPATH=prefix-sharing python -m tests.benchmark.bench_prefix_sharing --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import torch
except ImportError:
    print("PyTorch required for benchmarks")
    sys.exit(1)

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import PrefixAttentionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    prefix_len: int
    suffix_len: int
    batch_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    device: str

    # Times in milliseconds
    independent_ms: float = 0.0
    ps_build_kv_ms: float = 0.0
    ps_attention_ms: float = 0.0
    ps_total_ms: float = 0.0
    speedup: float = 0.0

    # Token counts
    total_tokens_original: int = 0
    total_tokens_trimmed: int = 0
    tokens_saved_pct: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _make_sequences(batch_size: int, prefix_len: int, suffix_len: int) -> list[list[int]]:
    """Create RL-like batch: shared prompt + different responses."""
    prompt = list(range(100, 100 + prefix_len))
    sequences = []
    for i in range(batch_size):
        response = [200 + i * 100 + j for j in range(suffix_len)]
        sequences.append(prompt + response)
    return sequences


def _warmup(device: str, n: int = 5):
    """Warm up GPU kernels."""
    for _ in range(n):
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        _ = a @ b
    if device == "cuda":
        torch.cuda.synchronize()


def _benchmark_fn(fn, *args, n_runs: int = 10, device: str = "cpu", **kwargs) -> float:
    """Benchmark a function, return median time in milliseconds."""
    times = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]  # Median


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def bench_single_config(
    batch_size: int,
    prefix_len: int,
    suffix_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: str,
    n_runs: int = 10,
) -> BenchResult:
    """Benchmark a single configuration."""
    result = BenchResult(
        name=f"B{batch_size}_P{prefix_len}_S{suffix_len}_H{num_q_heads}x{head_dim}",
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device=device,
    )

    sequences = _make_sequences(batch_size, prefix_len, suffix_len)
    total_len = prefix_len + suffix_len
    result.total_tokens_original = batch_size * total_len

    # Plan
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    planner = PrefixSharingPlanner(config)
    plan = planner.plan(sequences, forward_id=0, micro_batch_id=0)
    result.total_tokens_trimmed = sum(plan.kept_lengths_q)
    result.tokens_saved_pct = (
        (result.total_tokens_original - result.total_tokens_trimmed)
        / result.total_tokens_original * 100
    )

    if not plan.has_sharing:
        print(f"  [SKIP] No sharing detected for {result.name}")
        return result

    backend = TorchReferenceBackend()

    # Generate QKV tensors
    all_q = torch.randn(result.total_tokens_original, num_q_heads, head_dim, device=device)
    all_k = torch.randn(result.total_tokens_original, num_kv_heads, head_dim, device=device)
    all_v = torch.randn(result.total_tokens_original, num_kv_heads, head_dim, device=device)

    seq_lens = [total_len] * batch_size
    q_rows = list(torch.split(all_q, seq_lens))
    k_rows = list(torch.split(all_k, seq_lens))
    v_rows = list(torch.split(all_v, seq_lens))

    # --- Benchmark independent forward ---
    def independent_forward():
        outputs = []
        for q, k, v in zip(q_rows, k_rows, v_rows):
            import math
            scale = math.sqrt(head_dim)
            if num_q_heads != num_kv_heads:
                repeat = num_q_heads // num_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q, k) / scale
            mask = torch.arange(k.shape[0], device=device).unsqueeze(0) <= torch.arange(q.shape[0], device=device).unsqueeze(1)
            scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hqk,khd->qhd", probs, v)
            outputs.append(out)

    result.independent_ms = _benchmark_fn(independent_forward, n_runs=n_runs, device=device)

    # --- Benchmark prefix-sharing forward ---
    # Prepare trimmed tensors
    trimmed_q_rows = []
    trimmed_k_rows = []
    trimmed_v_rows = []
    for i, (q_row, k_row, v_row) in enumerate(zip(q_rows, k_rows, v_rows)):
        keep_start, keep_end = plan.input_keep_ranges[i]
        trimmed_q_rows.append(q_row[keep_start:keep_end].clone())
        trimmed_k_rows.append(k_row[keep_start:keep_end].clone())
        trimmed_v_rows.append(v_row[keep_start:keep_end].clone())

    packed_q = torch.cat(trimmed_q_rows, dim=0)
    packed_k = torch.cat(trimmed_k_rows, dim=0)
    packed_v = torch.cat(trimmed_v_rows, dim=0)

    # Benchmark build_kv
    def build_kv_fn():
        store = PrefixAttentionStore()
        return backend.build_kv(packed_k, packed_v, store, plan, layer_id=0)

    result.ps_build_kv_ms = _benchmark_fn(build_kv_fn, n_runs=n_runs, device=device)

    # Build expanded KV once for attention benchmark
    store = PrefixAttentionStore()
    expanded_k, expanded_v = backend.build_kv(packed_k, packed_v, store, plan, layer_id=0)

    # Benchmark attention
    def attention_fn():
        return backend.attention(packed_q, expanded_k, expanded_v, plan)

    result.ps_attention_ms = _benchmark_fn(attention_fn, n_runs=n_runs, device=device)

    # Total prefix-sharing time
    result.ps_total_ms = result.ps_build_kv_ms + result.ps_attention_ms
    if result.ps_total_ms > 0:
        result.speedup = result.independent_ms / result.ps_total_ms

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prefix-sharing performance benchmark")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--output", default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Prefix-Sharing Benchmark | device={args.device}")
    print("=" * 100)

    _warmup(args.device)

    results = []

    # --- Sweep 1: Varying prefix length ---
    print("\n--- Sweep 1: Varying prefix length (B=8, S=32, H=8x64, GQA 8:2) ---")
    for prefix_len in [0, 16, 32, 64, 128, 256, 512]:
        r = bench_single_config(
            batch_size=8, prefix_len=prefix_len, suffix_len=32,
            num_q_heads=8, num_kv_heads=2, head_dim=64,
            device=args.device, n_runs=args.n_runs,
        )
        results.append(r)
        print(f"  P={prefix_len:4d} | saved={r.tokens_saved_pct:5.1f}% | "
              f"indep={r.independent_ms:7.2f}ms | ps_total={r.ps_total_ms:7.2f}ms | "
              f"speedup={r.speedup:.2f}x")

    # --- Sweep 2: Varying batch size ---
    print("\n--- Sweep 2: Varying batch size (P=128, S=32, H=8x64, GQA 8:2) ---")
    for batch_size in [2, 4, 8, 16, 32]:
        r = bench_single_config(
            batch_size=batch_size, prefix_len=128, suffix_len=32,
            num_q_heads=8, num_kv_heads=2, head_dim=64,
            device=args.device, n_runs=args.n_runs,
        )
        results.append(r)
        print(f"  B={batch_size:2d} | saved={r.tokens_saved_pct:5.1f}% | "
              f"indep={r.independent_ms:7.2f}ms | ps_total={r.ps_total_ms:7.2f}ms | "
              f"speedup={r.speedup:.2f}x")

    # --- Sweep 3: Varying suffix length ---
    print("\n--- Sweep 3: Varying suffix length (B=8, P=128, H=8x64, GQA 8:2) ---")
    for suffix_len in [8, 16, 32, 64, 128]:
        r = bench_single_config(
            batch_size=8, prefix_len=128, suffix_len=suffix_len,
            num_q_heads=8, num_kv_heads=2, head_dim=64,
            device=args.device, n_runs=args.n_runs,
        )
        results.append(r)
        print(f"  S={suffix_len:3d} | saved={r.tokens_saved_pct:5.1f}% | "
              f"indep={r.independent_ms:7.2f}ms | ps_total={r.ps_total_ms:7.2f}ms | "
              f"speedup={r.speedup:.2f}x")

    # --- Sweep 4: MHA vs GQA ---
    print("\n--- Sweep 4: MHA vs GQA (B=8, P=128, S=32, H=8x64) ---")
    for num_kv_heads in [8, 4, 2, 1]:
        r = bench_single_config(
            batch_size=8, prefix_len=128, suffix_len=32,
            num_q_heads=8, num_kv_heads=num_kv_heads, head_dim=64,
            device=args.device, n_runs=args.n_runs,
        )
        results.append(r)
        label = f"MHA(8:{num_kv_heads})" if num_kv_heads == 8 else f"GQA(8:{num_kv_heads})"
        print(f"  {label:12s} | saved={r.tokens_saved_pct:5.1f}% | "
              f"indep={r.independent_ms:7.2f}ms | ps_total={r.ps_total_ms:7.2f}ms | "
              f"speedup={r.speedup:.2f}x")

    # --- Summary ---
    print("\n" + "=" * 100)
    print("Summary:")
    valid = [r for r in results if r.speedup > 0]
    if valid:
        print(f"  Average speedup: {sum(r.speedup for r in valid) / len(valid):.2f}x")
        print(f"  Max speedup: {max(r.speedup for r in valid):.2f}x")
        print(f"  Min speedup: {min(r.speedup for r in valid):.2f}x")
        avg_saved = sum(r.tokens_saved_pct for r in valid) / len(valid)
        print(f"  Average token savings: {avg_saved:.1f}%")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
