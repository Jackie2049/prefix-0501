#!/usr/bin/env python3
"""Compare TorchReferenceBackend vs GpuFlashAttentionBackend performance.

Usage:
    PYTHONPATH=prefix-sharing python tests/benchmark/bench_backends.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import sys
import time

try:
    import torch
except ImportError:
    print("PyTorch required")
    sys.exit(1)

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import PrefixAttentionStore


def make_sequences(batch_size, prefix_len, suffix_len):
    prompt = list(range(100, 100 + prefix_len))
    return [prompt + [200 + i * 100 + j for j in range(suffix_len)] for i in range(batch_size)]


def bench_torch_ref(sequences, head_dim, num_q_heads, num_kv_heads, device, n_runs=10):
    """Benchmark TorchReferenceBackend."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)
    if not plan.has_sharing:
        return None

    backend = TorchReferenceBackend()
    seq_lens = [len(s) for s in sequences]

    # Token-based K/V embeddings
    max_tid = max(max(s) for s in sequences) + 1
    k_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device)
    v_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device)
    q_per_pos = [torch.randn(sl, num_q_heads, head_dim, device=device) for sl in seq_lens]
    k_rows = [k_emb[seq] for seq in sequences]
    v_rows = [v_emb[seq] for seq in sequences]

    # Trim
    trimmed_q, trimmed_k, trimmed_v = [], [], []
    for i, (q, k, v) in enumerate(zip(q_per_pos, k_rows, v_rows)):
        s, e = plan.input_keep_ranges[i]
        trimmed_q.append(q[s:e])
        trimmed_k.append(k[s:e])
        trimmed_v.append(v[s:e])

    packed_q = torch.cat(trimmed_q, dim=0)
    packed_k = torch.cat(trimmed_k, dim=0)
    packed_v = torch.cat(trimmed_v, dim=0)

    def run():
        store = PrefixAttentionStore()
        ek, ev = backend.build_kv(packed_k, packed_v, store, plan, layer_id=0)
        out = backend.attention(packed_q, ek, ev, plan)
        return out

    # Warmup
    for _ in range(3):
        run()
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return {
        "backend": "torch_ref",
        "median_ms": times[len(times) // 2],
        "min_ms": min(times),
        "max_ms": max(times),
        "plan": plan,
    }


def bench_flash_attn(sequences, head_dim, num_q_heads, num_kv_heads, device, n_runs=10):
    """Benchmark GpuFlashAttentionBackend."""
    try:
        from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
    except (ImportError, RuntimeError):
        return None

    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)
    if not plan.has_sharing:
        return None

    # FlashAttention requires 3D THD format
    backend = GpuFlashAttentionBackend()
    seq_lens = [len(s) for s in sequences]

    max_tid = max(max(s) for s in sequences) + 1
    k_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device)
    v_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device)
    q_per_pos = [torch.randn(sl, num_q_heads, head_dim, device=device) for sl in seq_lens]
    k_rows = [k_emb[seq] for seq in sequences]
    v_rows = [v_emb[seq] for seq in sequences]

    trimmed_q, trimmed_k, trimmed_v = [], [], []
    for i, (q, k, v) in enumerate(zip(q_per_pos, k_rows, v_rows)):
        s, e = plan.input_keep_ranges[i]
        trimmed_q.append(q[s:e])
        trimmed_k.append(k[s:e])
        trimmed_v.append(v[s:e])

    packed_q = torch.cat(trimmed_q, dim=0)
    packed_k = torch.cat(trimmed_k, dim=0)
    packed_v = torch.cat(trimmed_v, dim=0)

    def run():
        store = PrefixAttentionStore()
        # build_kv delegates to torch_ref
        ek, ev = backend.build_kv(packed_k, packed_v, store, plan, layer_id=0)
        # attention uses flash_attn_varlen_func
        out = backend.attention(packed_q, ek, ev, plan)
        return out

    # Warmup
    for _ in range(3):
        run()
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return {
        "backend": "flash_atten_gpu",
        "median_ms": times[len(times) // 2],
        "min_ms": min(times),
        "max_ms": max(times),
        "plan": plan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    print(f"Backend Comparison | device={args.device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU'}")
    print("=" * 80)

    configs = [
        # (name, batch_size, prefix_len, suffix_len, q_heads, kv_heads, head_dim)
        ("RL-8x128x32-GQA8:2", 8, 128, 32, 8, 2, 128),
        ("RL-8x256x32-GQA8:2", 8, 256, 32, 8, 2, 128),
        ("RL-16x128x32-GQA8:2", 16, 128, 32, 8, 2, 128),
        ("RL-8x512x64-GQA8:2", 8, 512, 64, 8, 2, 128),
        ("RL-8x128x32-MHA8:8", 8, 128, 32, 8, 8, 64),
        ("RL-32x128x32-GQA8:2", 32, 128, 32, 8, 2, 128),
    ]

    print(f"\n{'Config':<30} {'torch_ref(ms)':<15} {'flash_attn(ms)':<15} {'speedup':<10}")
    print("-" * 80)

    for name, bs, pl, sl, qh, kvh, hd in configs:
        seqs = make_sequences(bs, pl, sl)
        ref_result = bench_torch_ref(seqs, hd, qh, kvh, args.device, args.n_runs)
        flash_result = bench_flash_attn(seqs, hd, qh, kvh, args.device, args.n_runs)

        ref_ms = ref_result["median_ms"] if ref_result else float("nan")
        flash_ms = flash_result["median_ms"] if flash_result else float("nan")
        speedup = ref_ms / flash_ms if flash_result and flash_ms > 0 else float("nan")

        flash_label = f"{flash_ms:.2f}" if flash_result else "N/A"
        print(f"{name:<30} {ref_ms:<15.2f} {flash_label:<15} {speedup:<10.2f}")


if __name__ == "__main__":
    main()
