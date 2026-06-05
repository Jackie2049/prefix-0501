"""Qwen3.6-27B realistic benchmark: SDPA vs Flash Attention backend.

Usage:
    PYTHONPATH=prefix-sharing python tests/benchmark/bench_qwen36_realistic.py
"""

from __future__ import annotations

import math
import sys
import time

try:
    import torch
except ImportError:
    print("PyTorch required")
    sys.exit(1)

from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import PrefixAttentionStore


def make_rl_batch(n_prompts, responses_per_prompt, prompt_len, response_len):
    sequences = []
    for p in range(n_prompts):
        prompt = list(range(100 + p * 1000, 100 + p * 1000 + prompt_len))
        for r in range(responses_per_prompt):
            resp = [200 + p * 10000 + r * 100 + i for i in range(response_len)]
            sequences.append(prompt + resp)
    return sequences


def bench_one(name, sequences, qh, kvh, hd, device, dtype, n_runs=5):
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences)
    if not plan.has_sharing:
        return None

    seq_lens = [len(s) for s in sequences]
    total_tokens = sum(seq_lens)
    saved_pct = (total_tokens - sum(plan.kept_lengths_q)) / total_tokens * 100

    max_tid = max(max(s) for s in sequences) + 1
    k_emb = torch.randn(max_tid, kvh, hd, device=device, dtype=dtype)
    v_emb = torch.randn(max_tid, kvh, hd, device=device, dtype=dtype)
    q_per_pos = [torch.randn(sl, qh, hd, device=device, dtype=dtype) for sl in seq_lens]
    k_rows = [k_emb[seq] for seq in sequences]
    v_rows = [v_emb[seq] for seq in sequences]

    # Independent baseline
    def independent():
        for q, k, v in zip(q_per_pos, k_rows, v_rows):
            scale = math.sqrt(hd)
            if qh != kvh:
                repeat = qh // kvh
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q, k) / scale
            mask = torch.arange(k.shape[0], device=device).unsqueeze(0) <= torch.arange(q.shape[0], device=device).unsqueeze(1)
            scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            _ = torch.einsum("hqk,khd->qhd", probs, v)

    indep_ms = _bench_fn(independent, device, n_runs)

    # Prefix sharing with SDPA (torch_ref)
    trimmed_q, trimmed_k, trimmed_v = [], [], []
    for i, (q, k, v) in enumerate(zip(q_per_pos, k_rows, v_rows)):
        s, e = plan.input_keep_ranges[i]
        trimmed_q.append(q[s:e])
        trimmed_k.append(k[s:e])
        trimmed_v.append(v[s:e])
    packed_q = torch.cat(trimmed_q, dim=0)
    packed_k = torch.cat(trimmed_k, dim=0)
    packed_v = torch.cat(trimmed_v, dim=0)

    ref = TorchReferenceBackend()
    def ps_sdpa():
        store = PrefixAttentionStore()
        ek, ev = ref.build_kv(packed_k, packed_v, store, plan, layer_id=0)
        return ref.attention(packed_q, ek, ev, plan)

    sdpa_ms = _bench_fn(ps_sdpa, device, n_runs)

    # Prefix sharing with Flash Attention
    fa2 = GpuFlashAttentionBackend()
    def ps_fa2():
        store = PrefixAttentionStore()
        ek, ev = fa2.build_kv(packed_k, packed_v, store, plan, layer_id=0)
        return fa2.attention(packed_q, ek, ev, plan)

    fa2_ms = _bench_fn(ps_fa2, device, n_runs)

    return {
        "name": name,
        "saved_pct": saved_pct,
        "indep_ms": indep_ms,
        "sdpa_ms": sdpa_ms,
        "fa2_ms": fa2_ms,
        "sdpa_speedup": indep_ms / sdpa_ms if sdpa_ms > 0 else 0,
        "fa2_speedup": indep_ms / fa2_ms if fa2_ms > 0 else 0,
    }


def _bench_fn(fn, device, n_runs):
    for _ in range(3):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def main():
    device = "cuda"
    dtype = torch.bfloat16

    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    print(f"Qwen3.6-27B Realistic Benchmark | {torch.cuda.get_device_name(0)} | {dtype}")
    print("=" * 100)
    print(f"{'Config':<30} {'Saved%':>8} {'Indep(ms)':>10} {'SDPA(ms)':>10} {'FA2(ms)':>10} {'SDPA-up':>8} {'FA2-up':>8}")
    print("-" * 100)

    # Qwen3.6-27B: GQA 24:4, head_dim=256
    configs = [
        ("GRPO-8x512x128-Qwen36", 1, 8, 512, 128, 24, 4, 128),
        ("GRPO-8x1024x128-Qwen36", 1, 8, 1024, 128, 24, 4, 128),
        ("GRPO-8x1024x256-Qwen36", 1, 8, 1024, 256, 24, 4, 256),
        ("GRPO-8x2048x128-Qwen36", 1, 8, 2048, 128, 24, 4, 128),
        ("GRPO-8x2048x256-Qwen36", 1, 8, 2048, 256, 24, 4, 256),
        ("GRPO-16x1024x128-Qwen36", 1, 16, 1024, 128, 24, 4, 128),
    ]

    for name, np_, rpp, pl, rl, qh, kvh, hd in configs:
        seqs = make_rl_batch(np_, rpp, pl, rl)
        r = bench_one(name, seqs, qh, kvh, hd, device, dtype)
        if r is None:
            print(f"{name:<30} {'N/A':>6}")
            continue
        print(f"{r['name']:<30} {r['saved_pct']:>7.1f}% {r['indep_ms']:>9.2f} "
              f"{r['sdpa_ms']:>9.2f} {r['fa2_ms']:>9.2f} "
              f"{r['sdpa_speedup']:>7.2f}x {r['fa2_speedup']:>7.2f}x")

    print("=" * 100)


if __name__ == "__main__":
    main()
