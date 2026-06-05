"""RL workload benchmark: simulate realistic GRPO/PPO training scenarios.

Usage:
    PYTHONPATH=prefix-sharing python tests/benchmark/bench_rl_workload.py --device cuda
"""

from __future__ import annotations

import argparse
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


def make_rl_batch(n_prompts, responses_per_prompt, prompt_len, response_len):
    """Create RL-like batch with shared prompts."""
    sequences = []
    for p in range(n_prompts):
        prompt = list(range(100 + p * 1000, 100 + p * 1000 + prompt_len))
        for r in range(responses_per_prompt):
            resp = [200 + p * 10000 + r * 100 + i for i in range(response_len)]
            sequences.append(prompt + resp)
    return sequences


def bench_workload(sequences, num_q_heads, num_kv_heads, head_dim, device, n_runs=10):
    """Benchmark a workload: independent vs prefix-sharing."""
    import math

    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)

    if not plan.has_sharing:
        return None

    backend = TorchReferenceBackend()
    seq_lens = [len(s) for s in sequences]
    total_tokens = sum(seq_lens)

    # Token-ID-based K/V
    max_tid = max(max(s) for s in sequences) + 1
    k_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    v_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    q_per_pos = [torch.randn(sl, num_q_heads, head_dim, device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32) for sl in seq_lens]
    k_rows = [k_emb[seq] for seq in sequences]
    v_rows = [v_emb[seq] for seq in sequences]

    # --- Independent ---
    def independent():
        for q, k, v in zip(q_per_pos, k_rows, v_rows):
            scale = math.sqrt(head_dim)
            if num_q_heads != num_kv_heads:
                repeat = num_q_heads // num_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q, k) / scale
            mask = torch.arange(k.shape[0], device=device).unsqueeze(0) <= torch.arange(q.shape[0], device=device).unsqueeze(1)
            scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            _ = torch.einsum("hqk,khd->qhd", probs, v)

    # Warmup
    for _ in range(3):
        independent()
    torch.cuda.synchronize()

    ind_times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        independent()
        torch.cuda.synchronize()
        ind_times.append((time.perf_counter() - t0) * 1000)
    ind_times.sort()

    # --- Prefix-sharing ---
    trimmed_q, trimmed_k, trimmed_v = [], [], []
    for i, (q, k, v) in enumerate(zip(q_per_pos, k_rows, v_rows)):
        s, e = plan.input_keep_ranges[i]
        trimmed_q.append(q[s:e])
        trimmed_k.append(k[s:e])
        trimmed_v.append(v[s:e])

    packed_q = torch.cat(trimmed_q, dim=0)
    packed_k = torch.cat(trimmed_k, dim=0)
    packed_v = torch.cat(trimmed_v, dim=0)

    def prefix_sharing():
        store = PrefixAttentionStore()
        ek, ev = backend.build_kv(packed_k, packed_v, store, plan, layer_id=0)
        return backend.attention(packed_q, ek, ev, plan)

    for _ in range(3):
        prefix_sharing()
    torch.cuda.synchronize()

    ps_times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefix_sharing()
        torch.cuda.synchronize()
        ps_times.append((time.perf_counter() - t0) * 1000)
    ps_times.sort()

    saved = total_tokens - sum(plan.kept_lengths_q)
    return {
        "total_tokens": total_tokens,
        "kept_tokens": sum(plan.kept_lengths_q),
        "saved_pct": saved / total_tokens * 100,
        "indep_ms": ind_times[len(ind_times) // 2],
        "ps_ms": ps_times[len(ps_times) // 2],
        "speedup": ind_times[len(ind_times) // 2] / ps_times[len(ps_times) // 2],
        "batch_size": len(sequences),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--n-runs", type=int, default=10)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    print(f"RL Workload Benchmark | device={args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 90)

    # Qwen3.6-27B full attention config: GQA 24:4, head_dim=256
    # Using smaller head counts for faster benchmark
    configs = [
        # (name, n_prompts, resp_per_prompt, prompt_len, resp_len, q_heads, kv_heads, head_dim)
        ("GRPO-8x512x64-GQA8:2", 1, 8, 512, 64, 8, 2, 128),
        ("GRPO-8x512x128-GQA8:2", 1, 8, 512, 128, 8, 2, 128),
        ("GRPO-16x512x64-GQA8:2", 1, 16, 512, 64, 8, 2, 128),
        ("PPO-4x1024x128-GQA8:2", 4, 1, 1024, 128, 8, 2, 128),
        ("GRPO-8x1024x128-GQA8:2", 1, 8, 1024, 128, 8, 2, 128),
        ("GRPO-8x2048x128-GQA8:2", 1, 8, 2048, 128, 8, 2, 128),
        # Qwen3.6-like GQA 24:4
        ("GRPO-8x512x64-Qwen36", 1, 8, 512, 64, 24, 4, 64),
        ("GRPO-8x1024x128-Qwen36", 1, 8, 1024, 128, 24, 4, 64),
    ]

    print(f"\n{'Config':<30} {'Batch':>6} {'Saved%':>8} {'Indep(ms)':>10} {'PS(ms)':>10} {'Speedup':>8}")
    print("-" * 90)

    for name, np_, rpp, pl, rl, qh, kvh, hd in configs:
        seqs = make_rl_batch(np_, rpp, pl, rl)
        r = bench_workload(seqs, qh, kvh, hd, args.device, args.n_runs)
        if r is None:
            print(f"{name:<30} {'N/A':>6}")
            continue
        print(f"{name:<30} {r['batch_size']:>6} {r['saved_pct']:>7.1f}% "
              f"{r['indep_ms']:>9.2f} {r['ps_ms']:>9.2f} {r['speedup']:>7.2f}x")

    print("=" * 90)


if __name__ == "__main__":
    main()
