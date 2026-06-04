"""Tensor Parallelism test: verify prefix-sharing correctness under TP.

This test simulates how prefix-sharing KV tensors would be split across
TP ranks, verifying that the attention output remains numerically identical
regardless of the number of TP ranks.

Usage (single GPU test):
    PYTHONPATH=prefix-sharing python tests/benchmark/bench_tp.py

Usage (multi-GPU TP test):
    PYTHONPATH=prefix-sharing torchrun --nproc_per_node=2 tests/benchmark/bench_tp.py --tp 2
    PYTHONPATH=prefix-sharing torchrun --nproc_per_node=4 tests/benchmark/bench_tp.py --tp 4
"""

from __future__ import annotations

import argparse
import math
import sys
import time

try:
    import torch
    import torch.distributed as dist
except ImportError:
    print("PyTorch required")
    sys.exit(1)

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


def independent_attention(q_rows, k_rows, v_rows, device):
    """Run independent attention for each sequence."""
    outputs = []
    for q, k, v in zip(q_rows, k_rows, v_rows):
        q_heads, kv_heads = q.shape[1], k.shape[1]
        scale = math.sqrt(q.shape[-1])
        # Handle GQA and TP partitioning
        if q_heads > kv_heads:
            repeat = q_heads // kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        elif kv_heads > q_heads:
            repeat = kv_heads // q_heads
            q = q.repeat_interleave(repeat, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) / scale
        mask = torch.arange(k.shape[0], device=device).unsqueeze(0) <= torch.arange(q.shape[0], device=device).unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, v))
    return outputs


def test_tp_simulation():
    """Simulate TP by splitting heads across ranks.

    In Megatron with TP, attention heads are partitioned across ranks.
    Each rank computes attention for its subset of heads, then an
    AllReduce combines the output projection. This test verifies that
    prefix-sharing produces correct results for each TP partition.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Initialize distributed if available
    rank = 0
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    elif args.tp > 1:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()

    tp_size = args.tp
    tp_rank = rank % tp_size

    if rank == 0:
        print(f"TP Simulation | tp_size={tp_size} | world_size={world_size} | rank={rank}")
        print("=" * 70)

    device = f"cuda:{rank}" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    # Model config: total heads = 8, split across TP ranks
    total_q_heads = 8
    total_kv_heads = 2
    head_dim = 64

    if total_q_heads % tp_size != 0:
        if rank == 0:
            print(f"ERROR: total_q_heads={total_q_heads} not divisible by tp_size={tp_size}")
        sys.exit(1)

    q_heads_per_rank = total_q_heads // tp_size
    # In Megatron TP, KV heads are replicated when tp_size > num_kv_heads
    if total_kv_heads >= tp_size:
        kv_heads_per_rank = total_kv_heads // tp_size
        kv_rank_offset = tp_rank * kv_heads_per_rank
    else:
        # All ranks get all KV heads (replicated)
        kv_heads_per_rank = total_kv_heads
        kv_rank_offset = 0

    # Create sequences
    prompt = list(range(100, 228))  # 128 tokens
    sequences = [prompt + [200+i*100+j for j in range(32)] for i in range(8)]

    torch.manual_seed(42 + rank)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)

    if rank == 0:
        print(f"Sharing: {plan.has_sharing}, saved: {sum(plan.original_lengths)-sum(plan.kept_lengths_q)}/{sum(plan.original_lengths)} tokens")

    seq_lens = [len(s) for s in sequences]
    max_tid = max(max(s) for s in sequences) + 1

    # Full K/V embeddings (same across all ranks since embeddings are shared)
    k_emb = torch.randn(max_tid, total_kv_heads, head_dim, device=device)
    v_emb = torch.randn(max_tid, total_kv_heads, head_dim, device=device)

    # Each rank gets its head partition for Q
    q_per_pos_full = [torch.randn(sl, total_q_heads, head_dim, device=device) for sl in seq_lens]

    # Slice Q to this rank's heads
    q_per_pos = [q[:, tp_rank*q_heads_per_rank:(tp_rank+1)*q_heads_per_rank, :] for q in q_per_pos_full]

    # Slice K/V to this rank's KV heads
    k_rows_full = [k_emb[seq] for seq in sequences]
    v_rows_full = [v_emb[seq] for seq in sequences]
    k_rows = [k[:, kv_rank_offset:kv_rank_offset+kv_heads_per_rank, :] for k in k_rows_full]
    v_rows = [v[:, kv_rank_offset:kv_rank_offset+kv_heads_per_rank, :] for v in v_rows_full]

    # Independent reference (per-rank heads)
    ind_outs = independent_attention(q_per_pos, k_rows, v_rows, device)

    # Prefix-sharing
    backend = TorchReferenceBackend()
    store = PrefixAttentionStore()

    trimmed_q, trimmed_k, trimmed_v = [], [], []
    for i, (q, k, v) in enumerate(zip(q_per_pos, k_rows, v_rows)):
        s, e = plan.input_keep_ranges[i]
        trimmed_q.append(q[s:e])
        trimmed_k.append(k[s:e])
        trimmed_v.append(v[s:e])

    packed_q = torch.cat(trimmed_q, dim=0)
    packed_k = torch.cat(trimmed_k, dim=0)
    packed_v = torch.cat(trimmed_v, dim=0)

    expanded_k, expanded_v = backend.build_kv(packed_k, packed_v, store, plan, layer_id=0, tp_rank=tp_rank)
    ps_output = backend.attention(packed_q, expanded_k, expanded_v, plan)
    ps_rows = list(torch.split(ps_output, plan.kept_lengths_q))

    # Verify
    all_pass = True
    for i in range(len(sequences)):
        s, e = plan.input_keep_ranges[i]
        if not torch.allclose(ps_rows[i], ind_outs[i][s:e], atol=1e-5, rtol=1e-4):
            max_diff = (ps_rows[i] - ind_outs[i][s:e]).abs().max().item()
            print(f"  [rank={rank}] FAIL seq{i}: max_diff={max_diff:.2e}")
            all_pass = False

    if all_pass:
        print(f"  [rank={rank}] TP={tp_size} rank={rank}: ALL PASS (q_heads={q_heads_per_rank}, kv_heads={kv_heads_per_rank})")

    if world_size > 1:
        dist.barrier()

    return all_pass


if __name__ == "__main__":
    ok = test_tp_simulation()
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0 if ok else 1)
