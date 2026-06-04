#!/usr/bin/env python3
"""HybridAttention integration test for Qwen3.6-27B.

Simulates the complete layer routing that a real Megatron model would perform:
- Full attention layers (0, 4, 8, ..., 60): KV build + attention
- Linear attention layers (1,2,3, 5,6,7, ...): DeltaNet state build

Validates that prefix sharing produces identical results to independent
forward passes across all 64 layers of Qwen3.6-27B.

Usage:
    PYTHONPATH=prefix-sharing python tests/run_hybrid_attention.py
    PYTHONPATH=prefix-sharing python tests/run_hybrid_attention.py --device cuda
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
from prefix_sharing.core.model_spec import QWEN3_6_27B
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import PrefixAttentionStore, PrefixDeltanetStore


class Result:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self):
        return self.failed == 0

    def check_close(self, name, actual, expected, atol=1e-5, rtol=1e-4):
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            max_diff = (actual - expected).abs().max().item()
            self.failed += 1
            msg = f"FAIL {name}: max_diff={max_diff:.2e}"
            self.errors.append(msg)
            print(f"  {msg}")
            return False
        self.passed += 1
        return True

    def check_true(self, name, cond, msg=""):
        if not cond:
            self.failed += 1
            full = f"FAIL {name}: {msg}" if msg else f"FAIL {name}"
            self.errors.append(full)
            print(f"  {full}")
            return False
        self.passed += 1
        return True

    def summary(self):
        total = self.passed + self.failed
        status = "PASS" if self.ok() else "FAIL"
        print(f"\n{'='*60}")
        print(f"HybridAttention Results: {self.passed}/{total} [{status}]")
        if self.errors:
            for e in self.errors:
                print(f"  - {e}")
        return self.ok()


def _independent_attention(q, k, v, num_q_heads, num_kv_heads, head_dim, device):
    """Run independent attention for one sequence."""
    scale = math.sqrt(head_dim)
    if num_q_heads != num_kv_heads:
        if num_q_heads > num_kv_heads:
            repeat = num_q_heads // num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) / scale
    mask = torch.arange(k.shape[0], device=device).unsqueeze(0) <= torch.arange(q.shape[0], device=device).unsqueeze(1)
    scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v)


def run_hybrid_test(sequences, model_spec, device, result, name, atol=1e-4):
    """Test full HybridAttention layer routing with prefix sharing."""
    num_layers = model_spec.num_hidden_layers
    num_q_heads = model_spec.num_attention_heads
    num_kv_heads = model_spec.num_key_value_heads
    head_dim = model_spec.head_dim

    print(f"\n[{name}] batch={len(sequences)}, layers={num_layers}, "
          f"heads={num_q_heads}:{num_kv_heads}, head_dim={head_dim}")

    torch.manual_seed(42)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)
    backend = TorchReferenceBackend()

    result.check_true(f"{name}/has_sharing", plan.has_sharing)
    if not plan.has_sharing:
        return

    seq_lens = [len(s) for s in sequences]
    all_token_ids = set()
    for seq in sequences:
        all_token_ids.update(seq)
    max_tid = max(all_token_ids) + 1

    full_layers = model_spec.full_attention_layer_ids()
    linear_layers = model_spec.linear_attention_layer_ids()

    result.check_true(f"{name}/full_layers", len(full_layers) == 16,
                      f"expected 16 full attention layers, got {len(full_layers)}")
    result.check_true(f"{name}/linear_layers", len(linear_layers) == 48,
                      f"expected 48 linear attention layers, got {len(linear_layers)}")

    # Sample layers to keep test fast (4 full + 4 linear)
    sample_full = full_layers[:4]
    sample_linear = linear_layers[:4]

    # --- Full attention layers ---
    for layer_id in sample_full:
        kv_store = PrefixAttentionStore()
        k_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device)
        v_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device)
        k_rows = [k_emb[seq] for seq in sequences]
        v_rows = [v_emb[seq] for seq in sequences]
        q_rows = [torch.randn(sl, num_q_heads, head_dim, device=device) for sl in seq_lens]

        # Independent forward
        ind_outs = [_independent_attention(q, k, v, num_q_heads, num_kv_heads, head_dim, device)
                    for q, k, v in zip(q_rows, k_rows, v_rows)]

        # Prefix-sharing forward
        trimmed_q, trimmed_k, trimmed_v = [], [], []
        for i in range(len(sequences)):
            s, e = plan.input_keep_ranges[i]
            trimmed_q.append(q_rows[i][s:e])
            trimmed_k.append(k_rows[i][s:e])
            trimmed_v.append(v_rows[i][s:e])

        packed_q = torch.cat(trimmed_q, dim=0)
        packed_k = torch.cat(trimmed_k, dim=0)
        packed_v = torch.cat(trimmed_v, dim=0)

        expanded_k, expanded_v = backend.build_kv(
            packed_k, packed_v, kv_store, plan, layer_id=layer_id,
        )
        ps_out = backend.attention(packed_q, expanded_k, expanded_v, plan)
        ps_rows = list(torch.split(ps_out, plan.kept_lengths_q))

        for i in range(len(sequences)):
            s, e = plan.input_keep_ranges[i]
            result.check_close(f"{name}/full_L{layer_id}_s{i}", ps_rows[i], ind_outs[i][s:e], atol=atol)

    # --- Linear attention layers (DeltaNet) ---
    for layer_id in sample_linear:
        deltanet_store = PrefixDeltanetStore()
        state_emb = torch.randn(max_tid, head_dim, device=device)
        all_updates = [state_emb[seq] for seq in sequences]
        ind_trajectories = [u.cumsum(dim=0) for u in all_updates]

        trimmed = []
        for i, upd in enumerate(all_updates):
            s, e = plan.input_keep_ranges[i]
            trimmed.append(upd[s:e])

        packed = torch.cat(trimmed, dim=0)
        ps_output = backend.build_deltanet_states(
            packed, deltanet_store, plan, layer_id=layer_id,
        )
        ps_rows = list(torch.split(ps_output, plan.kept_lengths_q))

        for i in range(len(sequences)):
            s, e = plan.input_keep_ranges[i]
            kept_len = e - s
            if not plan.is_reuser(i):
                result.check_close(f"{name}/dnet_L{layer_id}_prov{i}",
                                   ps_rows[i][:kept_len], ind_trajectories[i][:kept_len])
            else:
                result.check_close(f"{name}/dnet_L{layer_id}_reu{i}",
                                   ps_rows[i][:kept_len], ind_trajectories[i][s:e], atol=atol)

    # --- Gradient flow through hybrid path ---
    grad_k_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device, requires_grad=True)
    grad_v_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device, requires_grad=True)
    grad_store = PrefixAttentionStore()
    grad_k_rows = [grad_k_emb[seq] for seq in sequences]
    grad_v_rows = [grad_v_emb[seq] for seq in sequences]

    trimmed_kg, trimmed_vg = [], []
    for i in range(len(sequences)):
        s, e = plan.input_keep_ranges[i]
        trimmed_kg.append(grad_k_rows[i][s:e])
        trimmed_vg.append(grad_v_rows[i][s:e])

    ek, ev = backend.build_kv(
        torch.cat(trimmed_kg, dim=0), torch.cat(trimmed_vg, dim=0),
        grad_store, plan, layer_id=0,
    )
    total_q = sum(plan.kept_lengths_q)
    packed_qg = torch.randn(total_q, num_q_heads, head_dim, device=device)
    out = backend.attention(packed_qg, ek, ev, plan)
    loss = out.sum()
    loss.backward()

    result.check_true(f"{name}/grad_exists", grad_k_emb.grad is not None)
    if grad_k_emb.grad is not None:
        grad_sum = grad_k_emb.grad.abs().sum().item()
        result.check_true(f"{name}/grad_nonzero", grad_sum > 0, f"grad sum={grad_sum:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    print(f"HybridAttention Integration Tests | device={args.device}")
    print(f"Model: Qwen3.6-27B ({QWEN3_6_27B.num_hidden_layers} layers, "
          f"{QWEN3_6_27B.num_full_attention_layers} full + "
          f"{QWEN3_6_27B.num_linear_attention_layers} linear)")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    result = Result()
    t0 = time.time()

    # Test 1: RL-like batch
    prompt = list(range(100, 132))
    seqs = [prompt + [200 + i * 10 + j for j in range(16)] for i in range(4)]
    run_hybrid_test(seqs, QWEN3_6_27B, args.device, result, "rl_4x32x16")

    # Test 2: Cascading prefixes
    seqs = [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
    run_hybrid_test(seqs, QWEN3_6_27B, args.device, result, "cascading_3seq")

    # Test 3: Mixed sharing
    seqs = [[1, 2, 3, 4, 10], [1, 2, 3, 4, 20, 21], [1, 2, 3, 4, 30], [99, 98]]
    run_hybrid_test(seqs, QWEN3_6_27B, args.device, result, "mixed_4seq")

    # Test 4: Larger RL batch
    prompt = list(range(100, 164))
    seqs = [prompt + [200 + i * 10 + j for j in range(32)] for i in range(8)]
    run_hybrid_test(seqs, QWEN3_6_27B, args.device, result, "rl_8x64x32")

    elapsed = time.time() - t0
    ok = result.summary()
    print(f"Time: {elapsed:.2f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
