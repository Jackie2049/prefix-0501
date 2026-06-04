#!/usr/bin/env python3
"""Standalone precision test runner (no pytest dependency).

This script validates that prefix-sharing produces identical numerical results
compared to independent forward passes. It runs on CPU or CUDA.

Usage:
    PYTHONPATH=prefix-sharing python tests/run_precision.py
    PYTHONPATH=prefix-sharing python tests/run_precision.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import sys
import time

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required. Install with: pip install torch")
    sys.exit(1)

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import PrefixAttentionStore


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self):
        return self.failed == 0

    def check_close(self, name, actual, expected, atol=1e-6, rtol=1e-5):
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            max_diff = (actual - expected).abs().max().item()
            self.failed += 1
            msg = f"FAIL {name}: max_diff={max_diff:.2e} (atol={atol})"
            self.errors.append(msg)
            print(f"  {msg}")
            return False
        self.passed += 1
        return True

    def check_true(self, name, condition, msg=""):
        if not condition:
            self.failed += 1
            full_msg = f"FAIL {name}: {msg}" if msg else f"FAIL {name}"
            self.errors.append(full_msg)
            print(f"  {full_msg}")
            return False
        self.passed += 1
        return True

    def summary(self):
        total = self.passed + self.failed
        status = "PASS" if self.ok() else "FAIL"
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} checks passed [{status}]")
        if self.errors:
            print("Failures:")
            for e in self.errors:
                print(f"  - {e}")
        return self.ok()


def independent_attention(query_rows, key_rows, value_rows, device="cpu"):
    """Run independent attention for each sequence."""
    outputs = []
    for q, k, v in zip(query_rows, key_rows, value_rows):
        if q.dim() == 2:
            scale = math.sqrt(q.shape[-1])
            scores = q @ k.transpose(-1, -2) / scale
            mask = torch.arange(k.shape[0], device=device).unsqueeze(0) <= torch.arange(q.shape[0], device=device).unsqueeze(1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            outputs.append(probs @ v)
        else:
            q_heads, kv_heads = q.shape[1], k.shape[1]
            scale = math.sqrt(q.shape[-1])
            if q_heads != kv_heads:
                repeat = q_heads // kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q, k) / scale
            mask = torch.arange(k.shape[0], device=device).unsqueeze(0) <= torch.arange(q.shape[0], device=device).unsqueeze(1)
            scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            outputs.append(torch.einsum("hqk,khd->qhd", probs, v))
    return outputs


def run_ps_vs_independent(sequences, head_dim, num_q_heads, num_kv_heads, device, seed=42):
    """Run both prefix-sharing and independent attention, return results.

    Key: K/V are determined by token ID + position, so identical tokens at
    the same position produce identical K/V. This mirrors real model behavior
    where K = W_k * embedding(token_id).
    """
    torch.manual_seed(seed)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)
    backend = TorchReferenceBackend()
    store = PrefixAttentionStore()

    seq_lens = [len(s) for s in sequences]

    # Generate K/V lookup by token ID (same token → same K/V everywhere)
    all_token_ids = set()
    for seq in sequences:
        all_token_ids.update(seq)
    max_token_id = max(all_token_ids) + 1 if all_token_ids else 0

    # K/V embeddings: [max_token_id, num_heads, head_dim]
    k_embedding = torch.randn(max_token_id, num_kv_heads, head_dim, device=device)
    v_embedding = torch.randn(max_token_id, num_kv_heads, head_dim, device=device)
    # Q is per-position, not shared (query depends on hidden state, not just token ID)
    q_per_position = []
    offset = 0
    for seq_len in seq_lens:
        q_per_position.append(torch.randn(seq_len, num_q_heads, head_dim, device=device))
        offset += seq_len

    # Build per-sequence K/V from token embeddings
    k_rows = [k_embedding[seq] for seq in sequences]
    v_rows = [v_embedding[seq] for seq in sequences]
    q_rows = q_per_position

    # Independent
    ind_outs = independent_attention(q_rows, k_rows, v_rows, device=device)

    # Prefix-sharing: trim Q/K/V to kept ranges
    trimmed_q, trimmed_k, trimmed_v = [], [], []
    for i, (q, k, v) in enumerate(zip(q_rows, k_rows, v_rows)):
        s, e = plan.input_keep_ranges[i]
        trimmed_q.append(q[s:e])
        trimmed_k.append(k[s:e])
        trimmed_v.append(v[s:e])

    packed_q = torch.cat(trimmed_q, dim=0)
    packed_k = torch.cat(trimmed_k, dim=0)
    packed_v = torch.cat(trimmed_v, dim=0)

    expanded_k, expanded_v = backend.build_kv(packed_k, packed_v, store, plan, layer_id=0)
    ps_output = backend.attention(packed_q, expanded_k, expanded_v, plan)
    ps_rows = list(torch.split(ps_output, plan.kept_lengths_q))

    return ps_rows, ind_outs, plan


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_2seq(result, device):
    """Test 2 sequences with shared prefix, MHA."""
    print("\n[test_basic_2seq] 2 sequences, shared prefix, MHA")
    sequences = [[1,2,3,10,20], [1,2,3,30,40,50]]
    ps_outs, ind_outs, plan = run_ps_vs_independent(sequences, 8, 4, 4, device)
    result.check_true("has_sharing", plan.has_sharing)

    # Provider
    result.check_close("provider_out", ps_outs[0], ind_outs[0])
    # Reuser suffix
    s, e = plan.input_keep_ranges[1]
    result.check_close("reuser_suffix", ps_outs[1], ind_outs[1][s:e])


def test_basic_gqa(result, device):
    """Test GQA (8 query heads, 2 kv heads)."""
    print("\n[test_basic_gqa] 2 sequences, GQA 8:2")
    sequences = [[1,2,3,10,20], [1,2,3,30,40]]
    ps_outs, ind_outs, plan = run_ps_vs_independent(sequences, 8, 8, 2, device)
    result.check_true("has_sharing", plan.has_sharing)
    result.check_close("provider_out", ps_outs[0], ind_outs[0])
    s, e = plan.input_keep_ranges[1]
    result.check_close("reuser_suffix", ps_outs[1], ind_outs[1][s:e])


def test_cascading_3seq(result, device):
    """Test 3 sequences with cascading prefix reuse."""
    print("\n[test_cascading_3seq] 3 sequences, cascading")
    sequences = [[1,2,3], [1,2,3,4], [1,2,3,4,5]]
    ps_outs, ind_outs, plan = run_ps_vs_independent(sequences, 8, 4, 4, device)
    result.check_true("has_sharing", plan.has_sharing)
    result.check_close("seq0_provider", ps_outs[0], ind_outs[0])
    for i in range(1, 3):
        s, e = plan.input_keep_ranges[i]
        result.check_close(f"seq{i}_suffix", ps_outs[i], ind_outs[i][s:e])


def test_4seq_mixed(result, device):
    """Test 4 sequences with mixed prefix + 1 unshared."""
    print("\n[test_4seq_mixed] 4 sequences, 1 unshared")
    sequences = [[1,2,3,4,10,11], [1,2,3,4,20,21,22], [1,2,3,4,30], [99,98]]
    ps_outs, ind_outs, plan = run_ps_vs_independent(sequences, 8, 4, 4, device)
    result.check_true("has_sharing", plan.has_sharing)
    for i in range(4):
        s, e = plan.input_keep_ranges[i]
        result.check_close(f"seq{i}", ps_outs[i], ind_outs[i][s:e])


def test_rl_batch_8(result, device):
    """Simulate RL batch: 1 prompt x 8 responses."""
    print("\n[test_rl_batch_8] RL-like batch: 1 prompt x 8 responses")
    prompt = list(range(100, 164))  # 64 tokens
    sequences = [prompt + [200+j*10+i for i in range(32)] for j in range(8)]
    ps_outs, ind_outs, plan = run_ps_vs_independent(sequences, 16, 8, 2, device)
    result.check_true("has_sharing", plan.has_sharing)
    for i in range(8):
        s, e = plan.input_keep_ranges[i]
        result.check_close(f"seq{i}", ps_outs[i], ind_outs[i][s:e], atol=1e-5)


def test_prefix_lengths(result, device):
    """Test varying prefix lengths."""
    print("\n[test_prefix_lengths] Varying prefix lengths")
    for prefix_len in [2, 8, 32, 64, 128]:
        prefix = list(range(100, 100 + prefix_len))
        sequences = [
            prefix + [200+i for i in range(16)],
            prefix + [300+i for i in range(16)],
        ]
        ps_outs, ind_outs, plan = run_ps_vs_independent(sequences, 8, 4, 4, device)
        result.check_true(f"has_sharing_P{prefix_len}", plan.has_sharing)
        for i in range(2):
            s, e = plan.input_keep_ranges[i]
            result.check_close(f"P{prefix_len}_seq{i}", ps_outs[i], ind_outs[i][s:e], atol=1e-5)


def test_gradient_preservation(result, device):
    """Test that gradients flow through prefix-sharing KV injection."""
    print("\n[test_gradient_preservation] Gradient flow through provider KV")
    sequences = [[1,2,3,10], [1,2,3,20,21]]
    torch.manual_seed(42)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)
    backend = TorchReferenceBackend()
    store = PrefixAttentionStore()

    seq_lens = [len(s) for s in sequences]
    all_k = torch.randn(sum(seq_lens), 4, device=device, requires_grad=True)
    all_v = torch.randn(sum(seq_lens), 4, device=device, requires_grad=True)

    k_rows = list(torch.split(all_k, seq_lens))
    v_rows = list(torch.split(all_v, seq_lens))

    trimmed_k, trimmed_v = [], []
    for i, (k, v) in enumerate(zip(k_rows, v_rows)):
        s, e = plan.input_keep_ranges[i]
        trimmed_k.append(k[s:e])
        trimmed_v.append(v[s:e])

    packed_k = torch.cat(trimmed_k, dim=0)
    packed_v = torch.cat(trimmed_v, dim=0)
    expanded_k, expanded_v = backend.build_kv(packed_k, packed_v, store, plan, layer_id=0)

    total_q = sum(plan.kept_lengths_q)
    packed_q = torch.randn(total_q, 4, device=device)
    output = backend.attention(packed_q, expanded_k, expanded_v, plan)

    # Backward through reuser output only
    loss = output[plan.kept_lengths_q[0]:].sum()
    loss.backward()

    result.check_true("k_grad_exists", all_k.grad is not None, "K gradient is None")
    result.check_true("v_grad_exists", all_v.grad is not None, "V gradient is None")
    if all_k.grad is not None:
        provider_k_grad_sum = all_k.grad[:seq_lens[0]].abs().sum().item()
        result.check_true("provider_k_grad_nonzero", provider_k_grad_sum > 0,
                         f"Provider K grad sum={provider_k_grad_sum:.6f} (KV may be detached)")


def test_single_token_suffix(result, device):
    """Test reuser with only 1 suffix token."""
    print("\n[test_single_token_suffix] 1 suffix token after trimming")
    sequences = [[1,2,3,4,5], [1,2,3,4,99]]
    ps_outs, ind_outs, plan = run_ps_vs_independent(sequences, 8, 4, 4, device)
    result.check_true("has_sharing", plan.has_sharing)
    result.check_true("suffix_len_1", plan.kept_lengths_q[1] == 1)
    s, e = plan.input_keep_ranges[1]
    result.check_close("1tok_suffix", ps_outs[1], ind_outs[1][s:e])


def test_identical_sequences(result, device):
    """Test completely identical sequences."""
    print("\n[test_identical_sequences] Identical sequences")
    seq = [1,2,3,4,5]
    sequences = [seq, seq[:]]
    ps_outs, ind_outs, plan = run_ps_vs_independent(sequences, 8, 4, 4, device)
    result.check_true("has_sharing", plan.has_sharing)
    result.check_close("provider", ps_outs[0], ind_outs[0])
    s, e = plan.input_keep_ranges[1]
    if e > s:
        result.check_close("reuser", ps_outs[1], ind_outs[1][s:e])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prefix-sharing precision tests")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    print(f"Prefix-Sharing Precision Tests | device={args.device}")
    print(f"PyTorch: {torch.__version__}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    result = TestResult()
    t0 = time.time()

    test_basic_2seq(result, args.device)
    test_basic_gqa(result, args.device)
    test_cascading_3seq(result, args.device)
    test_4seq_mixed(result, args.device)
    test_rl_batch_8(result, args.device)
    test_prefix_lengths(result, args.device)
    test_gradient_preservation(result, args.device)
    test_single_token_suffix(result, args.device)
    test_identical_sequences(result, args.device)

    elapsed = time.time() - t0
    ok = result.summary()
    print(f"Time: {elapsed:.2f}s")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
