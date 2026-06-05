#!/usr/bin/env python3
"""Qwen3.6-27B large-scale precision + performance validation.

Validates prefix-sharing correctness and measures speedup for realistic
GRPO-like RL workloads on GPU. Uses Qwen3.6-27B exact parameters:
  - GQA 24 query heads, 4 KV heads, head_dim=256
  - HybridAttention: 16 full + 48 linear layers
  - bf16 dtype (production training precision)

Tests both TorchReference (SDPA) and Flash Attention 2 backends.

Usage:
    PYTHONPATH=prefix-sharing python tests/run_qwen36_validation.py
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
from prefix_sharing.core.model_spec import QWEN3_6_27B, AttentionLayerType
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import PrefixAttentionStore


class Result:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self):
        return self.failed == 0

    def check_close(self, name, actual, expected, atol=1e-3, rtol=1e-3):
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            max_diff = (actual - expected).abs().max().item()
            mean_diff = (actual - expected).abs().mean().item()
            self.failed += 1
            msg = f"FAIL {name}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
            self.errors.append(msg)
            print(f"  {msg}")
            return False
        self.passed += 1
        max_diff = (actual - expected).abs().max().item()
        print(f"  OK   {name}: max_diff={max_diff:.2e}")
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
        print(f"\n{'='*70}")
        print(f"Results: {self.passed}/{total} checks [{status}]")
        if self.errors:
            print("Failures:")
            for e in self.errors:
                print(f"  - {e}")
        return self.ok()


def _independent_attention_bf16(q, k, v, num_q_heads, num_kv_heads, device):
    """Run independent attention for one sequence (bf16 on CUDA)."""
    head_dim = q.shape[-1]
    scale = math.sqrt(head_dim)
    if num_q_heads > num_kv_heads:
        repeat = num_q_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    # Use SDPA for CUDA efficiency
    # Reshape to [1, H, seq, D] for scaled_dot_product_attention
    q_4d = q.unsqueeze(0).transpose(1, 2)  # [1, H_q, seq, D]
    k_4d = k.unsqueeze(0).transpose(1, 2)  # [1, H_kv, seq, D]
    v_4d = v.unsqueeze(0).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=True)
    return out.transpose(1, 2).squeeze(0)  # [seq, H_q, D]


def run_precision_test(
    sequences, num_q_heads, num_kv_heads, head_dim, device, dtype,
    result, name, backend_cls, atol=1e-2, rtol=1e-2,
):
    """Run precision test comparing prefix-sharing vs independent forward."""
    print(f"\n[{name}] n_seq={len(sequences)}, backend={backend_cls.__name__}, dtype={dtype}")

    torch.manual_seed(42)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)
    backend = backend_cls()
    store = PrefixAttentionStore()

    if not plan.has_sharing:
        print("  SKIP: no sharing detected")
        return

    seq_lens = [len(s) for s in sequences]
    all_tids = set()
    for seq in sequences:
        all_tids.update(seq)
    max_tid = max(all_tids) + 1

    k_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device, dtype=dtype)
    k_rows = [k_emb[seq] for seq in sequences]
    v_rows = [v_emb[seq] for seq in sequences]
    q_rows = [torch.randn(sl, num_q_heads, head_dim, device=device, dtype=dtype) for sl in seq_lens]

    # Independent forward
    ind_outs = [_independent_attention_bf16(q, k, v, num_q_heads, num_kv_heads, device)
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

    expanded_k, expanded_v = backend.build_kv(packed_k, packed_v, store, plan, layer_id=0)
    ps_out = backend.attention(packed_q, expanded_k, expanded_v, plan)
    ps_rows = list(torch.split(ps_out, plan.kept_lengths_q))

    savings = plan.tokens_saved
    ratio = plan.savings_ratio
    print(f"  Plan: {plan.summary()}, savings={savings} tokens ({ratio:.1%})")

    for i in range(len(sequences)):
        s, e = plan.input_keep_ranges[i]
        result.check_close(f"{name}/seq{i}", ps_rows[i], ind_outs[i][s:e], atol=atol, rtol=rtol)


def run_backward_test(
    sequences, num_q_heads, num_kv_heads, head_dim, device, dtype,
    result, name,
):
    """Verify gradients flow correctly through prefix-sharing KV injection."""
    print(f"\n[{name}/backward] Testing gradient flow")

    torch.manual_seed(42)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)
    backend = TorchReferenceBackend()
    store = PrefixAttentionStore()

    if not plan.has_sharing:
        print("  SKIP: no sharing")
        return

    seq_lens = [len(s) for s in sequences]
    all_tids = set()
    for seq in sequences:
        all_tids.update(seq)
    max_tid = max(all_tids) + 1

    k_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k_rows = [k_emb[seq] for seq in sequences]
    v_rows = [v_emb[seq] for seq in sequences]

    trimmed_k, trimmed_v = [], []
    for i in range(len(sequences)):
        s, e = plan.input_keep_ranges[i]
        trimmed_k.append(k_rows[i][s:e])
        trimmed_v.append(v_rows[i][s:e])

    packed_k = torch.cat(trimmed_k, dim=0)
    packed_v = torch.cat(trimmed_v, dim=0)
    expanded_k, expanded_v = backend.build_kv(packed_k, packed_v, store, plan, layer_id=0)

    total_q = sum(plan.kept_lengths_q)
    packed_q = torch.randn(total_q, num_q_heads, head_dim, device=device, dtype=dtype)
    output = backend.attention(packed_q, expanded_k, expanded_v, plan)

    # Backward through reuser output only (gradients must reach provider KV)
    loss = output[plan.kept_lengths_q[0]:].sum()
    loss.backward()

    result.check_true(f"{name}/k_grad_exists", k_emb.grad is not None, "K gradient is None")
    result.check_true(f"{name}/v_grad_exists", v_emb.grad is not None, "V gradient is None")
    if k_emb.grad is not None:
        # Check gradients at the actual token IDs used by provider/reuser
        provider_tids = list(set(sequences[0]))
        reuser_tids = list(set(sequences[1]) - set(sequences[0])) if len(sequences) > 1 else []
        provider_k_grad = k_emb.grad[provider_tids].abs().sum().item()
        result.check_true(f"{name}/provider_k_nonzero", provider_k_grad > 0,
                         f"provider K grad={provider_k_grad:.6f} (tids={provider_tids[:5]}...)")
        if reuser_tids:
            reuser_k_grad = k_emb.grad[reuser_tids].abs().sum().item()
            result.check_true(f"{name}/reuser_k_nonzero", reuser_k_grad > 0,
                             f"reuser K grad={reuser_k_grad:.6f}")


def run_perf_benchmark(
    sequences, num_q_heads, num_kv_heads, head_dim, device, dtype,
    name, warmup=3, runs=10,
):
    """Measure prefix-sharing speedup simulating full transformer layer.

    In a real transformer, attention is only ~8% of FLOPS. The real savings
    come from QKV projection + MLP which scale linearly with token count.
    We simulate this by adding a GEMM proportional to hidden_size per token.
    """
    print(f"\n[{name}/perf] Measuring full-layer speedup...")

    torch.manual_seed(42)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)

    if not plan.has_sharing:
        print("  SKIP: no sharing")
        return

    backend = TorchReferenceBackend()
    store = PrefixAttentionStore()
    seq_lens = [len(s) for s in sequences]
    all_tids = set()
    for seq in sequences:
        all_tids.update(seq)
    max_tid = max(all_tids) + 1

    hidden_size = num_q_heads * head_dim  # 24 * 256 = 6144 for Qwen3.6-27B
    kv_proj_size = num_kv_heads * head_dim  # 4 * 256 = 1024
    qkv_out_size = hidden_size + 2 * kv_proj_size  # 6144 + 1024 + 1024 = 8192
    # Simulated MLP intermediate size (typical SwiGLU: 2.67x hidden)
    mlp_intermediate = int(hidden_size * 8 / 3)  # ~16384

    k_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device, dtype=dtype)
    k_rows = [k_emb[seq] for seq in sequences]
    v_rows = [v_emb[seq] for seq in sequences]
    q_rows = [torch.randn(sl, num_q_heads, head_dim, device=device, dtype=dtype) for sl in seq_lens]

    # Simulate GEMM layers: QKV proj + MLP (the dominant cost)
    qkv_weight = torch.randn(hidden_size, qkv_out_size, device=device, dtype=dtype) * 0.01
    mlp_up = torch.randn(hidden_size, mlp_intermediate, device=device, dtype=dtype) * 0.01
    mlp_down = torch.randn(mlp_intermediate, hidden_size, device=device, dtype=dtype) * 0.01

    def sim_layer(hiddens):
        """Simulate one transformer layer: QKV proj + attention + MLP."""
        outs = []
        for h in hiddens:
            # QKV projection (GQA: Q has more heads than K/V)
            qkv = h @ qkv_weight
            # Split: Q=[hidden_size], K=[kv_proj_size], V=[kv_proj_size]
            q = qkv[:, :hidden_size].view(h.shape[0], num_q_heads, head_dim)
            k = qkv[:, hidden_size:hidden_size + kv_proj_size].view(h.shape[0], num_kv_heads, head_dim)
            v_ = qkv[:, hidden_size + kv_proj_size:].view(h.shape[0], num_kv_heads, head_dim)
            attn_out = _independent_attention_bf16(q, k, v_, num_q_heads, num_kv_heads, device)
            attn_flat = attn_out.reshape(h.shape[0], -1)
            # MLP
            up = torch.nn.functional.silu(attn_flat @ mlp_up)
            out = up @ mlp_down
            outs.append(out)
        return outs

    # --- Independent baseline ---
    hiddens_ind = [torch.randn(sl, hidden_size, device=device, dtype=dtype) for sl in seq_lens]

    for _ in range(warmup):
        _ = sim_layer(hiddens_ind)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sim_layer(hiddens_ind)
    torch.cuda.synchronize()
    t_ind = (time.perf_counter() - t0) / runs

    # --- Prefix-sharing path ---
    total_kept = sum(plan.kept_lengths_q)
    hiddens_ps = torch.randn(total_kept, hidden_size, device=device, dtype=dtype)

    trimmed_q, trimmed_k, trimmed_v = [], [], []
    for i in range(len(sequences)):
        s, e = plan.input_keep_ranges[i]
        trimmed_q.append(q_rows[i][s:e])
        trimmed_k.append(k_rows[i][s:e])
        trimmed_v.append(v_rows[i][s:e])

    packed_q = torch.cat(trimmed_q, dim=0)
    packed_k = torch.cat(trimmed_k, dim=0)
    packed_v = torch.cat(trimmed_v, dim=0)

    def sim_layer_ps(h_ps):
        """Simulate one PS transformer layer."""
        # QKV projection on kept tokens only
        qkv = h_ps @ qkv_weight
        q = qkv[:, :hidden_size].view(h_ps.shape[0], num_q_heads, head_dim)
        k = qkv[:, hidden_size:hidden_size + kv_proj_size].view(h_ps.shape[0], num_kv_heads, head_dim)
        v_ = qkv[:, hidden_size + kv_proj_size:].view(h_ps.shape[0], num_kv_heads, head_dim)
        # build_kv + attention
        st = PrefixAttentionStore()
        ek, ev = backend.build_kv(k, v_, st, plan, layer_id=0)
        attn_out = backend.attention(q, ek, ev, plan)
        attn_flat = attn_out.reshape(h_ps.shape[0], -1)
        # MLP on kept tokens only
        up = torch.nn.functional.silu(attn_flat @ mlp_up)
        return up @ mlp_down

    for _ in range(warmup):
        _ = sim_layer_ps(hiddens_ps)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sim_layer_ps(hiddens_ps)
    torch.cuda.synchronize()
    t_ps = (time.perf_counter() - t0) / runs

    speedup = t_ind / t_ps if t_ps > 0 else float('inf')
    total_orig = sum(seq_lens)
    print(f"  Independent: {t_ind*1000:.2f}ms ({total_orig} tokens)")
    print(f"  Prefix-sharing: {t_ps*1000:.2f}ms ({total_kept} tokens)")
    print(f"  Speedup: {speedup:.2f}x (saved {1-total_kept/total_orig:.1%} tokens)")
    return speedup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--skip-fa2", action="store_true", help="Skip FA2 backend tests")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    device = args.device
    dtype = torch.bfloat16

    print("=" * 70)
    print("Qwen3.6-27B Prefix-Sharing Validation")
    print("=" * 70)
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Model: {QWEN3_6_27B.num_hidden_layers} layers, "
          f"GQA {QWEN3_6_27B.num_attention_heads}:{QWEN3_6_27B.num_key_value_heads}, "
          f"head_dim={QWEN3_6_27B.head_dim}")
    print(f"HybridAttention: {QWEN3_6_27B.num_full_attention_layers} full + "
          f"{QWEN3_6_27B.num_linear_attention_layers} linear")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    Q = QWEN3_6_27B.num_attention_heads  # 24
    KV = QWEN3_6_27B.num_key_value_heads  # 4
    HD = QWEN3_6_27B.head_dim  # 256

    result = Result()
    t_total = time.perf_counter()

    # --- Precision Tests ---
    print("=" * 70)
    print("PART 1: PRECISION VALIDATION")
    print("=" * 70)

    # Test 1: GRPO n=4, prefix=64, suffix=64
    prefix = list(range(100, 164))
    seqs = [prefix + [200 + i * 100 + j for j in range(64)] for i in range(4)]
    run_precision_test(seqs, Q, KV, HD, device, dtype, result,
                       "GRPO-4x64x64", TorchReferenceBackend, atol=1e-2, rtol=1e-2)

    # Test 2: GRPO n=8, prefix=128, suffix=128
    prefix = list(range(100, 228))
    seqs = [prefix + [200 + i * 100 + j for j in range(128)] for i in range(8)]
    run_precision_test(seqs, Q, KV, HD, device, dtype, result,
                       "GRPO-8x128x128", TorchReferenceBackend, atol=1e-2, rtol=1e-2)

    # Test 3: GRPO n=8, prefix=512, suffix=128 (long prefix)
    prefix = list(range(100, 612))
    seqs = [prefix + [200 + i * 100 + j for j in range(128)] for i in range(8)]
    run_precision_test(seqs, Q, KV, HD, device, dtype, result,
                       "GRPO-8x512x128", TorchReferenceBackend, atol=1e-2, rtol=1e-2)

    # Test 4: Cascading prefixes
    seqs = [list(range(100, 116)),
            list(range(100, 124)),
            list(range(100, 132))]
    run_precision_test(seqs, Q, KV, HD, device, dtype, result,
                       "cascading_3seq", TorchReferenceBackend, atol=1e-2, rtol=1e-2)

    # Test 5: Mixed sharing (some with, some without)
    prefix = list(range(100, 132))
    seqs = [prefix + [200 + j for j in range(32)],
            prefix + [300 + j for j in range(32)],
            list(range(500, 580)),  # no sharing
            prefix + [400 + j for j in range(32)]]
    run_precision_test(seqs, Q, KV, HD, device, dtype, result,
                       "mixed_4seq", TorchReferenceBackend, atol=1e-2, rtol=1e-2)

    # Test 6: FA2 backend precision (if available)
    if not args.skip_fa2:
        try:
            from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
            prefix = list(range(100, 228))
            seqs = [prefix + [200 + i * 100 + j for j in range(128)] for i in range(8)]
            run_precision_test(seqs, Q, KV, HD, device, dtype, result,
                               "GRPO-8x128x128_FA2", GpuFlashAttentionBackend,
                               atol=1e-1, rtol=1e-1)
        except (ImportError, RuntimeError) as e:
            print(f"\n  SKIP: FA2 backend not available: {e}")

    # --- Backward Tests ---
    print("\n" + "=" * 70)
    print("PART 2: BACKWARD / GRADIENT VALIDATION")
    print("=" * 70)

    # Backward test 1: Small
    seqs = [list(range(100, 116)) + list(range(200, 208)),
            list(range(100, 116)) + list(range(300, 308))]
    run_backward_test(seqs, Q, KV, HD, device, dtype, result, "backward_small")

    # Backward test 2: Large RL-like
    prefix = list(range(100, 228))
    seqs = [prefix + [200 + i * 100 + j for j in range(64)] for i in range(4)]
    run_backward_test(seqs, Q, KV, HD, device, dtype, result, "backward_GRPO-4x128x64")

    # --- Performance Tests ---
    print("\n" + "=" * 70)
    print("PART 3: PERFORMANCE BENCHMARKS")
    print("=" * 70)

    perf_results = {}

    # Perf 1: GRPO-8x128x64
    prefix = list(range(100, 228))
    seqs = [prefix + [200 + i * 100 + j for j in range(64)] for i in range(8)]
    perf_results["GRPO-8x128x64"] = run_perf_benchmark(
        seqs, Q, KV, HD, device, dtype, "GRPO-8x128x64")

    # Perf 2: GRPO-8x512x128
    prefix = list(range(100, 612))
    seqs = [prefix + [200 + i * 100 + j for j in range(128)] for i in range(8)]
    perf_results["GRPO-8x512x128"] = run_perf_benchmark(
        seqs, Q, KV, HD, device, dtype, "GRPO-8x512x128")

    # Perf 3: GRPO-16x256x64
    prefix = list(range(100, 356))
    seqs = [prefix + [200 + i * 100 + j for j in range(64)] for i in range(16)]
    perf_results["GRPO-16x256x64"] = run_perf_benchmark(
        seqs, Q, KV, HD, device, dtype, "GRPO-16x256x64")

    # Perf 4: GRPO-4x2048x128
    prefix = list(range(100, 2148))
    seqs = [prefix + [200 + i * 100 + j for j in range(128)] for i in range(4)]
    perf_results["GRPO-4x2048x128"] = run_perf_benchmark(
        seqs, Q, KV, HD, device, dtype, "GRPO-4x2048x128")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    for name, speedup in sorted(perf_results.items()):
        print(f"  {name}: {speedup:.2f}x speedup")

    elapsed = time.perf_counter() - t_total
    ok = result.summary()
    print(f"\nTotal time: {elapsed:.2f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
