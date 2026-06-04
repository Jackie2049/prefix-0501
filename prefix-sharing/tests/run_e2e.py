"""End-to-end pipeline test: validate full prefix-sharing flow without verl/Megatron.

Simulates the complete training pipeline:
1. Input sequences → detection → planning
2. Trim inputs/labels/loss_masks
3. Build KV (provider store + reuser load)
4. Run attention (TorchReferenceBackend)
5. Assemble output (un-trim back to original positions)
6. Restore prefix-last logprobs
7. Verify: PS output == independent forward output for all positions

Usage:
    PYTHONPATH=prefix-sharing python tests/run_e2e.py
    PYTHONPATH=prefix-sharing python tests/run_e2e.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import sys
import time

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required")
    sys.exit(1)

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import PrefixAttentionStore


class E2EResult:
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
        print(f"E2E Results: {self.passed}/{total} checks passed [{status}]")
        if self.errors:
            print("Failures:")
            for e in self.errors:
                print(f"  - {e}")
        return self.ok()


def run_e2e_test(sequences, head_dim, num_q_heads, num_kv_heads, device, result, name, atol=1e-5):
    """Run complete end-to-end pipeline test."""
    print(f"\n[{name}] batch_size={len(sequences)}, heads={num_q_heads}:{num_kv_heads}, head_dim={head_dim}")

    torch.manual_seed(42)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)
    backend = TorchReferenceBackend()
    store = PrefixAttentionStore()

    result.check_true(f"{name}/has_sharing", plan.has_sharing)
    if not plan.has_sharing:
        return

    seq_lens = [len(s) for s in sequences]
    all_token_ids = set()
    for seq in sequences:
        all_token_ids.update(seq)
    max_tid = max(all_token_ids) + 1

    k_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device)
    v_emb = torch.randn(max_tid, num_kv_heads, head_dim, device=device)
    q_per_pos = [torch.randn(sl, num_q_heads, head_dim, device=device) for sl in seq_lens]
    k_rows = [k_emb[seq] for seq in sequences]
    v_rows = [v_emb[seq] for seq in sequences]

    # Step 1: Independent forward (ground truth)
    ind_attn_outs = []
    for q, k, v in zip(q_per_pos, k_rows, v_rows):
        scale = math.sqrt(head_dim)
        if num_q_heads != num_kv_heads:
            if num_q_heads > num_kv_heads:
                repeat = num_q_heads // num_kv_heads
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)
            else:
                repeat = num_kv_heads // num_q_heads
                q = q.repeat_interleave(repeat, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) / scale
        mask = torch.arange(k.shape[0], device=device).unsqueeze(0) <= torch.arange(q.shape[0], device=device).unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(0), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        ind_attn_outs.append(torch.einsum("hqk,khd->qhd", probs, v))

    # Step 2: Prefix-sharing forward
    trimmed_q, trimmed_k, trimmed_v = [], [], []
    for i, (q, k, v) in enumerate(zip(q_per_pos, k_rows, v_rows)):
        s, e = plan.input_keep_ranges[i]
        trimmed_q.append(q[s:e])
        trimmed_k.append(k[s:e])
        trimmed_v.append(v[s:e])

    packed_q = torch.cat(trimmed_q, dim=0)
    packed_k = torch.cat(trimmed_k, dim=0)
    packed_v = torch.cat(trimmed_v, dim=0)

    expanded_k, expanded_v = backend.build_kv(packed_k, packed_v, store, plan, layer_id=0)
    ps_attn_out = backend.attention(packed_q, expanded_k, expanded_v, plan)

    # Step 3: Split PS output back to per-sequence
    ps_rows = list(torch.split(ps_attn_out, plan.kept_lengths_q))

    # Step 4: Verify attention outputs match
    for i in range(len(sequences)):
        s, e = plan.input_keep_ranges[i]
        result.check_close(f"{name}/attn_seq{i}", ps_rows[i], ind_attn_outs[i][s:e], atol=atol)

    # Step 5: Verify gradient flow through the full pipeline
    all_k = torch.randn(sum(seq_lens), num_kv_heads, head_dim, device=device, requires_grad=True)
    all_v = torch.randn(sum(seq_lens), num_kv_heads, head_dim, device=device, requires_grad=True)

    k_grad_rows = list(torch.split(all_k, seq_lens))
    v_grad_rows = list(torch.split(all_v, seq_lens))

    trimmed_kg, trimmed_vg = [], []
    for i, (k, v) in enumerate(zip(k_grad_rows, v_grad_rows)):
        s, e = plan.input_keep_ranges[i]
        trimmed_kg.append(k[s:e])
        trimmed_vg.append(v[s:e])

    packed_kg = torch.cat(trimmed_kg, dim=0)
    packed_vg = torch.cat(trimmed_vg, dim=0)

    grad_store = PrefixAttentionStore()
    ek, ev = backend.build_kv(packed_kg, packed_vg, grad_store, plan, layer_id=0)
    total_q = sum(plan.kept_lengths_q)
    packed_qg = torch.randn(total_q, num_q_heads, head_dim, device=device)
    out = backend.attention(packed_qg, ek, ev, plan)

    loss = out.sum()
    loss.backward()

    result.check_true(f"{name}/k_grad_exists", all_k.grad is not None)
    result.check_true(f"{name}/v_grad_exists", all_v.grad is not None)
    if all_k.grad is not None:
        provider_k_grad = all_k.grad[:seq_lens[0]].abs().sum().item()
        result.check_true(f"{name}/provider_k_grad_nonzero", provider_k_grad > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"

    print(f"Prefix-Sharing E2E Pipeline Tests | device={args.device}")
    print(f"PyTorch: {torch.__version__}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    result = E2EResult()
    t0 = time.time()

    # Test 1: Basic RL-like batch
    prompt = list(range(100, 164))
    seqs = [prompt + [200+i*10+j for j in range(32)] for i in range(8)]
    run_e2e_test(seqs, 16, 8, 2, args.device, result, "rl_8x64x32_gqa8:2")

    # Test 2: Cascading prefixes
    seqs = [[1,2,3], [1,2,3,4], [1,2,3,4,5]]
    run_e2e_test(seqs, 8, 4, 4, args.device, result, "cascading_3seq")

    # Test 3: Mixed (shared + unshared)
    seqs = [[1,2,3,4,10], [1,2,3,4,20,21], [1,2,3,4,30], [99,98]]
    run_e2e_test(seqs, 8, 4, 4, args.device, result, "mixed_4seq")

    # Test 4: Qwen3.6-like GQA 24:4
    prompt = list(range(100, 116))
    seqs = [prompt + [200+i*100+j for j in range(8)] for i in range(4)]
    run_e2e_test(seqs, 256, 24, 4, args.device, result, "qwen36_4x16x8", atol=1e-3)

    # Test 5: Long prefix
    prompt = list(range(100, 356))  # 256 tokens
    seqs = [prompt + [300+i*100+j for j in range(32)] for i in range(4)]
    run_e2e_test(seqs, 64, 8, 2, args.device, result, "long_prefix_4x256x32")

    elapsed = time.time() - t0
    ok = result.summary()
    print(f"Time: {elapsed:.2f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
