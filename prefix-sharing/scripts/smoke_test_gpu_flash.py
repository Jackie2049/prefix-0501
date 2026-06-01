"""Standalone smoke test for GPU Flash Attention backend.

Run on an A100 (or any CUDA GPU) with:
    python scripts/smoke_test_gpu_flash.py

Requirements:
    torch + CUDA
    flash-attn
    prefix-sharing package on PYTHONPATH
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root so imports work when run standalone
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch

from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner


def _make_plan(batch_sizes: list[int], prefix_lens: list[int]):
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_gpu")
    planner = PrefixSharingPlanner(config)
    input_ids = [list(range(s)) for s in batch_sizes]
    plan = planner.plan(input_ids)

    object.__setattr__(plan, "batch_size", len(batch_sizes))
    object.__setattr__(plan, "original_lengths", batch_sizes)
    object.__setattr__(plan, "prefix_lens", prefix_lens)
    object.__setattr__(plan, "kept_lengths_q", [b - p for b, p in zip(batch_sizes, prefix_lens)])
    object.__setattr__(plan, "expanded_lengths_kv", list(batch_sizes))
    object.__setattr__(plan, "q_position_offsets", prefix_lens)
    object.__setattr__(plan, "kv_position_offsets", [0] * len(batch_sizes))

    cu_seqlens_q = [0]
    cu_seqlens_kv = [0]
    max_seqlen_q = 0
    max_seqlen_kv = 0
    for b, p in zip(batch_sizes, prefix_lens):
        q_len = b - p
        kv_len = b
        cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
        cu_seqlens_kv.append(cu_seqlens_kv[-1] + kv_len)
        max_seqlen_q = max(max_seqlen_q, q_len)
        max_seqlen_kv = max(max_seqlen_kv, kv_len)

    object.__setattr__(plan, "cu_seqlens_q", cu_seqlens_q)
    object.__setattr__(plan, "cu_seqlens_kv", cu_seqlens_kv)
    object.__setattr__(plan, "max_seqlen_q", max_seqlen_q)
    object.__setattr__(plan, "max_seqlen_kv", max_seqlen_kv)
    object.__setattr__(plan, "provider_index", [0] * len(batch_sizes))
    object.__setattr__(plan, "is_provider", [p == 0 for p in prefix_lens])
    object.__setattr__(plan, "reuse_specs", ())
    object.__setattr__(plan, "prefix_last_restore", [])
    return plan


def _random_qkv(total_q, total_kv, num_heads, num_kv_heads, head_dim, dtype=torch.float32):
    torch.manual_seed(42)
    device = torch.device("cuda")
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device) * 0.02
    k = torch.randn(total_kv, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02
    v = torch.randn(total_kv, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.02
    return q, k, v


def main():
    if not torch.cuda.is_available():
        print("CUDA not available — skipping GPU flash attention smoke test.")
        sys.exit(0)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    backend = GpuFlashAttentionBackend()
    ref_backend = TorchReferenceBackend()

    # Test 1: same q/kv lengths (no sharing)
    print("\n[Test 1] same q/kv lengths ...")
    plan = _make_plan(batch_sizes=[4, 6], prefix_lens=[0, 0])
    q, k, v = _random_qkv(10, 10, 2, 2, 64)
    out = backend.attention(q, k, v, plan)
    assert out.shape == q.shape
    print("  OK")

    # Test 2: different q/kv lengths (provider + reuser)
    print("[Test 2] different q/kv lengths vs torch_ref ...")
    plan = _make_plan(batch_sizes=[8, 8], prefix_lens=[0, 4])
    q, k, v = _random_qkv(12, 16, 2, 2, 64, dtype=torch.float32)
    out_fa = backend.attention(q, k, v, plan)
    out_ref = ref_backend.attention(q, k, v, plan)
    max_diff = (out_fa - out_ref).abs().max().item()
    print(f"  max_diff={max_diff:.6e}")
    assert max_diff < 5e-3, f"max_diff too large: {max_diff}"
    print("  OK")

    # Test 3: GQA
    print("[Test 3] GQA (num_heads=4, num_kv_heads=2) ...")
    plan = _make_plan(batch_sizes=[4, 6], prefix_lens=[0, 2])
    q, k, v = _random_qkv(8, 10, 4, 2, 64)
    out = backend.attention(q, k, v, plan)
    assert out.shape == q.shape
    print("  OK")

    # Test 4: backward
    print("[Test 4] backward gradient flow ...")
    plan = _make_plan(batch_sizes=[4, 6], prefix_lens=[0, 2])
    q, k, v = _random_qkv(8, 10, 2, 2, 64, dtype=torch.float32)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    out = backend.attention(q, k, v, plan)
    out.sum().backward()
    assert q.grad is not None and not torch.isnan(q.grad).any()
    assert k.grad is not None and not torch.isnan(k.grad).any()
    assert v.grad is not None and not torch.isnan(v.grad).any()
    print("  OK")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
