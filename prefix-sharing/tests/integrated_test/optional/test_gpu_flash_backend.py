"""Optional integration tests for the GPU Flash Attention 2 backend.

These tests require:
* PyTorch with CUDA
* ``flash-attn`` package
* A CUDA-capable GPU (they will skip on CPU-only hosts).
"""

from __future__ import annotations

from typing import Any

import pytest

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner

pytest.importorskip("torch")
pytest.importorskip("flash_attn")

import torch  # noqa: E402

from prefix_sharing.backends.gpu_flash_attn import GpuFlashAttentionBackend  # noqa: E402
from prefix_sharing.backends.torch_ref import TorchReferenceBackend  # noqa: E402 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def backend() -> GpuFlashAttentionBackend:
    return GpuFlashAttentionBackend()


@pytest.fixture
def ref_backend() -> TorchReferenceBackend:
    return TorchReferenceBackend()


def _make_plan(batch_sizes: list[int], prefix_lens: list[int]) -> Any:
    """Build a minimal PrefixSharingPlan with the given layout.

    * batch_sizes[i] = original sequence length of sample i
    * prefix_lens[i] = number of prefix tokens shared by sample i
      (0 means the sample is a provider / no sharing).
    """
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="gpu_flash_attn")
    planner = PrefixSharingPlanner(config)

    # We need real input_ids to feed the planner, but for these tests we only
    # care about the structural metadata (lengths / cu_seqlens).  Use simple
    # monotonic tokens so that the detector may or may not find sharing; we
    # override the plan manually afterwards.
    input_ids = [list(range(s)) for s in batch_sizes]
    plan = planner.plan(input_ids)

    # Force the structural fields to match the caller's intent.
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


def _random_qkv(
    total_q: int,
    total_kv: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=DEVICE) * 0.02
    k = torch.randn(total_kv, num_kv_heads, head_dim, dtype=dtype, device=DEVICE) * 0.02
    v = torch.randn(total_kv, num_kv_heads, head_dim, dtype=dtype, device=DEVICE) * 0.02
    return q, k, v


# ------------------------------------------------------------------
# Shared assertion helpers
# ------------------------------------------------------------------
_ATOL_FP16 = 5e-2
_ATOL_GRAD_FP16 = 2e-1


def _assert_outputs_close(
    out_fa: torch.Tensor,
    out_ref: torch.Tensor,
    atol: float = _ATOL_FP16,
) -> None:
    assert out_fa.shape == out_ref.shape
    max_diff = (out_fa - out_ref).abs().max().item()
    assert max_diff < atol, f"output max_diff={max_diff} >= {atol}"


def _assert_grads_close(
    grads_fa: dict[str, torch.Tensor | None],
    grads_ref: dict[str, torch.Tensor | None],
    atol: float = _ATOL_GRAD_FP16,
) -> None:
    for name in ("q", "k", "v"):
        g_fa = grads_fa[name]
        g_ref = grads_ref[name]
        assert g_fa is not None, f"FA {name}.grad is None"
        assert g_ref is not None, f"ref {name}.grad is None"
        max_diff = (g_fa - g_ref).abs().max().item()
        assert max_diff < atol, f"{name}.grad max_diff={max_diff} >= {atol}"


def _run_forward_backward(
    backend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    """Run attention + sum().backward() and return output + grads."""
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    out = backend.attention(q, k, v, plan)
    out.sum().backward()
    return out, {"q": q.grad, "k": k.grad, "v": v.grad}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_same_q_kv_lengths(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """Simplest case: all samples are providers (no prefix sharing)."""
    plan = _make_plan(batch_sizes=[4, 6], prefix_lens=[0, 0])
    q, k, v = _random_qkv(
        total_q=10, total_kv=10, num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16
    )
    out_fa = backend.attention(q, k, v, plan)
    out_ref = ref_backend.attention(q, k, v, plan)
    assert out_fa.shape == q.shape
    assert out_fa.device == q.device
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_vs_torch_ref(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """Numerical alignment: one provider + one reuser with prefix sharing."""
    plan = _make_plan(batch_sizes=[8, 8], prefix_lens=[0, 4])
    # total_q = 8 + 4 = 12
    # total_kv = 8 + 8 = 16
    q, k, v = _random_qkv(
        total_q=12, total_kv=16, num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16
    )

    out_fa = backend.attention(q, k, v, plan)
    out_ref = ref_backend.attention(q, k, v, plan)

    assert out_fa.shape == out_ref.shape == q.shape
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_gqa(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """Flash Attention natively supports GQA (num_heads > num_kv_heads)."""
    plan = _make_plan(batch_sizes=[4, 6], prefix_lens=[0, 2])
    q, k, v = _random_qkv(
        total_q=8, total_kv=10, num_heads=4, num_kv_heads=2, head_dim=64, dtype=torch.float16
    )
    out_fa = backend.attention(q, k, v, plan)
    out_ref = ref_backend.attention(q, k, v, plan)
    assert out_fa.shape == q.shape
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_backward_smoke(backend: GpuFlashAttentionBackend) -> None:
    """Smoke test for gradient flow through the FA kernel."""
    plan = _make_plan(batch_sizes=[4, 6], prefix_lens=[0, 2])
    q, k, v = _random_qkv(
        total_q=8, total_kv=10, num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16
    )
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out = backend.attention(q, k, v, plan)
    loss = out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()
    assert not torch.isnan(v.grad).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_backward_vs_torch_ref(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """Gradient numerical alignment: FA2 vs PyTorch reference."""
    plan = _make_plan(batch_sizes=[8, 8], prefix_lens=[0, 4])
    q, k, v = _random_qkv(
        total_q=12, total_kv=16, num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16
    )

    out_fa, grads_fa = _run_forward_backward(backend, q, k, v, plan)
    out_ref, grads_ref = _run_forward_backward(ref_backend, q, k, v, plan)

    _assert_outputs_close(out_fa, out_ref)
    _assert_grads_close(grads_fa, grads_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_multi_reusers(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """1 provider + 2 reusers with different prefix lengths."""
    plan = _make_plan(batch_sizes=[10, 10, 10], prefix_lens=[0, 3, 5])
    # total_q = 10 + 7 + 5 = 22
    # total_kv = 10 + 10 + 10 = 30
    q, k, v = _random_qkv(
        total_q=22, total_kv=30, num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16
    )
    out_fa = backend.attention(q, k, v, plan)
    out_ref = ref_backend.attention(q, k, v, plan)
    assert out_fa.shape == q.shape
    assert out_fa.device == q.device
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_gqa_vs_torch_ref(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """Numerical alignment for GQA with prefix sharing."""
    plan = _make_plan(batch_sizes=[8, 8], prefix_lens=[0, 4])
    q, k, v = _random_qkv(
        total_q=12, total_kv=16, num_heads=4, num_kv_heads=2, head_dim=64, dtype=torch.float16
    )

    out_fa = backend.attention(q, k, v, plan)
    out_ref = ref_backend.attention(q, k, v, plan)

    assert out_fa.shape == out_ref.shape == q.shape
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_mixed_lengths(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """Mixed sequence lengths: provider short, reuser long, another provider medium."""
    plan = _make_plan(batch_sizes=[5, 12, 7], prefix_lens=[0, 5, 0])
    # total_q = 5 + 7 + 7 = 19
    # total_kv = 5 + 12 + 7 = 24
    q, k, v = _random_qkv(
        total_q=19, total_kv=24, num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16
    )
    out_fa = backend.attention(q, k, v, plan)
    out_ref = ref_backend.attention(q, k, v, plan)
    assert out_fa.shape == q.shape
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_various_head_dims(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
    head_dim: int,
) -> None:
    """Numerical alignment across common head_dim values."""
    plan = _make_plan(batch_sizes=[8, 8], prefix_lens=[0, 4])
    q, k, v = _random_qkv(
        total_q=12, total_kv=16, num_heads=2, num_kv_heads=2, head_dim=head_dim, dtype=torch.float16
    )
    out_fa = backend.attention(q, k, v, plan)
    out_ref = ref_backend.attention(q, k, v, plan)
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_multi_reusers_backward(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """1 provider + 2 reusers: gradient alignment."""
    plan = _make_plan(batch_sizes=[10, 10, 10], prefix_lens=[0, 3, 5])
    q, k, v = _random_qkv(
        total_q=22, total_kv=30, num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16
    )
    out_fa, grads_fa = _run_forward_backward(backend, q, k, v, plan)
    out_ref, grads_ref = _run_forward_backward(ref_backend, q, k, v, plan)
    _assert_outputs_close(out_fa, out_ref)
    _assert_grads_close(grads_fa, grads_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_mixed_lengths_backward(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """Mixed sequence lengths: gradient alignment."""
    plan = _make_plan(batch_sizes=[5, 12, 7], prefix_lens=[0, 5, 0])
    q, k, v = _random_qkv(
        total_q=19, total_kv=24, num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16
    )
    out_fa, grads_fa = _run_forward_backward(backend, q, k, v, plan)
    out_ref, grads_ref = _run_forward_backward(ref_backend, q, k, v, plan)
    _assert_outputs_close(out_fa, out_ref)
    _assert_grads_close(grads_fa, grads_ref)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_various_head_dims_backward(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
    head_dim: int,
) -> None:
    """Gradient alignment across common head_dim values."""
    plan = _make_plan(batch_sizes=[8, 8], prefix_lens=[0, 4])
    q, k, v = _random_qkv(
        total_q=12, total_kv=16, num_heads=2, num_kv_heads=2, head_dim=head_dim, dtype=torch.float16
    )
    out_fa, grads_fa = _run_forward_backward(backend, q, k, v, plan)
    out_ref, grads_ref = _run_forward_backward(ref_backend, q, k, v, plan)
    _assert_outputs_close(out_fa, out_ref)
    _assert_grads_close(grads_fa, grads_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gpu_flash_attn_validate_checks_import() -> None:
    """validate() should succeed when flash-attn is importable."""
    backend = GpuFlashAttentionBackend()
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="gpu_flash_attn")
    backend.validate(config)
