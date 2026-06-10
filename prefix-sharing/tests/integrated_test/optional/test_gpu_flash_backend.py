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

from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend  # noqa: E402
from prefix_sharing.backends.packed_layout import PackedBatchLayout  # noqa: E402
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
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_gpu")
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
    *,
    packed_batch_layout: Any | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    """Run attention + sum().backward() and return output + grads."""
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    out = backend.attention(q, k, v, plan, packed_batch_layout=packed_batch_layout)
    out.sum().backward()
    return out, {"q": q.grad, "k": k.grad, "v": v.grad}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_atten_gpu_same_q_kv_lengths(
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
def test_flash_atten_gpu_vs_torch_ref(
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
def test_flash_atten_gpu_gqa(
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
def test_flash_atten_gpu_backward_smoke(backend: GpuFlashAttentionBackend) -> None:
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
def test_flash_atten_gpu_backward_vs_torch_ref(
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
def test_flash_atten_gpu_multi_reusers(
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
def test_flash_atten_gpu_gqa_vs_torch_ref(
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
def test_flash_atten_gpu_mixed_lengths(
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
def test_flash_atten_gpu_various_head_dims(
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
def test_flash_atten_gpu_multi_reusers_backward(
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
def test_flash_atten_gpu_mixed_lengths_backward(
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
def test_flash_atten_gpu_various_head_dims_backward(
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


# ------------------------------------------------------------------
# Helpers for TP-padding test scenarios
# ------------------------------------------------------------------

def _make_packed_layout_with_padding(
    kept_lengths_q: list[int], align_size: int
) -> PackedBatchLayout:
    """Create a PackedBatchLayout with TP-style padding.

    Under TP>1 Megatron pads each row's token count to a multiple of
    *align_size* (``tp_size`` or ``tp_size * cp_size * 2``).  This
    helper mirrors :meth:`PackedBatchLayout.from_kept_position_rows`.
    """
    rows = [torch.zeros(length, dtype=torch.long) for length in kept_lengths_q]
    return PackedBatchLayout.from_kept_position_rows(rows, align_size=align_size)


# ------------------------------------------------------------------
# TP > 1: Q tensor is padded, K/V tensors are unpadded (after build_kv)
# ------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_atten_gpu_tp_padding_forward(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """TP=4: Q has padding (align to 4), K/V are unpadded.

    The Q tensor is padded to align with TP rank boundaries while
    K/V have already been de-padded by build_kv().  The FA backend
    must use padded cu_seqlens for Q batch boundaries.
    """
    original_lengths = [99, 81]
    prefix_lens = [0, 40]
    # kept_lengths_q: provider keeps full 99, reuser keeps 81-40=41
    # padded to align_size=4: [100, 44]
    plan = _make_plan(batch_sizes=original_lengths, prefix_lens=prefix_lens)
    layout = _make_packed_layout_with_padding(plan.kept_lengths_q, align_size=4)

    total_q = layout.total_padded_length  # 100 + 44 = 144
    total_kv = sum(plan.expanded_lengths_kv)  # 99 + 81 = 180

    q, k, v = _random_qkv(
        total_q=total_q, total_kv=total_kv,
        num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16,
    )

    out_fa = backend.attention(q, k, v, plan, packed_batch_layout=layout)
    out_ref = ref_backend.attention(q, k, v, plan, packed_batch_layout=layout)

    assert out_fa.shape == out_ref.shape == q.shape
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_atten_gpu_tp_padding_backward(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """TP=4: gradient alignment with Q padding."""
    original_lengths = [99, 81]
    prefix_lens = [0, 40]
    plan = _make_plan(batch_sizes=original_lengths, prefix_lens=prefix_lens)
    layout = _make_packed_layout_with_padding(plan.kept_lengths_q, align_size=4)

    total_q = layout.total_padded_length
    total_kv = sum(plan.expanded_lengths_kv)

    q, k, v = _random_qkv(
        total_q=total_q, total_kv=total_kv,
        num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16,
    )

    out_fa, grads_fa = _run_forward_backward(
        backend, q, k, v, plan, packed_batch_layout=layout,
    )
    out_ref, grads_ref = _run_forward_backward(
        ref_backend, q, k, v, plan, packed_batch_layout=layout,
    )

    _assert_outputs_close(out_fa, out_ref)
    _assert_grads_close(grads_fa, grads_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_atten_gpu_tp_padding_gqa(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """TP=8: GQA (4 Q heads, 2 KV heads) with Q padding."""
    original_lengths = [50, 70]
    prefix_lens = [0, 30]
    plan = _make_plan(batch_sizes=original_lengths, prefix_lens=prefix_lens)
    layout = _make_packed_layout_with_padding(plan.kept_lengths_q, align_size=8)

    total_q = layout.total_padded_length
    total_kv = sum(plan.expanded_lengths_kv)

    q, k, v = _random_qkv(
        total_q=total_q, total_kv=total_kv,
        num_heads=4, num_kv_heads=2, head_dim=64, dtype=torch.float16,
    )

    out_fa = backend.attention(q, k, v, plan, packed_batch_layout=layout)
    out_ref = ref_backend.attention(q, k, v, plan, packed_batch_layout=layout)

    assert out_fa.shape == out_ref.shape == q.shape
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_atten_gpu_tp_padding_multi_reusers(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """TP=4: 1 provider + 2 reusers with Q padding."""
    original_lengths = [50, 50, 50]
    prefix_lens = [0, 20, 35]
    plan = _make_plan(batch_sizes=original_lengths, prefix_lens=prefix_lens)
    layout = _make_packed_layout_with_padding(plan.kept_lengths_q, align_size=4)

    total_q = layout.total_padded_length
    total_kv = sum(plan.expanded_lengths_kv)

    q, k, v = _random_qkv(
        total_q=total_q, total_kv=total_kv,
        num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16,
    )

    out_fa = backend.attention(q, k, v, plan, packed_batch_layout=layout)
    out_ref = ref_backend.attention(q, k, v, plan, packed_batch_layout=layout)

    assert out_fa.shape == out_ref.shape == q.shape
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_atten_gpu_tp_padding_no_sharing(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """TP=4: all providers (no prefix sharing) with Q padding."""
    original_lengths = [35, 67, 42]
    prefix_lens = [0, 0, 0]
    plan = _make_plan(batch_sizes=original_lengths, prefix_lens=prefix_lens)
    layout = _make_packed_layout_with_padding(plan.kept_lengths_q, align_size=4)

    total_q = layout.total_padded_length
    total_kv = sum(plan.expanded_lengths_kv)

    q, k, v = _random_qkv(
        total_q=total_q, total_kv=total_kv,
        num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16,
    )

    out_fa = backend.attention(q, k, v, plan, packed_batch_layout=layout)
    out_ref = ref_backend.attention(q, k, v, plan, packed_batch_layout=layout)

    assert out_fa.shape == out_ref.shape == q.shape
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_atten_gpu_tp1_no_padding_still_works(
    backend: GpuFlashAttentionBackend,
    ref_backend: TorchReferenceBackend,
) -> None:
    """TP=1: layout with no actual padding (align_size=1) — backward compat."""
    original_lengths = [8, 8]
    prefix_lens = [0, 4]
    plan = _make_plan(batch_sizes=original_lengths, prefix_lens=prefix_lens)
    # align_size=1 never introduces padding
    layout = _make_packed_layout_with_padding(plan.kept_lengths_q, align_size=1)

    total_q = layout.total_padded_length  # equals sum(kept_lengths_q)
    total_kv = sum(plan.expanded_lengths_kv)

    q, k, v = _random_qkv(
        total_q=total_q, total_kv=total_kv,
        num_heads=2, num_kv_heads=2, head_dim=64, dtype=torch.float16,
    )

    out_fa = backend.attention(q, k, v, plan, packed_batch_layout=layout)
    out_ref = ref_backend.attention(q, k, v, plan, packed_batch_layout=layout)

    assert out_fa.shape == out_ref.shape == q.shape
    _assert_outputs_close(out_fa, out_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_atten_gpu_validate_checks_import() -> None:
    """validate() should succeed when flash-attn is importable."""
    backend = GpuFlashAttentionBackend()
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_gpu")
    backend.validate(config)
