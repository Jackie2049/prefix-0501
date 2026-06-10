"""Optional integration tests for the NPU Flash Attention backend.

These tests verify the NPU backend's control flow, parameter preparation,
reuser fallback logic, and integration with shared infrastructure. They are
designed to run on CPU/GPU environments without NPU hardware by mocking the
MindSpeed kernel.

Tests that require actual NPU hardware are gated behind the
``NPU_HARDWARE_TEST=1`` environment variable.
"""

from __future__ import annotations

import inspect
import os
from typing import Any
from unittest import mock

import pytest

pytest.importorskip("torch")

import torch

from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.torch_ref import TorchReferenceBackend, PrefixAttentionStore
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner, PrefixSharingPlan

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Mock helpers for NPU kernel
# ------------------------------------------------------------------

def _mock_npu_fusion_attention(q, k, v, num_heads, layout, *,
                                atten_mask=None, scale=None, keep_prob=None,
                                sparse_mode=0, actual_seq_qlen=None,
                                actual_seq_kvlen=None):
    """Mock that returns the same shape output as the real kernel.

    The real npu_fusion_attention returns a tuple of 7; [0] is the output.
    We produce a mock output with the correct shape.
    """
    out = torch.zeros_like(q)
    return (out, None, None, None, None, None, None)

# ------------------------------------------------------------------
# Fixtures and helpers
# ------------------------------------------------------------------

@pytest.fixture
def backend() -> NpuFlashAttentionBackend:
    return NpuFlashAttentionBackend()

@pytest.fixture
def ref_backend() -> TorchReferenceBackend:
    return TorchReferenceBackend()

def _make_plan(batch_sizes: list[int], prefix_lens: list[int]) -> PrefixSharingPlan:
    """Build a PrefixSharingPlan with the given structural layout."""
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1)
    planner = PrefixSharingPlanner(config)

    # Build sequences that naturally produce the desired prefix_lens.
    sequences: list[list[int]] = []
    next_token = 100
    provider_tokens: dict[int, list[int]] = {}
    for i, (size, p) in enumerate(zip(batch_sizes, prefix_lens)):
        if p == 0:
            seq = list(range(next_token, next_token + size))
            next_token += size
            sequences.append(seq)
            provider_tokens[i] = seq
        else:
            provider_idx = max(
                (j for j, s in enumerate(prefix_lens[:i]) if s == 0),
                default=None,
            )
            assert provider_idx is not None
            provider_seq = provider_tokens[provider_idx]
            suffix = list(range(next_token, next_token + size - p))
            next_token += size - p
            sequences.append(provider_seq[:p] + suffix)

    return planner.plan(sequences)

def _make_packed_layout_with_padding(
    kept_lengths_q: list[int], align_size: int
) -> PackedBatchLayout:
    """Create a PackedBatchLayout with TP-style padding."""
    rows = [torch.zeros(length, dtype=torch.long) for length in kept_lengths_q]
    return PackedBatchLayout.from_kept_position_rows(rows, align_size=align_size)

def _random_qkv(
    total_q: int,
    total_kv: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=DEVICE) * 0.02
    k = torch.randn(total_kv, num_kv_heads, head_dim, dtype=dtype, device=DEVICE) * 0.02
    v = torch.randn(total_kv, num_kv_heads, head_dim, dtype=dtype, device=DEVICE) * 0.02
    return q, k, v

# ------------------------------------------------------------------
# Backend instantiation and validation
# ------------------------------------------------------------------

class TestNpuBackendInstantiation:
    """Backend can be instantiated and configured without NPU hardware."""

    def test_backend_instantiable(self, backend: NpuFlashAttentionBackend) -> None:
        assert backend is not None
        assert isinstance(backend, NpuFlashAttentionBackend)

    def test_capabilities_are_correct(self, backend: NpuFlashAttentionBackend) -> None:
        caps = backend.capabilities
        assert caps.name == "flash_atten_npu"
        assert caps.supports_cann is True
        assert caps.supports_cuda is False
        assert caps.supports_cpu is False
        assert caps.supports_different_q_kv_lengths is True
        assert caps.supports_prefix_last_restore is True

    def test_validate_succeeds_with_known_backend_name(self) -> None:
        backend = NpuFlashAttentionBackend()
        config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_npu")
        # validate() tries to import mindspeed — mock it for CPU testing
        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=_mock_npu_fusion_attention,
        ):
            backend.validate(config)

    def test_validate_fails_when_mindspeed_not_installed(self) -> None:
        backend = NpuFlashAttentionBackend()
        config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_npu")
        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            side_effect=RuntimeError("MindSpeed not found"),
        ):
            with pytest.raises(RuntimeError, match="MindSpeed"):
                backend.validate(config)

# ------------------------------------------------------------------
# apply_rope and build_kv delegation
# ------------------------------------------------------------------

class TestNpuBackendDelegation:
    """apply_rope and build_kv delegate to TorchReferenceBackend."""

    def test_apply_rope_delegates_to_torch_ref(self, backend: NpuFlashAttentionBackend) -> None:
        plan = _make_plan(batch_sizes=[4, 4], prefix_lens=[0, 2])
        q = torch.randn(6, 2, 32)
        k = torch.randn(8, 2, 32)

        q_out, k_out = backend.apply_rope(q, k, plan)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_build_kv_delegates_to_torch_ref(self, backend: NpuFlashAttentionBackend) -> None:
        plan = _make_plan(batch_sizes=[4, 4], prefix_lens=[0, 2])
        k = torch.randn(6, 2, 32)  # kept_lengths = [4, 2] = 6
        v = torch.randn(6, 2, 32)
        store = PrefixAttentionStore()

        k_out, v_out = backend.build_kv(k, v, store, plan, layer_id=0, tp_rank=0)
        expected_kv_len = sum(plan.expanded_lengths_kv)  # [4, 4] = 8
        assert k_out.shape[0] == expected_kv_len
        assert v_out.shape[0] == expected_kv_len
        assert k_out.shape[1:] == k.shape[1:]
        assert v_out.shape[1:] == v.shape[1:]

    def test_build_kv_accepts_packed_batch_layout(self, backend: NpuFlashAttentionBackend) -> None:
        plan = _make_plan(batch_sizes=[4, 4], prefix_lens=[0, 2])
        k = torch.randn(6, 2, 32)
        v = torch.randn(6, 2, 32)
        store = PrefixAttentionStore()

        layout = PackedBatchLayout.from_valid_lengths([4, 2])
        k_out, v_out = backend.build_kv(
            k, v, store, plan, packed_batch_layout=layout, layer_id=0, tp_rank=0,
        )
        assert k_out.shape[0] == sum(plan.expanded_lengths_kv)

# ------------------------------------------------------------------
# Reuser detection and routing
# ------------------------------------------------------------------

class TestNpuReuserDetection:
    """Verify that reuser plans fall back to TorchReferenceBackend."""

    def test_provider_only_plan_uses_npu_kernel(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """Provider-only plan → NPU kernel path (mock is called)."""
        plan = _make_plan(batch_sizes=[4, 6], prefix_lens=[0, 0])
        q, k, v = _random_qkv(total_q=10, total_kv=10, num_heads=2,
                              num_kv_heads=2, head_dim=32)

        called = [False]

        def tracking_mock(*args, **kwargs):
            called[0] = True
            return _mock_npu_fusion_attention(*args, **kwargs)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=tracking_mock,
        ):
            out = backend.attention(q, k, v, plan)
            assert out.shape == q.shape
            assert called[0], (
                "NPU kernel should be called for provider-only plan"
            )

    def test_reuser_plan_falls_back_to_torch_ref(
        self, backend: NpuFlashAttentionBackend, ref_backend: TorchReferenceBackend,
    ) -> None:
        """Plan with reuser → torch_ref fallback (NPU kernel NOT called)."""
        plan = _make_plan(batch_sizes=[8, 8], prefix_lens=[0, 4])
        q, k, v = _random_qkv(total_q=12, total_kv=16, num_heads=2,
                              num_kv_heads=2, head_dim=32)

        called = [False]

        def tracking_mock(*args, **kwargs):
            called[0] = True
            return _mock_npu_fusion_attention(*args, **kwargs)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=tracking_mock,
        ):
            out = backend.attention(q, k, v, plan)
            assert out.shape == q.shape
            assert not called[0], (
                "NPU kernel must NOT be called for plan with reusers — "
                "should fall back to TorchReferenceBackend"
            )

    def test_reuser_fallback_output_matches_torch_ref(
        self, backend: NpuFlashAttentionBackend, ref_backend: TorchReferenceBackend,
    ) -> None:
        """Reuser fallback output matches TorchReferenceBackend directly."""
        plan = _make_plan(batch_sizes=[8, 8], prefix_lens=[0, 4])
        q, k, v = _random_qkv(total_q=12, total_kv=16, num_heads=2,
                              num_kv_heads=2, head_dim=32)

        out_backend = backend.attention(q, k, v, plan)
        out_ref = ref_backend.attention(q, k, v, plan)

        assert out_backend.shape == out_ref.shape
        assert torch.allclose(out_backend, out_ref, atol=1e-5), (
            "Reuser fallback output should match TorchReferenceBackend"
        )

    def test_mixed_plan_with_reuser_falls_back(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """Plan with mixed providers and reusers → falls back."""
        plan = _make_plan(batch_sizes=[5, 12, 7], prefix_lens=[0, 5, 0])
        q, k, v = _random_qkv(total_q=19, total_kv=24, num_heads=2,
                              num_kv_heads=2, head_dim=32)

        called = [False]

        def tracking_mock(*args, **kwargs):
            called[0] = True
            return _mock_npu_fusion_attention(*args, **kwargs)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=tracking_mock,
        ):
            out = backend.attention(q, k, v, plan)
            assert out.shape == q.shape
            assert not called[0], (
                "NPU kernel must NOT be called for plan with any reuser"
            )

    def test_multi_reusers_falls_back(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """1 provider + 2 reusers → falls back to torch_ref."""
        plan = _make_plan(batch_sizes=[10, 10, 10], prefix_lens=[0, 3, 5])
        q, k, v = _random_qkv(total_q=22, total_kv=30, num_heads=2,
                              num_kv_heads=2, head_dim=32)

        called = [False]

        def tracking_mock(*args, **kwargs):
            called[0] = True
            return _mock_npu_fusion_attention(*args, **kwargs)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=tracking_mock,
        ):
            out = backend.attention(q, k, v, plan)
            assert out.shape == q.shape
            assert not called[0], "NPU kernel must NOT be called for multi-reuser plan"

# ------------------------------------------------------------------
# attention() control flow and parameter preparation (NPU path)
# ------------------------------------------------------------------

class TestNpuAttentionControlFlow:
    """Verify parameter preparation when the NPU kernel IS used."""

    def test_attention_signature_includes_packed_batch_layout(self) -> None:
        """[BUG-04 check] attention() must accept packed_batch_layout parameter."""
        sig = inspect.signature(NpuFlashAttentionBackend.attention)
        assert "packed_batch_layout" in sig.parameters, (
            "NpuFlashAttentionBackend.attention() missing packed_batch_layout parameter"
        )

    def test_provider_only_passes_correct_actual_seq_lens(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """Verify actual_seq_qlen/kvlen are per-sample lengths (NOT cumulative).

        Uses a provider-only plan so the NPU kernel IS called.
        """
        # Two providers with different lengths — no sharing
        plan = _make_plan(batch_sizes=[6, 4], prefix_lens=[0, 0])
        q, k, v = _random_qkv(total_q=10, total_kv=10, num_heads=2,
                              num_kv_heads=2, head_dim=32)

        captured_kwargs = {}

        def capture_mock(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return _mock_npu_fusion_attention(*args, **kwargs)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=capture_mock,
        ):
            backend.attention(q, k, v, plan)

        assert captured_kwargs["actual_seq_qlen"] == [6, 4], (
            f"actual_seq_qlen should be [6, 4] (per-sample), "
            f"got {captured_kwargs['actual_seq_qlen']}"
        )
        assert captured_kwargs["actual_seq_kvlen"] == [6, 4], (
            f"actual_seq_kvlen should be [6, 4] (per-sample), "
            f"got {captured_kwargs['actual_seq_kvlen']}"
        )

    def test_provider_only_passes_correct_atten_mask_shape(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """Verify atten_mask shape is (max_seqlen_q, max_seqlen_kv), not (total_q, total_kv).

        The NPU kernel in TND mode requires atten_mask of shape
        (max_seqlen_q, max_seqlen_kv) — it broadcasts this single mask
        to every sample.
        """
        # Providers with different lengths: [6, 4] → max_q=6, max_kv=6
        plan = _make_plan(batch_sizes=[6, 4], prefix_lens=[0, 0])
        q, k, v = _random_qkv(total_q=10, total_kv=10, num_heads=2,
                              num_kv_heads=2, head_dim=32)

        captured_kwargs = {}

        def capture_mock(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return _mock_npu_fusion_attention(*args, **kwargs)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=capture_mock,
        ):
            backend.attention(q, k, v, plan)

        atten_mask = captured_kwargs.get("atten_mask")
        assert atten_mask is not None, "atten_mask not passed to kernel"
        assert atten_mask.dtype == torch.bool, (
            f"atten_mask dtype should be bool, got {atten_mask.dtype}"
        )
        assert atten_mask.shape == (plan.max_seqlen_q, plan.max_seqlen_kv), (
            f"atten_mask shape should be (max_q, max_kv) = "
            f"({plan.max_seqlen_q}, {plan.max_seqlen_kv}), "
            f"got {atten_mask.shape}"
        )

    def test_provider_only_atten_mask_is_causal(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """Verify the atten_mask passed to the kernel is standard causal."""
        plan = _make_plan(batch_sizes=[5, 3], prefix_lens=[0, 0])
        q, k, v = _random_qkv(total_q=8, total_kv=8, num_heads=2,
                              num_kv_heads=2, head_dim=32)

        captured_kwargs = {}

        def capture_mock(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return _mock_npu_fusion_attention(*args, **kwargs)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=capture_mock,
        ):
            backend.attention(q, k, v, plan)

        atten_mask = captured_kwargs["atten_mask"]
        # Standard causal: Q[i] can attend to KV[0..i]
        # True = masked, so triu(1) should be masked
        for i in range(plan.max_seqlen_q):
            for j in range(plan.max_seqlen_kv):
                if j <= i:
                    assert not atten_mask[i, j].item(), (
                        f"atten_mask[{i},{j}] should be False (visible) for j<=i"
                    )
                else:
                    assert atten_mask[i, j].item(), (
                        f"atten_mask[{i},{j}] should be True (masked) for j>i"
                    )

    def test_attention_prepare_flash_inputs_returns_8_values(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """_prepare_flash_inputs returns 8 values (including pad_layout)."""
        plan = _make_plan(batch_sizes=[6, 4], prefix_lens=[0, 2])
        q = torch.randn(8, 2, 32)   # kept_lengths_q = [6, 2] = 8
        k = torch.randn(10, 2, 32)  # expanded_lengths_kv = [6, 4] = 10
        v = torch.randn(10, 2, 32)

        result = backend._prepare_flash_inputs(q, k, v, plan)
        assert len(result) == 8, (
            f"_prepare_flash_inputs returned {len(result)} values, expected 8"
        )

    def test_provider_only_gqa_uses_npu_kernel(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """GQA with provider-only plan → NPU kernel path."""
        plan = _make_plan(batch_sizes=[8, 6], prefix_lens=[0, 0])
        q, k, v = _random_qkv(total_q=14, total_kv=14, num_heads=4,
                              num_kv_heads=2, head_dim=32)

        called = [False]

        def tracking_mock(*args, **kwargs):
            called[0] = True
            return _mock_npu_fusion_attention(*args, **kwargs)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=tracking_mock,
        ):
            out = backend.attention(q, k, v, plan)
            assert out.shape == q.shape
            assert called[0], "NPU kernel should be called for provider-only GQA"

# ------------------------------------------------------------------
# TP padding support
# ------------------------------------------------------------------

class TestNpuAttentionTpPadding:
    """Verify TP padding handling in the NPU backend."""

    def test_attention_with_tp_padding_no_sharing_uses_npu(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """TP=4: all providers (no sharing) with Q padding → NPU kernel used."""
        original_lengths = [35, 67, 42]
        prefix_lens = [0, 0, 0]
        plan = _make_plan(batch_sizes=original_lengths, prefix_lens=prefix_lens)
        layout = _make_packed_layout_with_padding(plan.kept_lengths_q, align_size=4)

        total_q = layout.total_padded_length
        total_kv = sum(plan.expanded_lengths_kv)

        q, k, v = _random_qkv(total_q=total_q, total_kv=total_kv,
                              num_heads=2, num_kv_heads=2, head_dim=32)

        called = [False]

        def tracking_mock(*args, **kwargs):
            called[0] = True
            return _mock_npu_fusion_attention(*args, **kwargs)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=tracking_mock,
        ):
            out = backend.attention(q, k, v, plan, packed_batch_layout=layout)
            assert out.shape == q.shape
            assert called[0], "NPU kernel should be called for provider-only plan"

    def test_attention_with_tp_padding_reuser_falls_back(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """TP=4: has reusers → falls back to torch_ref, output shape preserved."""
        original_lengths = [99, 81]
        prefix_lens = [0, 40]
        plan = _make_plan(batch_sizes=original_lengths, prefix_lens=prefix_lens)
        layout = _make_packed_layout_with_padding(plan.kept_lengths_q, align_size=4)

        total_q = layout.total_padded_length  # padded
        total_kv = sum(plan.expanded_lengths_kv)  # unpadded

        q, k, v = _random_qkv(total_q=total_q, total_kv=total_kv,
                              num_heads=2, num_kv_heads=2, head_dim=32)

        called = [False]

        def tracking_mock(*args, **kwargs):
            called[0] = True
            return _mock_npu_fusion_attention(*args, **kwargs)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=tracking_mock,
        ):
            out = backend.attention(q, k, v, plan, packed_batch_layout=layout)
            # Output must match the padded Q shape (repad happens in attention())
            assert out.shape == q.shape, (
                f"Output shape {out.shape} != padded Q shape {q.shape}"
            )
            assert not called[0], (
                "NPU kernel should NOT be called for reuser plan (fallback to torch_ref)"
            )

    def test_attention_tp1_no_padding_reuser_falls_back(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """TP=1: layout with no actual padding, reuser plan → torch_ref fallback."""
        original_lengths = [8, 8]
        prefix_lens = [0, 4]
        plan = _make_plan(batch_sizes=original_lengths, prefix_lens=prefix_lens)
        layout = _make_packed_layout_with_padding(plan.kept_lengths_q, align_size=1)

        total_q = layout.total_padded_length
        total_kv = sum(plan.expanded_lengths_kv)

        q, k, v = _random_qkv(total_q=total_q, total_kv=total_kv,
                              num_heads=2, num_kv_heads=2, head_dim=32)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=_mock_npu_fusion_attention,
        ):
            out = backend.attention(q, k, v, plan, packed_batch_layout=layout)
            assert out.shape == q.shape

    def test_unpad_repad_roundtrip_provider_only(
        self, backend: NpuFlashAttentionBackend,
    ) -> None:
        """Verify TP padding unpad→kernel→repad for provider-only plan."""
        # Use providers only so NPU kernel IS called
        original_lengths = [50, 70]
        prefix_lens = [0, 0]
        plan = _make_plan(batch_sizes=original_lengths, prefix_lens=prefix_lens)
        layout = _make_packed_layout_with_padding(plan.kept_lengths_q, align_size=4)

        total_q = layout.total_padded_length
        total_kv = sum(plan.expanded_lengths_kv)

        q, k, v = _random_qkv(total_q=total_q, total_kv=total_kv,
                              num_heads=2, num_kv_heads=2, head_dim=32)

        captured_q = []

        def capture_mock(q_arg, *args, **kwargs):
            captured_q.append(q_arg.shape)
            return _mock_npu_fusion_attention(q_arg, *args, **kwargs)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=capture_mock,
        ):
            out = backend.attention(q, k, v, plan, packed_batch_layout=layout)

        # The kernel should receive unpadded Q (valid lengths only)
        total_valid = layout.total_valid_length
        assert captured_q[0][0] == total_valid, (
            f"Kernel received Q with shape[0]={captured_q[0][0]}, "
            f"expected valid total length {total_valid}"
        )
        # Output should be repadded to the original padded shape
        assert out.shape[0] == total_q, (
            f"Repadded output shape[0]={out.shape[0]}, "
            f"expected padded total length {total_q}"
        )

# ------------------------------------------------------------------
# Factory integration
# ------------------------------------------------------------------

class TestNpuBackendFactory:
    """Verify factory can create NPU backend instances."""

    def test_factory_creates_npu_backend(self) -> None:
        from prefix_sharing.backends.factory import get_backend_instance

        config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_npu")
        backend = get_backend_instance(config)
        assert isinstance(backend, NpuFlashAttentionBackend)

    def test_factory_creates_npu_backend_with_explicit_backend(self) -> None:
        from prefix_sharing.backends.factory import get_backend_instance

        config = PrefixSharingConfig(enable_prefix_sharing=True, backend="torch_ref")
        explicit = NpuFlashAttentionBackend()
        backend = get_backend_instance(config, backend=explicit)
        assert backend is explicit

# ============================================================================
# NPU hardware verification (requires NPU_HARDWARE_TEST=1)
# ============================================================================

@pytest.mark.skipif(
    os.environ.get("NPU_HARDWARE_TEST") != "1",
    reason="Set NPU_HARDWARE_TEST=1 to run on NPU hardware",
)
class TestNpuHardwareValidation:
    """End-to-end validation on actual NPU hardware.

    Run with::

        NPU_HARDWARE_TEST=1 python -m pytest ...test_npu_flash_backend.py::TestNpuHardwareValidation -v

    Prerequisites:
      - torch_npu available
      - MindSpeed installed (mindspeed.ops.fusion_attention_v2.npu_fusion_attention)
    """

    def test_npu_is_available(self) -> None:
        """Verify NPU device is available."""
        try:
            import torch_npu  # noqa: F401
            assert torch.npu.is_available(), "NPU not available"
        except ImportError:
            pytest.fail("torch_npu not installed, cannot run NPU hardware tests")

    def test_mindspeed_importable(self) -> None:
        """Verify MindSpeed can be imported."""
        try:
            from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention  # noqa: F401
        except ModuleNotFoundError as e:
            pytest.fail(f"MindSpeed import failed: {e}")

    def test_npu_attention_no_sharing_basic(self) -> None:
        """Basic forward: no prefix sharing — uses NPU kernel."""
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
        planner = PrefixSharingPlanner(config)
        plan = planner.plan([[1, 2, 3, 4], [5, 6, 7, 8]])

        backend = NpuFlashAttentionBackend()

        total_q = plan.cu_seqlens_q[-1]
        total_kv = plan.cu_seqlens_kv[-1]

        q = torch.randn(total_q, 2, 32, device="npu")
        k = torch.randn(total_kv, 2, 32, device="npu")
        v = torch.randn(total_kv, 2, 32, device="npu")

        result = backend.attention(q, k, v, plan)
        assert result.shape == q.shape
        assert not torch.isnan(result).any(), "Output contains NaN"
        assert not torch.isinf(result).any(), "Output contains Inf"

    def test_npu_attention_with_reuser_falls_back(self) -> None:
        """Forward with prefix sharing — falls back to TorchReferenceBackend."""
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
        planner = PrefixSharingPlanner(config)
        plan = planner.plan([[1, 2, 3, 4], [1, 2, 5, 6]])

        backend = NpuFlashAttentionBackend()

        total_q = plan.cu_seqlens_q[-1]
        total_kv = plan.cu_seqlens_kv[-1]

        q = torch.randn(total_q, 2, 32, device="npu")
        k = torch.randn(total_kv, 2, 32, device="npu")
        v = torch.randn(total_kv, 2, 32, device="npu")

        result = backend.attention(q, k, v, plan)
        assert result.shape == q.shape
        assert not torch.isnan(result).any(), "Output contains NaN"
        assert not torch.isinf(result).any(), "Output contains Inf"

    def test_npu_attention_with_tp_padding(self) -> None:
        """Forward with TP padding (simulated) — provider-only to test NPU path."""
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
        planner = PrefixSharingPlanner(config)
        plan = planner.plan([[1, 2, 3, 4], [5, 6, 7, 8]])

        backend = NpuFlashAttentionBackend()

        layout = PackedBatchLayout.from_kept_position_rows(
            [torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3])],
            align_size=4,
        )

        total_q = layout.total_padded_length
        total_kv = plan.cu_seqlens_kv[-1]

        q = torch.randn(total_q, 2, 32, device="npu")
        k = torch.randn(total_kv, 2, 32, device="npu")
        v = torch.randn(total_kv, 2, 32, device="npu")

        result = backend.attention(q, k, v, plan, packed_batch_layout=layout)
        assert result.shape == q.shape
        assert not torch.isnan(result).any(), "Output contains NaN"
        assert not torch.isinf(result).any(), "Output contains Inf"
