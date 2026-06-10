"""NPU FA bug verification tests.

This file verifies that known NPU bugs from the original ``npu_fa_fix`` branch
are NOT present in the current implementation. Tests are organized by bug
severity level (blocking, high-risk, low-risk).

All tests run without NPU hardware — they are static/logical/unit-level checks
that can run in CPU/GPU environments. When all pass, the code is ready for
end-to-end validation on NPU hardware.

Run::

    python -m pytest prefix-sharing/tests/integrated_test/optional/test_npu_fixes.py -v
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan, PrefixSharingPlanner

# ============================================================================
# Helpers
# ============================================================================

def _make_multi_sample_plan() -> PrefixSharingPlan:
    """Build a three-sample plan for verification tests.

    batch_size=3:
      - sample 0: provider, full length 6, prefix_len=0
      - sample 1: reuser,   full length 6, prefix_len=3, suffix=3
      - sample 2: standalone, full length 4, prefix_len=0

    Key data:
      cu_seqlens_q  = [0, 6, 9, 13]
      cu_seqlens_kv = [0, 6, 12, 16]
      kept_lengths_q  = [6, 3, 4]
      expanded_lengths_kv = [6, 6, 4]
    """
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1)
    planner = PrefixSharingPlanner(config)
    plan = planner.plan([
        [100, 101, 102, 103, 104, 105],
        [100, 101, 102, 200, 201, 202],
        [300, 301, 302, 303],
    ])
    assert plan.batch_size == 3
    assert plan.cu_seqlens_q == [0, 6, 9, 13]
    assert plan.cu_seqlens_kv == [0, 6, 12, 16]
    assert plan.kept_lengths_q == [6, 3, 4]
    assert plan.expanded_lengths_kv == [6, 6, 4]
    return plan

def _read_source_lines(cls: type, method_name: str) -> list[str]:
    """Read source lines of a class method."""
    import textwrap
    src = textwrap.dedent(inspect.getsource(getattr(cls, method_name)))
    return src.splitlines()

# ============================================================================
# 🔴 Blocking-level bugs — would prevent correct execution
# ============================================================================

class TestBug01PackedLayoutHasUnpadRepad:
    """[BUG-01] PackedBatchLayout must have has_padding / unpad / repad methods.

    When TP > 1, PackedBatchLayout creates alignment padding tokens.
    unpad strips them before attention; repad restores them after.
    NPU version deleted these three methods, causing crashes at TP > 1.
    """

    def test_has_padding_property_exists(self) -> None:
        layout = PackedBatchLayout.from_kept_position_rows(
            [torch.tensor([0, 1, 2]), torch.tensor([3, 4])],
            align_size=4,
        )
        assert hasattr(layout, "has_padding"), (
            "BUG-01: PackedBatchLayout missing has_padding property"
        )
        assert layout.has_padding is True

    def test_unpad_method_exists_and_strips_padding(self) -> None:
        layout = PackedBatchLayout.from_kept_position_rows(
            [torch.tensor([0, 1, 2]), torch.tensor([3, 4])],
            align_size=4,
        )
        assert hasattr(layout, "unpad"), (
            "BUG-01: PackedBatchLayout missing unpad() method"
        )
        # valid=[3,2], padded=[4,4], total_padded=8, total_valid=5
        total_padded = layout.total_padded_length
        total_valid = layout.total_valid_length
        x = torch.randn(total_padded, 8)
        y = layout.unpad(x)
        assert y.shape == (total_valid, 8), f"unpad shape should be ({total_valid},8), got {y.shape}"

    def test_repad_method_exists_and_restores_padding(self) -> None:
        layout = PackedBatchLayout.from_kept_position_rows(
            [torch.tensor([0, 1, 2]), torch.tensor([3, 4])],
            align_size=4,
        )
        assert hasattr(layout, "repad"), (
            "BUG-01: PackedBatchLayout missing repad() method"
        )
        total_padded = layout.total_padded_length
        total_valid = layout.total_valid_length
        x = torch.randn(total_valid, 8)
        y = layout.repad(x)
        assert y.shape == (total_padded, 8), f"repad shape should be ({total_padded},8), got {y.shape}"

    def test_unpad_repad_roundtrip_preserves_valid_tokens(self) -> None:
        layout = PackedBatchLayout.from_kept_position_rows(
            [torch.tensor([0, 1, 2]), torch.tensor([3, 4])],
            align_size=4,
        )
        total_padded = layout.total_padded_length
        valid_lens = layout.valid_lengths  # [3, 2]
        padded_lens = layout.padded_lengths  # [4, 4]

        x_padded = torch.randn(total_padded, 4)
        x_unpadded = layout.unpad(x_padded)
        x_repadded = layout.repad(x_unpadded)

        # row 0: valid[0:3], pad[3:4]
        assert torch.allclose(x_repadded[0:valid_lens[0]], x_padded[0:valid_lens[0]]), \
            "roundtrip: row 0 valid tokens value mismatch"
        assert torch.all(x_repadded[valid_lens[0]:padded_lens[0]] == 0), "repad: row 0 padding should be zero"

        # row 1: valid offset = padded_lens[0] to padded_lens[0]+valid_lens[1]
        row1_start = padded_lens[0]
        row1_valid_end = row1_start + valid_lens[1]
        row1_padded_end = row1_start + padded_lens[1]
        assert torch.allclose(x_repadded[row1_start:row1_valid_end], x_padded[row1_start:row1_valid_end]), \
            "roundtrip: row 1 valid tokens value mismatch"
        assert torch.all(x_repadded[row1_valid_end:row1_padded_end] == 0), "repad: row 1 padding should be zero"

    def test_repad_gradient_flows_to_valid_input(self) -> None:
        layout = PackedBatchLayout.from_kept_position_rows(
            [torch.tensor([0, 1, 2]), torch.tensor([3, 4])],
            align_size=4,
        )
        total_valid = layout.total_valid_length
        x = torch.randn(total_valid, 2, requires_grad=True)
        y = layout.repad(x)
        y.sum().backward()

        assert x.grad is not None, (
            "BUG-01: repad output backward fails, x.grad is None"
        )
        assert torch.allclose(x.grad, torch.ones_like(x)), \
            "sum().backward() should produce all-ones gradient"

class TestBug02FlashAttenBaseHasTpPadding:
    """[BUG-02] FlashAttentionMixin must have _strip_tp_padding / _repad_output.

    NPU version deleted these methods and removed packed_batch_layout from
    _prepare_flash_inputs, causing TypeError and shape mismatches at TP > 1.
    """

    def test_prepare_flash_inputs_accepts_packed_batch_layout(self) -> None:
        from prefix_sharing.backends.flash_atten_base import FlashAttentionMixin

        sig = inspect.signature(FlashAttentionMixin._prepare_flash_inputs)
        assert "packed_batch_layout" in sig.parameters, (
            "BUG-02: _prepare_flash_inputs missing packed_batch_layout parameter"
        )

    def test_strip_tp_padding_method_exists(self) -> None:
        from prefix_sharing.backends.flash_atten_base import FlashAttentionMixin

        assert hasattr(FlashAttentionMixin, "_strip_tp_padding"), (
            "BUG-02: FlashAttentionMixin missing _strip_tp_padding() method"
        )

    def test_repad_output_method_exists(self) -> None:
        from prefix_sharing.backends.flash_atten_base import FlashAttentionMixin

        assert hasattr(FlashAttentionMixin, "_repad_output"), (
            "BUG-02: FlashAttentionMixin missing _repad_output() method"
        )

    def test_prepare_flash_inputs_returns_8_values(self) -> None:
        from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend

        backend = GpuFlashAttentionBackend()
        plan = _make_multi_sample_plan()

        q = torch.randn(13, 4, 64)
        k = torch.randn(16, 4, 64)
        v = torch.randn(16, 4, 64)

        result = backend._prepare_flash_inputs(q, k, v, plan)
        assert len(result) == 8, (
            f"BUG-02: _prepare_flash_inputs returns {len(result)} values, expected 8"
        )

class TestBug03ActualSeqLenCalculation:
    """[BUG-03] actual_seq_qlen/kvlen must use per-sample lengths, NOT cumulative.

    Source line in buggy NPU version:
        actual_seq_qlen = list(prefix_sharing_plan.cu_seqlens_q[1:])
    cu_seqlens is cumulative, so cu_seqlens[1:] gives cumulative values:
    [0, 6, 9, 13] → [6, 9, 13] instead of [6, 3, 4].

    This is a deterministic logic error causing wrong attention for batch_size > 1.
    """

    def test_cu_seqlens_1_is_cumulative_not_per_sample(self) -> None:
        plan = _make_multi_sample_plan()

        wrong_qlen = list(plan.cu_seqlens_q[1:])
        wrong_kvlen = list(plan.cu_seqlens_kv[1:])

        assert wrong_qlen == [6, 9, 13], "prerequisite: cu_seqlens_q[1:] = [6,9,13]"
        assert wrong_kvlen == [6, 12, 16], "prerequisite: cu_seqlens_kv[1:] = [6,12,16]"

        correct_qlen = plan.kept_lengths_q
        correct_kvlen = plan.expanded_lengths_kv

        assert correct_qlen == [6, 3, 4], "prerequisite: kept_lengths_q = [6,3,4]"
        assert correct_kvlen == [6, 6, 4], "prerequisite: expanded_lengths_kv = [6,6,4]"

        assert wrong_qlen != correct_qlen, (
            "For batch_size>1, cu_seqlens[1:] ≠ kept_lengths. "
            "Using cumulative values would be wrong."
        )
        assert wrong_kvlen != correct_kvlen, "Similarly, kvlen would be wrong."

    def test_actual_seq_qlen_source_code_not_cumulative(self) -> None:
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        lines = _read_source_lines(NpuFlashAttentionBackend, "attention")

        bug_line = None
        for line in lines:
            stripped = line.strip()
            if "actual_seq_qlen" in stripped and "cu_seqlens_q[1:]" in stripped:
                bug_line = stripped
                break

        assert bug_line is None, (
            f"BUG-03 CONFIRMED: flash_atten_npu.py uses cumulative cu_seqlens_q[1:]\n"
            f"  {bug_line}\n"
            "Fix: use kept_lengths_q and expanded_lengths_kv"
        )

    def test_actual_seq_kvlen_source_code_not_cumulative(self) -> None:
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        lines = _read_source_lines(NpuFlashAttentionBackend, "attention")

        bug_line = None
        for line in lines:
            stripped = line.strip()
            if "actual_seq_kvlen" in stripped and "cu_seqlens_kv[1:]" in stripped:
                bug_line = stripped
                break

        assert bug_line is None, (
            f"BUG-03 CONFIRMED: flash_atten_npu.py uses cumulative cu_seqlens_kv[1:]\n"
            f"  {bug_line}\n"
            "Fix: use kept_lengths_q and expanded_lengths_kv"
        )

class TestBug04NpuAttentionHasPackedBatchLayoutParam:
    """[BUG-04] NpuFlashAttentionBackend.attention() must accept packed_batch_layout.

    The integration layer (megatron_runtime.py / verl_mcore.py) passes
    packed_batch_layout when calling attention(). Missing this parameter
    causes immediate TypeError crash.
    """

    def test_attention_signature_includes_packed_batch_layout(self) -> None:
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        sig = inspect.signature(NpuFlashAttentionBackend.attention)
        params = list(sig.parameters.keys())

        assert "packed_batch_layout" in params, (
            f"BUG-04: NpuFlashAttentionBackend.attention() missing "
            f"'packed_batch_layout' parameter. Current params: {params}"
        )

    def test_gpu_flash_attention_signature_also_has_packed_batch_layout(self) -> None:
        from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend

        sig = inspect.signature(GpuFlashAttentionBackend.attention)
        params = list(sig.parameters.keys())

        assert "packed_batch_layout" in params, (
            f"GPU attention() also missing packed_batch_layout. Params: {params}"
        )

# ============================================================================
# 🟡 High-risk bugs — triggered under specific conditions
# ============================================================================

class TestBug05BlockCausalMaskModuleExists:
    """[BUG-05] block_causal_mask.py is a new module — verify it's importable and working."""

    def test_block_causal_mask_imports(self) -> None:
        try:
            from prefix_sharing.backends.block_causal_mask import (  # noqa: F401
                build_block_causal_mask,
                mask_to_te_bias,
            )
        except ImportError as e:
            pytest.fail(f"BUG-05: block_causal_mask import failed: {e}")

    def test_mask_is_on_correct_device(self) -> None:
        from prefix_sharing.backends.block_causal_mask import build_block_causal_mask

        plan = _make_multi_sample_plan()
        mask = build_block_causal_mask(plan, device=torch.device("cpu"))

        assert mask.device.type == "cpu", "mask should be on specified device"
        assert mask.dtype == torch.bool, "mask should be bool (True=masked)"
        assert mask.shape == (13, 16), (
            f"mask shape should be (total_q, total_kv) = (13, 16), got {mask.shape}"
        )

class TestBug06NpuBackendAttentionDispatch:
    """[BUG-06] NpuFlashAttentionBackend overall control flow correctness.

    Verify parameter preparation and routing logic without actual NPU hardware.
    """

    def test_attention_output_matches_shape(self) -> None:
        """Attention always returns tensor with the right shape.

        Uses _make_multi_sample_plan() which has a reuser — the NPU backend
        will internally fall back to TorchReferenceBackend. The output shape
        must still be correct.
        """
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        backend = NpuFlashAttentionBackend()
        plan = _make_multi_sample_plan()

        q = torch.randn(13, 4, 64)
        k = torch.randn(16, 4, 64)
        v = torch.randn(16, 4, 64)

        result = backend.attention(q, k, v, plan)
        assert result.shape == q.shape

    def test_provider_only_uses_npu_kernel_path(self) -> None:
        """Provider-only plan must use the NPU kernel (not fallback)."""
        from unittest import mock
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
        planner = PrefixSharingPlanner(config)
        plan = planner.plan([[1, 2, 3, 4], [5, 6, 7, 8]])  # providers only

        backend = NpuFlashAttentionBackend()
        q = torch.randn(8, 2, 32)
        k = torch.randn(8, 2, 32)
        v = torch.randn(8, 2, 32)

        called = [False]

        def mock_npu(*args, **kwargs):
            called[0] = True
            return (torch.zeros_like(args[0]), None, None, None, None, None, None)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=mock_npu,
        ):
            result = backend.attention(q, k, v, plan)
            assert result.shape == q.shape
            assert called[0], (
                "NPU kernel should be called for provider-only plan"
            )

    def test_reuser_plan_falls_back_to_torch_ref(self) -> None:
        """Plan with reuser must fall back to TorchReferenceBackend."""
        from unittest import mock
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        backend = NpuFlashAttentionBackend()
        plan = _make_multi_sample_plan()  # has a reuser

        q = torch.randn(13, 4, 64)
        k = torch.randn(16, 4, 64)
        v = torch.randn(16, 4, 64)

        called = [False]

        def mock_npu(*args, **kwargs):
            called[0] = True
            return (torch.zeros_like(args[0]), None, None, None, None, None, None)

        with mock.patch(
            "prefix_sharing.backends.flash_atten_npu._import_npu_fusion_attention",
            return_value=mock_npu,
        ):
            result = backend.attention(q, k, v, plan)
            assert result.shape == q.shape
            assert not called[0], (
                "NPU kernel must NOT be called for plan with reusers — "
                "should fall back to TorchReferenceBackend"
            )

    def test_apply_rope_delegates_to_torch_ref(self) -> None:
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
        planner = PrefixSharingPlanner(config)
        plan = planner.plan([[1, 2, 3, 4], [1, 2, 5, 6]])

        backend = NpuFlashAttentionBackend()
        q = torch.randn(6, 2, 32)
        k = torch.randn(8, 2, 32)

        q_out, k_out = backend.apply_rope(q, k, plan)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_build_kv_delegates_to_torch_ref(self) -> None:
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
        from prefix_sharing.backends.torch_ref import PrefixAttentionStore

        config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
        planner = PrefixSharingPlanner(config)
        plan = planner.plan([[1, 2, 3, 4], [1, 2, 5, 6]])

        backend = NpuFlashAttentionBackend()
        k = torch.randn(6, 2, 32)
        v = torch.randn(6, 2, 32)
        store = PrefixAttentionStore()

        k_out, v_out = backend.build_kv(k, v, store, plan, layer_id=0, tp_rank=0)
        expected_kv_len = sum(plan.expanded_lengths_kv)
        assert k_out.shape[0] == expected_kv_len
        assert v_out.shape[0] == expected_kv_len

class TestBug07GpuFlashAttentionReuserRouting:
    """[BUG-07] GPU flash_atten_gpu.py reuser detection and routing correctness.

    Verifies that is_reuser() logic works correctly (requires both
    provider_index != i AND prefix_lens[i] > 0).
    """

    def test_has_reuser_detection_for_provider_only_plan(self) -> None:
        config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
        planner = PrefixSharingPlanner(config)
        plan = planner.plan([[1, 2, 3, 4], [5, 6, 7, 8]])

        has_reuser = any(plan.is_reuser(i) for i in range(plan.batch_size))
        assert has_reuser is False, "plan with no sharing should not be detected as has_reuser"

    def test_has_reuser_detection_for_plan_with_reuser(self) -> None:
        config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
        planner = PrefixSharingPlanner(config)
        plan = planner.plan([[1, 2, 3, 4], [1, 2, 5, 6]])

        has_reuser = any(plan.is_reuser(i) for i in range(plan.batch_size))
        assert has_reuser is True, "plan with prefix sharing should be detected as has_reuser"

    def test_is_reuser_requires_both_conditions(self) -> None:
        """is_reuser() needs: provider_index != i AND prefix_lens[i] > 0."""
        plan = _make_multi_sample_plan()
        # sample 0: provider_index=0, prefix_len=0 → not reuser
        assert not plan.is_reuser(0)
        # sample 1: provider_index=0 (≠1), prefix_len=3 (>0) → reuser
        assert plan.is_reuser(1)
        # sample 2: provider_index=2, prefix_len=0 → not reuser
        assert not plan.is_reuser(2)

class TestBug08IntegrationHasTokenLengthCheck:
    """[BUG-08] integrations/utils.py must exist with ensure_global_packed_token_lengths.

    NPU version deleted this module and removed the check from integration code.
    Missing this check means silent token coordinate errors when SP-local shard
    inputs are used instead of global packed coordinates.
    """

    def test_utils_module_exists(self) -> None:
        try:
            utils = importlib.import_module("prefix_sharing.integrations.utils")
        except ModuleNotFoundError:
            pytest.fail("BUG-08: integrations/utils.py module does not exist")
        assert hasattr(utils, "ensure_global_packed_token_lengths"), (
            "BUG-08: utils module missing ensure_global_packed_token_lengths"
        )

    def test_megatron_runtime_has_token_length_validation(self) -> None:
        try:
            from prefix_sharing.integrations import megatron_runtime
            src = inspect.getsource(megatron_runtime)
        except (TypeError, OSError):
            pytest.skip("Cannot read megatron_runtime source")

        checks = [
            "ensure_global_packed_token_lengths",
            "total_padded_length",
            "query_length",
        ]
        found = [c for c in checks if c in src]
        assert len(found) >= 2, (
            "BUG-08: megatron_runtime.py lacks packed token length validation"
        )

# ============================================================================
# 🟢 Low-risk bugs — configuration and metadata integrity
# ============================================================================

class TestBug11BaseHasSupportsFlashAttention:
    """[BUG-11] BackendCapabilities must have supports_flash_attention field.

    NPU version deleted this field. It's used by capability queries.
    """

    def test_supports_flash_attention_flag_exists(self) -> None:
        from prefix_sharing.backends.base import BackendCapabilities

        fields = {f.name for f in dataclasses.fields(BackendCapabilities)}
        assert "supports_flash_attention" in fields, (
            "BUG-11: BackendCapabilities missing supports_flash_attention field"
        )

    def test_gpu_backend_reports_supports_flash_attention(self) -> None:
        from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend

        caps = GpuFlashAttentionBackend.capabilities
        assert hasattr(caps, "supports_flash_attention"), (
            "BackendCapabilities missing supports_flash_attention"
        )
        assert caps.supports_flash_attention is True, (
            "GPU backend should declare supports_flash_attention=True"
        )

class TestBug12ConfigValidation:
    """[BUG-12] Config validation for PP size and other constraints.

    NPU version simplified validation to only allow PP=1. The GPU version
    correctly validates PP >= 1 and rejects virtual PP.
    """

    def test_pp_size_1_passes_validation(self) -> None:
        config = PrefixSharingConfig(enable_prefix_sharing=True)
        model_config = SimpleNamespace(pipeline_model_parallel_size=1)
        config.validate(model_config=model_config)  # should not raise

    def test_pp_size_0_raises_error(self) -> None:
        from prefix_sharing.core.config import PrefixSharingConfigError

        config = PrefixSharingConfig(enable_prefix_sharing=True)
        model_config = SimpleNamespace(pipeline_model_parallel_size=0)

        with pytest.raises(PrefixSharingConfigError) as exc_info:
            config.validate(model_config=model_config)

        msg = str(exc_info.value)
        assert "pipeline" in msg.lower(), f"Error message should mention pipeline: {msg}"

    def test_virtual_pp_size_raises_error(self) -> None:
        from prefix_sharing.core.config import PrefixSharingConfigError

        config = PrefixSharingConfig(enable_prefix_sharing=True)
        model_config = SimpleNamespace(
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=2,
        )

        with pytest.raises(PrefixSharingConfigError) as exc_info:
            config.validate(model_config=model_config)

        msg = str(exc_info.value)
        assert "virtual" in msg.lower(), f"Error message should mention virtual PP: {msg}"

    def test_backend_validation_rejects_unknown_backend(self) -> None:
        from prefix_sharing.core.config import PrefixSharingConfigError

        config = PrefixSharingConfig(enable_prefix_sharing=True, backend="unknown_backend")

        with pytest.raises(PrefixSharingConfigError):
            config.validate()

class TestBackendFactoryNpuSupport:
    """Verify backend factory creates NPU backend correctly."""

    def test_factory_creates_npu_backend(self) -> None:
        from prefix_sharing.backends.factory import get_backend_instance
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_npu")
        backend = get_backend_instance(config)
        assert isinstance(backend, NpuFlashAttentionBackend)

    def test_npu_backend_caps_are_correct(self) -> None:
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        caps = NpuFlashAttentionBackend.capabilities
        assert caps.name == "flash_atten_npu"
        assert caps.supports_cann is True, "NPU backend must declare supports_cann=True"
        assert caps.supports_cuda is False, "NPU backend must not declare supports_cuda=True"
        assert caps.supports_cpu is False

# ============================================================================
# Comprehensive checklist — one-shot all-bug verification
# ============================================================================

class TestAllBugsChecklist:
    """One-shot verification of all known bugs.

    Run::

        python -m pytest ...test_npu_fixes.py::TestAllBugsChecklist -v
    """

    def test_checklist_bug01_packed_layout_has_unpad_repad(self) -> None:
        """[BUG-01] PackedBatchLayout has has_padding / unpad / repad"""
        layout = PackedBatchLayout.from_kept_position_rows(
            [torch.tensor([0, 1, 2]), torch.tensor([3, 4])],
            align_size=4,
        )
        assert hasattr(layout, "has_padding"), "BUG-01: missing has_padding"
        assert hasattr(layout, "unpad"), "BUG-01: missing unpad()"
        assert hasattr(layout, "repad"), "BUG-01: missing repad()"

    def test_checklist_bug02_flash_atten_base_has_tp_padding_methods(self) -> None:
        """[BUG-02] FlashAttentionMixin has _strip_tp_padding / _repad_output"""
        from prefix_sharing.backends.flash_atten_base import FlashAttentionMixin
        assert hasattr(FlashAttentionMixin, "_strip_tp_padding"), \
            "BUG-02: missing _strip_tp_padding()"
        assert hasattr(FlashAttentionMixin, "_repad_output"), \
            "BUG-02: missing _repad_output()"

    def test_checklist_bug03_actual_seq_len_not_using_cumulative(self) -> None:
        """[BUG-03] actual_seq_qlen/kvlen must not use cu_seqlens[1:]"""
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
        lines = _read_source_lines(NpuFlashAttentionBackend, "attention")
        bug_found = any(
            "cu_seqlens_q[1:]" in line or "cu_seqlens_kv[1:]" in line
            for line in lines
        )
        assert not bug_found, "BUG-03: flash_atten_npu.py still uses cu_seqlens[1:]"

    def test_checklist_bug04_npu_attention_has_packed_batch_layout_param(self) -> None:
        """[BUG-04] NpuFlashAttentionBackend.attention() has packed_batch_layout"""
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
        sig = inspect.signature(NpuFlashAttentionBackend.attention)
        assert "packed_batch_layout" in sig.parameters, \
            "BUG-04: attention() missing packed_batch_layout parameter"

    def test_checklist_bug05_block_causal_mask_importable(self) -> None:
        """[BUG-05] block_causal_mask module imports correctly"""
        try:
            from prefix_sharing.backends.block_causal_mask import (  # noqa: F401
                build_block_causal_mask,
                mask_to_te_bias,
            )
        except ImportError as e:
            pytest.fail(f"BUG-05: block_causal_mask import failed: {e}")

    def test_checklist_bug06_npu_backend_instantiable(self) -> None:
        """[BUG-06] NpuFlashAttentionBackend instantiates correctly"""
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend
        backend = NpuFlashAttentionBackend()
        assert backend is not None

    def test_checklist_bug07_gpu_attention_has_packed_batch_layout_param(self) -> None:
        """[BUG-07] GpuFlashAttentionBackend.attention() has packed_batch_layout"""
        from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
        sig = inspect.signature(GpuFlashAttentionBackend.attention)
        assert "packed_batch_layout" in sig.parameters, \
            "BUG-07: GPU attention() missing packed_batch_layout parameter"

    def test_checklist_bug08_utils_module_exists(self) -> None:
        """[BUG-08] integrations/utils.py exists with ensure_global_packed_token_lengths"""
        try:
            utils = importlib.import_module("prefix_sharing.integrations.utils")
        except ModuleNotFoundError:
            pytest.fail("BUG-08: integrations/utils.py module does not exist")
        assert hasattr(utils, "ensure_global_packed_token_lengths"), \
            "BUG-08: utils module missing ensure_global_packed_token_lengths"

    def test_checklist_bug11_supports_flash_attention_field_exists(self) -> None:
        """[BUG-11] BackendCapabilities has supports_flash_attention field"""
        from prefix_sharing.backends.base import BackendCapabilities
        fields = {f.name for f in dataclasses.fields(BackendCapabilities)}
        assert "supports_flash_attention" in fields, \
            "BUG-11: BackendCapabilities missing supports_flash_attention"

# ============================================================================
# NPU hardware validation (requires NPU_HARDWARE_TEST=1)
# ============================================================================

@pytest.mark.skipif(
    os.environ.get("NPU_HARDWARE_TEST") != "1",
    reason="Set NPU_HARDWARE_TEST=1 to run on NPU hardware",
)
class TestNpuHardwareValidation:
    """End-to-end validation on actual NPU hardware.

    Run::

        NPU_HARDWARE_TEST=1 python -m pytest ...test_npu_fixes.py::TestNpuHardwareValidation -v

    Prerequisites:
      - torch_npu available
      - MindSpeed installed (mindspeed.ops.npu_fusion_attention available)
    """

    def test_npu_is_available(self) -> None:
        try:
            import torch_npu  # noqa: F401
            assert torch.npu.is_available(), "NPU not available"
        except ImportError:
            pytest.fail("torch_npu not installed, cannot run NPU hardware tests")

    def test_mindspeed_importable(self) -> None:
        """Verify MindSpeed is importable.

        Bug risk: If MindSpeed is not installed or version mismatched,
        NpuFlashAttentionBackend crashes on first attention() call.
        """
        try:
            from mindspeed.ops.fusion_attention_v2 import npu_fusion_attention  # noqa: F401
        except ModuleNotFoundError as e:
            pytest.fail(f"MindSpeed import failed: {e}")

    def test_npu_attention_no_sharing_basic(self) -> None:
        """Basic forward: no prefix sharing."""
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
        assert result.shape == (total_q, 2, 32)
        assert not torch.isnan(result).any(), "Output contains NaN"
        assert not torch.isinf(result).any(), "Output contains Inf"

    def test_npu_attention_with_reuser(self) -> None:
        """Forward with prefix sharing (provider + reuser)."""
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
        assert result.shape == (total_q, 2, 32)
        assert not torch.isnan(result).any(), "Output contains NaN"
        assert not torch.isinf(result).any(), "Output contains Inf"

    def test_npu_attention_with_tp_padding(self) -> None:
        """Forward with TP padding (simulated)."""
        from prefix_sharing.backends.flash_atten_npu import NpuFlashAttentionBackend

        config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2)
        planner = PrefixSharingPlanner(config)
        plan = planner.plan([[1, 2, 3, 4], [1, 2, 5, 6]])

        backend = NpuFlashAttentionBackend()

        layout = PackedBatchLayout.from_kept_position_rows(
            [torch.tensor([0, 1, 2, 3]), torch.tensor([2, 3])],
            align_size=4,
        )

        total_q = layout.total_padded_length
        total_kv = plan.cu_seqlens_kv[-1]

        q = torch.randn(total_q, 2, 32, device="npu")
        k = torch.randn(total_kv, 2, 32, device="npu")
        v = torch.randn(total_kv, 2, 32, device="npu")

        result = backend.attention(q, k, v, plan, packed_batch_layout=layout)
        assert result.shape == (total_q, 2, 32)
        assert not torch.isnan(result).any(), "Output contains NaN"
        assert not torch.isinf(result).any(), "Output contains Inf"

