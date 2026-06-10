"""Unit tests for block-causal mask construction.

These tests verify the *semantic* correctness of the mask used by the
GPU/NPU Flash Attention backends. They are deliberately explicit (asserting
exact True/False at known coordinates) rather than relying on numerical
alignment between backends, because numerical atol can hide the segment-relative
vs absolute-position bug that motivated this module.

Conventions in the returned mask:
    True  = masked (invisible)
    False = visible
"""

from __future__ import annotations

import pytest

from prefix_sharing.backends.block_causal_mask import (
    build_block_causal_mask,
    mask_to_te_bias,
)
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner

torch = pytest.importorskip("torch")

def _make_plan(batch_sizes: list[int], prefix_lens: list[int]):
    """Build a PrefixSharingPlan by feeding sequences with a forced prefix.

    Sequences are constructed so that the trie detector naturally produces the
    requested ``prefix_lens`` pattern: the i-th sequence shares its first
    ``prefix_lens[i]`` tokens with the previous provider.
    """
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1)
    planner = PrefixSharingPlanner(config)

    # Build sequences: provider sequences are unique; reusers repeat the
    # provider's prefix then diverge. We pick providers as the first sample
    # of each sharing group.
    sequences: list[list[int]] = []
    next_token = 100  # disambiguate from prefix tokens
    group_provider_tokens: dict[int, list[int]] = {}
    for i, (size, p) in enumerate(zip(batch_sizes, prefix_lens)):
        if p == 0:
            # provider: fresh tokens
            seq = list(range(next_token, next_token + size))
            next_token += size
            sequences.append(seq)
            group_provider_tokens[i] = seq
        else:
            # find most recent provider in the input
            provider_idx = max(
                (j for j, s in enumerate(prefix_lens[:i]) if s == 0),
                default=None,
            )
            assert provider_idx is not None, "reuser without preceding provider"
            provider_seq = group_provider_tokens[provider_idx]
            assert p <= len(provider_seq), "prefix longer than provider sequence"
            suffix = list(range(next_token, next_token + size - p))
            next_token += size - p
            sequences.append(provider_seq[:p] + suffix)

    plan = planner.plan(sequences)
    # Sanity-check the structural fields.
    assert plan.batch_size == len(batch_sizes)
    return plan

# ---------------------------------------------------------------------
# Provider-only plans: should match standard causal mask
# ---------------------------------------------------------------------

def test_provider_only_mask_is_standard_causal():
    plan = _make_plan(batch_sizes=[4, 3], prefix_lens=[0, 0])
    mask = build_block_causal_mask(plan, device=torch.device("cpu"))

    # Cross-sample blocks must be entirely masked.
    assert mask[:4, 4:].all(), "cross-sample block must be masked"
    assert mask[4:, :4].all(), "cross-sample block must be masked"

    # Within sample 0 (rows 0..3, cols 0..3): lower-triangular visible.
    block0 = ~mask[:4, :4]  # visible
    expected0 = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(block0, expected0)

    # Within sample 1 (rows 4..6, cols 4..6): lower-triangular visible.
    block1 = ~mask[4:, 4:]
    expected1 = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    assert torch.equal(block1, expected1)

# ---------------------------------------------------------------------
# Reuser plans: prefix columns all-visible, suffix columns causal
# ---------------------------------------------------------------------

def test_reuser_q0_sees_all_prefix_plus_suffix0():
    """The defining correctness property.

    Reuser with prefix_len=P, suffix_len=L. Q[0] is the first suffix token.
    It must attend to: prefix[0..P-1] and suffix[0]; not suffix[1..].
    """
    # provider: len=6, reuser: prefix=3 + suffix=2 = total 5 (kv=6 from provider)
    plan = _make_plan(batch_sizes=[6, 5], prefix_lens=[0, 3])
    mask = build_block_causal_mask(plan, device=torch.device("cpu"))

    # Sample 1 is the reuser; sample 0 is the provider.
    # q_range_for_batch(1) = (kept_lengths_q[0], kept_lengths_q[0] + kept_lengths_q[1])
    q_lo, q_hi = plan.q_range_for_batch(1)
    kv_lo, kv_hi = plan.kv_range_for_batch(1)
    p = plan.prefix_lens[1]
    q_len = q_hi - q_lo
    kv_len = kv_hi - kv_lo
    assert q_len == 2, f"reuser q_len should be 2, got {q_len}"
    assert kv_len == 5, f"reuser kv_len should be 5, got {kv_len}"
    assert p == 3

    reuser_block = mask[q_lo:q_hi, kv_lo:kv_hi]
    # Q[0] (first row): prefix columns (0..p-1) all visible; suffix col 0 visible; rest masked.
    q0_row_visible = ~reuser_block[0]
    expected_visible = torch.tensor([True, True, True, True, False])
    assert torch.equal(q0_row_visible, expected_visible), (
        f"reuser Q[0] visibility = {q0_row_visible.tolist()}, "
        f"expected {expected_visible.tolist()}"
    )

def test_reuser_q_last_sees_everything_in_its_sample():
    plan = _make_plan(batch_sizes=[5, 4], prefix_lens=[0, 2])
    mask = build_block_causal_mask(plan, device=torch.device("cpu"))
    q_lo, q_hi = plan.q_range_for_batch(1)
    kv_lo, kv_hi = plan.kv_range_for_batch(1)
    reuser_block = mask[q_lo:q_hi, kv_lo:kv_hi]
    # Last Q row should see all KV columns of the sample.
    last_row_visible = ~reuser_block[-1]
    assert last_row_visible.all(), f"reuser Q[-1] should see everything, got {last_row_visible}"

def test_reuser_suffix_columns_are_causal():
    """Within the suffix sub-block, mask is lower-triangular."""
    plan = _make_plan(batch_sizes=[6, 6], prefix_lens=[0, 3])
    mask = build_block_causal_mask(plan, device=torch.device("cpu"))
    q_lo, q_hi = plan.q_range_for_batch(1)
    kv_lo, kv_hi = plan.kv_range_for_batch(1)
    p = plan.prefix_lens[1]
    q_len = q_hi - q_lo
    suffix_len = kv_hi - kv_lo - p
    suffix_block = mask[q_lo:q_hi, kv_lo + p:kv_hi]
    expected = ~torch.tril(torch.ones(q_len, suffix_len, dtype=torch.bool))
    assert torch.equal(suffix_block, expected)

# ---------------------------------------------------------------------
# Cross-sample invariant
# ---------------------------------------------------------------------

def test_cross_sample_attention_always_masked():
    plan = _make_plan(batch_sizes=[3, 4, 2], prefix_lens=[0, 2, 0])
    mask = build_block_causal_mask(plan, device=torch.device("cpu"))
    # Build the expected block-diagonal structure: every off-diagonal block masked.
    cu_q = plan.cu_seqlens_q
    cu_kv = plan.cu_seqlens_kv
    for i in range(plan.batch_size):
        for j in range(plan.batch_size):
            if i == j:
                continue
            block = mask[cu_q[i]:cu_q[i + 1], cu_kv[j]:cu_kv[j + 1]]
            assert block.all(), (
                f"cross-sample block (rows sample {i}, cols sample {j}) "
                "must be entirely masked"
            )

# ---------------------------------------------------------------------
# mask_to_te_bias
# ---------------------------------------------------------------------

def test_mask_to_te_bias_yields_zero_and_minus_inf():
    mask = torch.tensor([[False, True], [True, True]])
    bias = mask_to_te_bias(mask, dtype=torch.float32)
    assert bias[0, 0].item() == 0.0
    assert torch.isinf(bias[0, 1]) and bias[0, 1].item() < 0
    assert torch.isinf(bias[1, 0]) and bias[1, 0].item() < 0

def test_mask_to_te_bias_preserves_dtype_and_device():
    mask = torch.zeros(3, 3, dtype=torch.bool)
    bias = mask_to_te_bias(mask, dtype=torch.float16)
    assert bias.dtype == torch.float16
    assert bias.device == mask.device

# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------

def test_single_sample_provider():
    """Single sample with no sharing — just standard causal."""
    plan = _make_plan(batch_sizes=[5], prefix_lens=[0])
    mask = build_block_causal_mask(plan, device=torch.device("cpu"))
    visible = ~mask
    expected = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    assert torch.equal(visible, expected)

def test_all_providers_no_sharing():
    """Multiple providers with no actual sharing happening."""
    plan = _make_plan(batch_sizes=[3, 5, 4], prefix_lens=[0, 0, 0])
    mask = build_block_causal_mask(plan, device=torch.device("cpu"))
    # Each block should be standard causal, cross-blocks masked
    cu_q = plan.cu_seqlens_q
    cu_kv = plan.cu_seqlens_kv
    for i in range(plan.batch_size):
        block = mask[cu_q[i]:cu_q[i + 1], cu_kv[i]:cu_kv[i + 1]]
        visible = ~block
        q_len = cu_q[i + 1] - cu_q[i]
        kv_len = cu_kv[i + 1] - cu_kv[i]
        assert q_len == kv_len, "provider should have equal Q and KV lengths"
        expected = torch.tril(torch.ones(q_len, kv_len, dtype=torch.bool))
        assert torch.equal(visible, expected), f"sample {i} block wrong"

def test_reuser_with_full_prefix():
    """Reuser whose entire sequence is a prefix (kept_lengths_q = 0)."""
    # This shouldn't happen normally, but test that empty Q slices are handled.
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1)
    planner = PrefixSharingPlanner(config)
    # Two identical sequences — second is pure reuser with full prefix
    plan = planner.plan([[1, 2, 3], [1, 2, 3]])
    mask = build_block_causal_mask(plan, device=torch.device("cpu"))
    # Mask should be constructable without errors
    assert mask.shape[0] == plan.cu_seqlens_q[-1]
    assert mask.shape[1] == plan.cu_seqlens_kv[-1]
