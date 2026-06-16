"""Tests for ThdBatchLayout validation, unpad/repad, and edge cases.

Focuses on the core business logic paths that were uncovered: __post_init__
validation guards, unpad/repad for TP padding, has_padding, valid_tokens,
and construct_from_kept_position_ids edge cases.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.batch_layout import ThdBatchLayout


# ------------------------------------------------------------------
# __post_init__ validation guards
# ------------------------------------------------------------------


def test_post_init_rejects_padded_lengths_mismatch():
    with pytest.raises(ValueError, match="padded_lengths must match"):
        ThdBatchLayout(
            valid_lengths=[4, 3],
            padded_lengths=[4],  # length mismatch
            cu_seqlens=[0, 4],
            max_seqlen=4,
        )


def test_post_init_rejects_cu_seqlens_wrong_length():
    with pytest.raises(ValueError, match="cu_seqlens length"):
        ThdBatchLayout(
            valid_lengths=[4, 3],
            padded_lengths=[4, 3],
            cu_seqlens=[0, 4, 7, 99],  # should be 3, not 4
            max_seqlen=4,
        )


def test_post_init_rejects_cu_seqlens_not_starting_with_zero():
    with pytest.raises(ValueError, match="must start with 0"):
        ThdBatchLayout(
            valid_lengths=[4, 3],
            padded_lengths=[4, 3],
            cu_seqlens=[1, 4, 7],  # starts with 1 instead of 0
            max_seqlen=4,
        )


def test_post_init_rejects_negative_valid_lengths():
    with pytest.raises(ValueError, match="valid_lengths must be non-negative"):
        ThdBatchLayout(
            valid_lengths=[-1, 3],
            padded_lengths=[4, 3],
            cu_seqlens=[0, 4, 7],
            max_seqlen=4,
        )


def test_post_init_rejects_negative_padded_lengths():
    with pytest.raises(ValueError, match="padded_lengths must be non-negative"):
        ThdBatchLayout(
            valid_lengths=[4, 3],
            padded_lengths=[4, -1],
            cu_seqlens=[0, 4, 3],
            max_seqlen=4,
        )


def test_post_init_rejects_padded_smaller_than_valid():
    with pytest.raises(ValueError, match="padded_lengths cannot be smaller"):
        ThdBatchLayout(
            valid_lengths=[5, 3],
            padded_lengths=[4, 3],  # 4 < 5
            cu_seqlens=[0, 4, 7],
            max_seqlen=4,
        )


def test_post_init_rejects_cu_seqlens_not_cumsum_of_padded():
    with pytest.raises(ValueError, match="cumulative padded lengths"):
        ThdBatchLayout(
            valid_lengths=[4, 3],
            padded_lengths=[4, 3],
            cu_seqlens=[0, 3, 7],  # cumsum([4,3])=[0,4,7], but [0,3,7] wrong
            max_seqlen=4,
        )


# ------------------------------------------------------------------
# construct_from_kept_position_ids edge cases
# ------------------------------------------------------------------


def test_construct_from_kept_position_ids_empty_returns_empty_layout():
    layout = ThdBatchLayout.construct_from_kept_position_ids([], align_size=4)
    assert layout.valid_lengths == []
    assert layout.padded_lengths == []
    assert layout.cu_seqlens == [0]
    assert layout.total_padded_length == 0


def test_construct_from_kept_position_ids_align_size_zero_raises():
    with pytest.raises(ValueError, match="align_size must be >= 1"):
        ThdBatchLayout.construct_from_kept_position_ids(
            [torch.tensor([1, 2, 3])], align_size=0,
        )


# ------------------------------------------------------------------
# has_padding property
# ------------------------------------------------------------------


def test_has_padding_false_when_no_padding():
    layout = ThdBatchLayout.construct_from_valid_lengths([4, 3, 5])
    assert not layout.has_padding


def test_has_padding_true_when_padding_exists():
    rows = [torch.zeros(5, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    layout = ThdBatchLayout.construct_from_kept_position_ids(rows, align_size=4)
    # 5 padded to 8, 3 padded to 4 → has_padding=True
    assert layout.has_padding


def test_has_padding_true_when_only_some_rows_have_padding():
    rows = [torch.zeros(4, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    layout = ThdBatchLayout.construct_from_kept_position_ids(rows, align_size=4)
    # 4 → 4 (no pad), 3 → 4 (has pad) → overall has_padding=True
    assert layout.has_padding


# ------------------------------------------------------------------
# valid_tokens
# ------------------------------------------------------------------


def test_valid_tokens_returns_valid_tokens():
    layout = ThdBatchLayout.construct_from_valid_lengths([5, 3])
    tensor = torch.arange(8)  # [0,1,2,3,4,5,6,7]
    # Row 0: valid slice should be indices 0..4 → [0,1,2,3,4]
    assert torch.equal(layout.valid_tokens(tensor, 0), tensor[:5])
    # Row 1: valid slice should be indices 5..7 → [5,6,7]
    assert torch.equal(layout.valid_tokens(tensor, 1), tensor[5:8])


def test_valid_tokens_with_padding_layout():
    rows = [torch.zeros(5, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    layout = ThdBatchLayout.construct_from_kept_position_ids(rows, align_size=4)
    # padded_lengths: [8, 4]; valid_lengths: [5, 3]; cu_seqlens: [0, 8, 12]
    tensor = torch.arange(12)
    # Row 0 valid: indices 0..4
    assert torch.equal(layout.valid_tokens(tensor, 0), tensor[:5])
    # Row 1 valid: indices 8..10
    assert torch.equal(layout.valid_tokens(tensor, 1), tensor[8:11])


# ------------------------------------------------------------------
# unpad / repad — core TP padding operations
# ------------------------------------------------------------------


def test_unpad_no_padding_is_noop():
    layout = ThdBatchLayout.construct_from_valid_lengths([5, 3])
    tensor = torch.randn(8, 2, 8)
    result = layout.unpad(tensor)
    assert torch.equal(result, tensor)


def test_unpad_strips_padding_correctly():
    rows = [torch.zeros(5, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    layout = ThdBatchLayout.construct_from_kept_position_ids(rows, align_size=4)
    # padded_lengths: [8, 4]; valid_lengths: [5, 3]

    tensor = torch.arange(12).float()
    # Packed: [0,1,2,3,4,0,0,0, 5,6,7,0] (conceptually)
    tensor = torch.tensor([0., 1., 2., 3., 4., 0., 0., 0., 5., 6., 7., 0.])

    unpadded = layout.unpad(tensor)
    # Should keep only valid tokens: [0,1,2,3,4, 5,6,7]
    expected = torch.tensor([0., 1., 2., 3., 4., 5., 6., 7.])
    assert torch.equal(unpadded, expected)


def test_unpad_with_3d_tensor():
    """unpad on 3D (THD) tensor preserves head/dim dimensions."""
    rows = [torch.zeros(5, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    layout = ThdBatchLayout.construct_from_kept_position_ids(rows, align_size=4)

    # Create 3D tensor: (total_padded=12, num_heads=2, head_dim=4)
    tensor = torch.randn(12, 2, 4)
    unpadded = layout.unpad(tensor)

    # Total valid = 5 + 3 = 8
    assert unpadded.shape == (8, 2, 4)
    # Valid portions should match original
    assert torch.equal(unpadded[:5], tensor[:5])  # row 0 valid
    assert torch.equal(unpadded[5:8], tensor[8:11])  # row 1 valid


def test_repad_no_padding_is_noop():
    layout = ThdBatchLayout.construct_from_valid_lengths([5, 3])
    tensor = torch.randn(8, 2, 8)
    result = layout.repad(tensor)
    assert torch.equal(result, tensor)


def test_repad_reapplies_padding_correctly():
    rows = [torch.zeros(5, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    layout = ThdBatchLayout.construct_from_kept_position_ids(rows, align_size=4)
    # padded_lengths: [8, 4]; valid_lengths: [5, 3]

    # Input: valid-only tensor of shape (8,)
    valid_tensor = torch.tensor([0., 1., 2., 3., 4., 5., 6., 7.])
    repadded = layout.repad(valid_tensor)

    # Should produce padded shape (12,)
    assert repadded.shape[0] == 12
    # Valid positions should match
    assert torch.equal(repadded[:5], valid_tensor[:5])
    assert torch.equal(repadded[8:11], valid_tensor[5:8])
    # Padding positions should be zero
    assert torch.all(repadded[5:8] == 0)
    assert torch.all(repadded[11:] == 0)


def test_repad_with_3d_tensor():
    """repad on 3D tensor preserves head/dim and fills padding with zeros."""
    rows = [torch.zeros(5, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    layout = ThdBatchLayout.construct_from_kept_position_ids(rows, align_size=4)

    # Input: valid-only 3D tensor (total_valid=8, num_heads=2, head_dim=4)
    valid_tensor = torch.randn(8, 2, 4)
    repadded = layout.repad(valid_tensor)

    assert repadded.shape == (12, 2, 4)
    # Valid portions match
    assert torch.equal(repadded[:5], valid_tensor[:5])
    assert torch.equal(repadded[8:11], valid_tensor[5:8])
    # Padding positions are zero
    assert torch.all(repadded[5:8] == 0)
    assert torch.all(repadded[11:] == 0)


def test_unpad_repad_roundtrip():
    """unpad then repad should reconstruct the original padded tensor."""
    rows = [torch.zeros(5, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    layout = ThdBatchLayout.construct_from_kept_position_ids(rows, align_size=4)

    # Start with a padded tensor
    original = torch.randn(12, 2, 4)

    # Unpad to get valid-only
    unpadded = layout.unpad(original)

    # Repad to reconstruct
    repadded = layout.repad(unpadded)

    # Valid positions should match original
    assert torch.equal(repadded[:5], original[:5])
    assert torch.equal(repadded[8:11], original[8:11])
    # Padding positions should be zero (original might not be zero in padding)
    assert torch.all(repadded[5:8] == 0)
    assert torch.all(repadded[11:] == 0)


def test_repad_row_without_padding_is_passed_through():
    """When only some rows have padding, rows without padding pass through unchanged."""
    rows = [torch.zeros(4, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    layout = ThdBatchLayout.construct_from_kept_position_ids(rows, align_size=4)
    # padded_lengths: [4, 4]; valid_lengths: [4, 3]
    # Row 0 has no padding (4==4), row 1 has padding (4>3)

    valid_tensor = torch.randn(7, 2, 4)  # 4 + 3 = 7 total valid
    repadded = layout.repad(valid_tensor)

    # Row 0 should pass through unchanged (no padding)
    assert torch.equal(repadded[:4], valid_tensor[:4])
    # Row 1 should have zero-padded last position
    assert torch.equal(repadded[4:7], valid_tensor[4:7])
    assert torch.all(repadded[7:8] == 0)


# ------------------------------------------------------------------
# total_valid_length property
# ------------------------------------------------------------------


def test_total_valid_length():
    layout = ThdBatchLayout.construct_from_valid_lengths([5, 3, 7])
    assert layout.total_valid_length == 15


def test_total_valid_length_with_padding():
    rows = [torch.zeros(5, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    layout = ThdBatchLayout.construct_from_kept_position_ids(rows, align_size=4)
    # valid_lengths: [5, 3]
    assert layout.total_valid_length == 8
    # total_padded_length should be different
    assert layout.total_padded_length == 12
