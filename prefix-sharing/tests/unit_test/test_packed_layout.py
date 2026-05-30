import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.packed_layout import PackedBatchLayout


@pytest.mark.parametrize(
    ("tp_size", "expected_padded_lengths", "expected_cu_seqlens", "expected_positions", "expected_mask"),
    [
        (
            2,
            [6, 2],
            [0, 6, 8],
            [0, 1, 2, 3, 4, 0, 3, 4],
            [True, True, True, True, True, False, True, True],
        ),
        (
            4,
            [8, 4],
            [0, 8, 12],
            [0, 1, 2, 3, 4, 0, 0, 0, 3, 4, 0, 0],
            [True, True, True, True, True, False, False, False, True, True, False, False],
        ),
        (
            8,
            [8, 8],
            [0, 8, 16],
            [0, 1, 2, 3, 4, 0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0],
            [True, True, True, True, True, False, False, False, True, True, False, False, False, False, False, False],
        ),
    ],
)
def test_packed_batch_layout_aligns_valid_rows_for_common_tensor_parallel_sizes(
    tp_size,
    expected_padded_lengths,
    expected_cu_seqlens,
    expected_positions,
    expected_mask,
):
    layout = PackedBatchLayout.from_kept_position_rows(
        [
            torch.tensor([0, 1, 2, 3, 4]),
            torch.tensor([3, 4]),
        ],
        align_size=tp_size,
    )

    assert layout.valid_lengths == [5, 2]
    assert layout.padded_lengths == expected_padded_lengths
    assert layout.cu_seqlens == expected_cu_seqlens
    assert layout.max_seqlen == max(expected_padded_lengths)
    assert layout.total_valid_length == 7
    assert layout.total_padded_length == expected_cu_seqlens[-1]
    assert layout.packed_position_ids.tolist() == expected_positions
    assert layout.valid_token_mask.tolist() == expected_mask
    assert layout.packed_index(1, 0) == expected_cu_seqlens[1]


def test_packed_batch_layout_rejects_padding_slot_as_valid_index():
    layout = PackedBatchLayout.from_kept_position_rows(
        [torch.tensor([0, 1, 2])],
        align_size=2,
    )

    with pytest.raises(IndexError):
        layout.packed_index(0, 3)
