import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.batch_layout import BshdBatchLayout, BshdTokenIndex, ThdBatchLayout


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
def test_thd_batch_layout_aligns_valid_rows_for_common_tensor_parallel_sizes(
    tp_size,
    expected_padded_lengths,
    expected_cu_seqlens,
    expected_positions,
    expected_mask,
):
    layout = ThdBatchLayout.from_kept_position_rows(
        [
            torch.tensor([0, 1, 2, 3, 4]),
            torch.tensor([3, 4]),
        ],
        align_size=tp_size,
    )

    assert layout.layout_kind == "thd"
    assert layout.valid_lengths == [5, 2]
    assert layout.padded_lengths == expected_padded_lengths
    assert layout.cu_seqlens == expected_cu_seqlens
    assert layout.max_seqlen == max(expected_padded_lengths)
    assert layout.total_valid_length == 7
    assert layout.total_padded_length == expected_cu_seqlens[-1]
    assert layout.position_ids.tolist() == expected_positions
    assert layout.valid_token_mask.tolist() == expected_mask
    assert layout.token_index(1, 0) == expected_cu_seqlens[1]


def test_thd_batch_layout_rejects_padding_slot_as_valid_index():
    layout = ThdBatchLayout.from_kept_position_rows(
        [torch.tensor([0, 1, 2])],
        align_size=2,
    )

    with pytest.raises(IndexError):
        layout.token_index(0, 3)


def test_bshd_batch_layout_reads_and_scatters_dense_valid_tokens():
    mask = torch.tensor(
        [
            [True, True, True, True, True, False],
            [False, False, False, True, True, False],
        ]
    )
    position_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0],
            [0, 1, 2, 3, 4, 0],
        ]
    )
    layout = BshdBatchLayout.from_valid_token_mask(mask, position_ids=position_ids)
    tensor = torch.arange(2 * 6 * 2, dtype=torch.float32).reshape(2, 6, 2)
    output = torch.zeros_like(tensor)

    assert layout.layout_kind == "bshd"
    assert layout.valid_lengths == [5, 2]
    assert layout.max_seqlen == 6
    assert layout.total_valid_length == 7
    assert layout.token_index(1, 0) == BshdTokenIndex(row=1, seq_pos=3)
    assert layout.valid_row(tensor, 1).tolist() == tensor[1, 3:5].tolist()

    layout.scatter_valid_row(output, 1, torch.ones(2, 2))
    assert output[1, 3:5].tolist() == torch.ones(2, 2).tolist()
    assert output[1, :3].abs().sum().item() == 0
    assert output[1, 5].abs().sum().item() == 0
