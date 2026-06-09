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
def test_thd_batch_layout_aligns_valid_tokens_for_common_tensor_parallel_sizes(
    tp_size,
    expected_padded_lengths,
    expected_cu_seqlens,
    expected_positions,
    expected_mask,
):
    layout = ThdBatchLayout.construct_from_kept_position_ids(
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
    layout = ThdBatchLayout.construct_from_kept_position_ids(
        [torch.tensor([0, 1, 2])],
        align_size=2,
    )

    with pytest.raises(IndexError):
        layout.token_index(0, 3)


def test_thd_construct_from_kept_position_ids_returns_empty_layout_for_empty_input():
    layout = ThdBatchLayout.construct_from_kept_position_ids([], align_size=4)

    assert layout.valid_lengths == []
    assert layout.padded_lengths == []
    assert layout.cu_seqlens == [0]
    assert layout.max_seqlen == 0
    assert layout.position_ids is None
    assert layout.valid_token_mask is None
    assert layout.batch_size == 0


def test_thd_construct_from_kept_position_ids_with_align_size_1_produces_no_padding():
    layout = ThdBatchLayout.construct_from_kept_position_ids(
        [torch.tensor([5, 6, 7]), torch.tensor([8])],
        align_size=1,
    )

    assert layout.valid_lengths == [3, 1]
    assert layout.padded_lengths == [3, 1]
    assert layout.cu_seqlens == [0, 3, 4]
    assert layout.max_seqlen == 3
    assert layout.position_ids.tolist() == [5, 6, 7, 8]
    assert layout.valid_token_mask.tolist() == [True, True, True, True]


def test_thd_construct_from_kept_position_ids_when_valid_length_already_aligned():
    # valid_length=4, align_size=4 → pad_length=0, no cat inside loop
    layout = ThdBatchLayout.construct_from_kept_position_ids(
        [torch.tensor([0, 1, 2, 3])],
        align_size=4,
    )

    assert layout.valid_lengths == [4]
    assert layout.padded_lengths == [4]
    assert layout.cu_seqlens == [0, 4]
    assert layout.position_ids.tolist() == [0, 1, 2, 3]
    assert layout.valid_token_mask.tolist() == [True, True, True, True]


def test_thd_construct_from_kept_position_ids_single_row():
    layout = ThdBatchLayout.construct_from_kept_position_ids(
        [torch.tensor([10, 11, 12])],
        align_size=8,
    )

    assert layout.valid_lengths == [3]
    assert layout.padded_lengths == [8]
    assert layout.cu_seqlens == [0, 8]
    assert layout.max_seqlen == 8
    assert layout.position_ids.tolist() == [10, 11, 12, 0, 0, 0, 0, 0]
    assert layout.valid_token_mask.tolist() == [True, True, True, False, False, False, False, False]


def test_thd_construct_from_kept_position_ids_uniform_lengths():
    rows = [torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]), torch.tensor([6, 7, 8])]
    layout = ThdBatchLayout.construct_from_kept_position_ids(rows, align_size=4)

    assert layout.valid_lengths == [3, 3, 3]
    assert layout.padded_lengths == [4, 4, 4]
    assert layout.cu_seqlens == [0, 4, 8, 12]
    assert layout.max_seqlen == 4
    assert layout.position_ids.tolist() == [0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0]
    assert layout.valid_token_mask.tolist() == [True, True, True, False, True, True, True, False, True, True, True, False]


def test_thd_construct_from_kept_position_ids_padding_region_is_zero():
    layout = ThdBatchLayout.construct_from_kept_position_ids(
        [torch.tensor([100, 200]), torch.tensor([300])],
        align_size=4,
    )

    # padding slots in position_ids must be 0
    padded_positions = layout.position_ids.tolist()
    mask = layout.valid_token_mask.tolist()
    for i, is_valid in enumerate(mask):
        if not is_valid:
            assert padded_positions[i] == 0

    # valid slots preserve original values
    assert layout.valid_tokens(layout.position_ids, 0).tolist() == [100, 200]
    assert layout.valid_tokens(layout.position_ids, 1).tolist() == [300]


def test_thd_construct_from_kept_position_ids_rejects_align_size_below_1():
    with pytest.raises(ValueError, match="align_size must be >= 1"):
        ThdBatchLayout.construct_from_kept_position_ids(
            [torch.tensor([0, 1])],
            align_size=0,
        )


def test_thd_construct_from_kept_position_ids_float_dtype():
    layout = ThdBatchLayout.construct_from_kept_position_ids(
        [torch.tensor([1.5, 2.5, 3.5])],
        align_size=4,
    )

    assert layout.position_ids.dtype == torch.float32
    assert layout.valid_tokens(layout.position_ids, 0).tolist() == [1.5, 2.5, 3.5]
    # padding region must be 0.0 for float dtype
    padded = layout.padded_tokens(layout.position_ids, 0).tolist()
    assert padded == [1.5, 2.5, 3.5, 0.0]


def test_thd_construct_from_kept_position_ids_long_dtype():
    layout = ThdBatchLayout.construct_from_kept_position_ids(
        [torch.tensor([10, 20], dtype=torch.int64)],
        align_size=4,
    )

    assert layout.position_ids.dtype == torch.int64
    assert layout.position_ids.tolist() == [10, 20, 0, 0]


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
    assert layout.token_index(1, 0) == BshdTokenIndex(seq_idx_in_batch=1, token_idx_in_seq=3)
    assert layout.valid_tokens(tensor, 1).tolist() == tensor[1, 3:5].tolist()

    layout.write_valid_tokens(output, 1, torch.ones(2, 2))
    assert output[1, 3:5].tolist() == torch.ones(2, 2).tolist()
    assert output[1, :3].abs().sum().item() == 0
    assert output[1, 5].abs().sum().item() == 0


def test_bshd_batch_layout_reads_and_scatters_compact_valid_tokens():
    mask = torch.tensor(
        [
            [True, True, True, False],
            [False, True, True, False],
        ]
    )
    layout = BshdBatchLayout.from_valid_token_mask(mask, position_ids=torch.arange(4).repeat(2, 1))
    compact = torch.arange(layout.total_valid_length * 2, dtype=torch.float32).reshape(layout.total_valid_length, 2)
    output = torch.zeros_like(compact)

    assert layout.valid_lengths == [3, 2]
    assert layout.valid_tokens(compact, 0).tolist() == compact[:3].tolist()
    assert layout.valid_tokens(compact, 1).tolist() == compact[3:5].tolist()
    assert layout.padded_tokens(compact, 1).tolist() == compact[3:5].tolist()
    assert layout.valid_position_ids().tolist() == [0, 1, 2, 1, 2]

    layout.write_valid_tokens(output, 1, torch.ones(2, 2))
    assert output[:3].abs().sum().item() == 0
    assert output[3:5].tolist() == torch.ones(2, 2).tolist()


def test_bshd_batch_layout_reads_and_scatters_kept_padded_sbh_tokens():
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [False, True, True, False, False],
            [True, False, False, False, False],
        ]
    )
    layout = BshdBatchLayout.from_valid_token_mask(mask, position_ids=torch.arange(5).repeat(3, 1))
    kept_padded = torch.arange(3 * 3 * 2, dtype=torch.float32).reshape(3, 3, 2)
    output = torch.zeros_like(kept_padded)

    assert layout.valid_lengths == [3, 2, 1]
    assert layout.valid_tokens(kept_padded, 0).tolist() == kept_padded[:3, 0].tolist()
    assert layout.valid_tokens(kept_padded, 1).tolist() == kept_padded[:2, 1].tolist()
    assert layout.padded_tokens(kept_padded, 2).tolist() == kept_padded[:, 2].tolist()
    assert layout.kept_padded_position_ids().tolist() == [
        [0, 1, 0],
        [1, 2, 0],
        [2, 0, 0],
    ]

    layout.write_valid_tokens(output, 1, torch.ones(2, 2))
    assert output[:2, 1].tolist() == torch.ones(2, 2).tolist()
    assert output[2, 1].abs().sum().item() == 0


def test_bshd_batch_layout_reads_and_scatters_tp_padded_sbh_tokens():
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [False, True, True, False, False],
            [True, False, False, False, False],
        ]
    )
    layout = BshdBatchLayout.from_valid_token_mask(mask, position_ids=torch.arange(5).repeat(3, 1))
    tp_padded = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    output = torch.zeros_like(tp_padded)

    assert layout.valid_lengths == [3, 2, 1]
    assert layout.valid_tokens(tp_padded, 0).tolist() == tp_padded[:3, 0].tolist()
    assert layout.valid_tokens(tp_padded, 1).tolist() == tp_padded[:2, 1].tolist()
    assert layout.padded_tokens(tp_padded, 2).tolist() == tp_padded[:, 2].tolist()
    assert layout.kept_padded_position_ids(padded_length=4).tolist() == [
        [0, 1, 0],
        [1, 2, 0],
        [2, 0, 0],
        [0, 0, 0],
    ]

    layout.write_valid_tokens(output, 1, torch.ones(2, 2))
    assert output[:2, 1].tolist() == torch.ones(2, 2).tolist()
    assert output[2:, 1].abs().sum().item() == 0
