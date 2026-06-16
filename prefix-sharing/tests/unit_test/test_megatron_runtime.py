import pytest

torch = pytest.importorskip("torch")

from prefix_sharing.backends.batch_layout import BshdBatchLayout
from prefix_sharing.integrations.megatron_runtime import _positions_for_tensor


def test_positions_for_tensor_supports_tp_padded_bshd_sbh_tensor():
    mask = torch.zeros(8, 160, dtype=torch.bool)
    mask[0, :95] = True
    for row in range(1, 8):
        mask[row, :40] = True
    position_ids = torch.arange(160).repeat(8, 1)
    layout = BshdBatchLayout.from_valid_token_mask(mask, position_ids=position_ids)
    tensor = torch.zeros(96, 8, 7, 64)

    positions = _positions_for_tensor(layout, tensor, device=tensor.device)

    assert positions.numel() == 96 * 8
    assert positions.reshape(96, 8)[:95, 0].tolist() == list(range(95))
    assert positions.reshape(96, 8)[95, 0].item() == 0
