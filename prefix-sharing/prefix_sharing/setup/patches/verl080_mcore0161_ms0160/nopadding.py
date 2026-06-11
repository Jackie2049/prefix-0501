"""Patch: no_padding_2_padding — 修正 prefix-sharing 物理裁剪后的序列长度断言

物理裁剪后，模型输出 flatten token 数（trimmed）小于外层 batch_td 中
attention_mask 标注的原始 token 总数，导致 assertion failed。

forward_step 通过模块级变量 ``_ps_trimmed_valid_lengths`` 传裁剪后的长度，
本 patch 在此变量存在时用它替换 prompt_lens 计算。

``no_padding_2_padding`` 在 ``forward_step`` 返回后调用，
此时 PS ContextVar 已 exit，所以不能用 ContextVar 传值。
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from verl.utils import tensordict_utils as tu

logger = logging.getLogger(__name__)


def patch_no_padding_2_padding(original_func: Any) -> Any:
    """创建 no_padding_2_padding 的 patch wrapper。"""

    def patched_no_padding_2_padding(tensor: Any, data: Any) -> Any:
        # ── 读模块级变量（forward_step 在裁剪时存入，不消费） ──
        from prefix_sharing.integrations.verl_mcore import get_trimmed_valid_lengths

        trimmed_valid_lengths = get_trimmed_valid_lengths()

        if trimmed_valid_lengths is not None:
            return _no_padding_2_padding_with_trimmed_lengths(
                tensor, data, trimmed_valid_lengths,
            )

        # ── normal path ──
        return original_func(tensor, data)

    return patched_no_padding_2_padding


def _no_padding_2_padding_with_trimmed_lengths(
    tensor: Any,
    data: Any,
    trimmed_valid_lengths: list[int],
) -> Any:
    """使用裁剪后的序列长度代替原始 attention_mask 计算。"""
    values = tensor.values() if tensor.is_nested else tensor
    prompt_ids = data["prompts"]
    response_ids = data["responses"]

    max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

    # ── response_lens 不变（response 没被裁） ──
    if hasattr(response_ids, "is_nested") and response_ids.is_nested:
        response_lens = response_ids.offsets().diff()
    else:
        attention_mask = data.get("attention_mask")
        assert attention_mask is not None and not attention_mask.is_nested
        response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)

    if max_response_len < 0:
        max_response_len = int(response_lens.max().item())

    # ── 从裁剪后的 valid_lengths 减去 response_lens 得到裁剪后的 prompt_lens ──
    trimmed_prompt_lens = [
        max(0, int(vl) - int(rl))
        for vl, rl in zip(trimmed_valid_lengths, response_lens)
    ]

    sequence_lens = torch.tensor(trimmed_prompt_lens, device=values.device) + response_lens
    sequence_offsets = sequence_lens.cumsum(dim=0)

    assert sequence_offsets[-1].item() == values.shape[0], (
        f"[PS] sequence_offsets[-1]={sequence_offsets[-1].item()} != "
        f"values.shape[0]={values.shape[0]}, "
        f"trimmed_prompt_lens={trimmed_prompt_lens}, "
        f"response_lens={response_lens.tolist()}, "
        f"valid_lengths={[int(vl) for vl in sequence_lens]}"
    )
    assert trimmed_prompt_lens[0] > 0, (
        f"[PS] Provider (row 0) should have prompt_len > 0. "
        f"Got trimmed_prompt_lens={trimmed_prompt_lens}"
    )

    response_list = []
    skip_padding = (0, 0) * (values.ndim - 1)
    for resp_len, seq_offset in zip(response_lens, sequence_offsets, strict=True):
        pad_size = max_response_len - resp_len
        response_list.append(
            F.pad(
                values[seq_offset - resp_len - 1 : seq_offset - 1],
                (*skip_padding, 0, pad_size),
            )
        )

    return torch.stack(response_list, dim=0)
