"""Patch: no_padding_2_padding — prefix-sharing 物理裁剪后修正序列长度

物理裁剪后，模型输出 NestedTensor 的 offsets().diff() 给出每行的 trimmed 长度，
但 data["attention_mask"] 仍反映原始长度，导致
  sum(prompt_lens + response_lens) != values.shape[0]
断言失败。

修复：当 tensor 是 NestedTensor 且 offsets().diff().sum() 与
attention_mask 导出的 sum 不一致时，用 NestedTensor 自身的 offsets
作为 trimmed 序列长度。response 未被裁剪，所以
trimmed_prompt_lens = trimmed_seq_lens - response_lens。

不依赖任何跨进程通信或 metadata，纯靠模型输出的形状推导。
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
        values = tensor.values() if tensor.is_nested else tensor
        prompt_ids = data["prompts"]
        response_ids = data["responses"]

        max_response_len = tu.get_non_tensor_data(data=data, key="max_response_len", default=-1)

        if prompt_ids.is_nested:
            prompt_lens = prompt_ids.offsets().diff()
            response_lens = response_ids.offsets().diff()
            if max_response_len < 0:
                max_response_len = response_lens.max().item()
        else:
            attention_mask = data["attention_mask"]
            assert not attention_mask.is_nested
            prompt_lens = attention_mask[:, : prompt_ids.shape[1]].sum(dim=1)
            response_lens = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)
            max_response_len = response_ids.shape[1]

        sequence_lens = prompt_lens + response_lens
        sequence_offsets = sequence_lens.cumsum(dim=0)

        # ── prefix-sharing trimming detection ──
        # When PS physical trimming is active, the model output has fewer tokens
        # per row. The NestedTensor's offsets().diff() gives the true trimmed
        # per-row lengths. If they mismatch attention_mask-derived lengths,
        # use the NestedTensor's own offsets instead.
        if tensor.is_nested:
            trimmed_seq_lens = tensor.offsets().diff()
            expected_total = sequence_offsets[-1].item()
            actual_total = values.shape[0]
            if expected_total != actual_total:
                # PS trimming detected
                trimmed_prompt_lens = trimmed_seq_lens - response_lens
                sequence_lens = trimmed_prompt_lens + response_lens
                sequence_offsets = sequence_lens.cumsum(dim=0)
                logger.info(
                    "[PS][nopadding] trimming detected: "
                    "expected_total=%s actual_total=%s "
                    "original_prompt_lens=%s trimmed_prompt_lens=%s "
                    "response_lens=%s",
                    expected_total, actual_total,
                    prompt_lens.tolist(),
                    trimmed_prompt_lens.tolist(),
                    response_lens.tolist(),
                )

        assert sequence_offsets[-1].item() == values.shape[0], (
            f"sequence_offsets[-1]={sequence_offsets[-1].item()} != "
            f"values.shape[0]={values.shape[0]}"
        )
        assert not prompt_lens.eq(0).any(), (
            f"seq_offset - resp_len - 1 assumes prompt_len > 0. Got {prompt_lens}"
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

    return patched_no_padding_2_padding