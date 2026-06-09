"""新版 API 适配层 — setup 专用。

不修改 integrations/ 任何文件。所有新版 API 适配逻辑在这里：
1. RoPE：v0.16.0 apply_rotary_pos_emb 新增 cu_seqlens 参数
2. Batch：verl 0.8.x 使用 NestedTensor (jagged layout)
3. Logprob restore：THD packed 1D 格式适配
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════
# RoPE 适配
# ═══════════════════════════════════════

def apply_positioned_rope_v016(
    attention_module: Any,
    query: Any,
    key: Any,
    q_pos_emb: Any,
    k_pos_emb: Any,
    kept_position_ids: Any,
    packed_seq_params: Any,
    *,
    mscale: float,
) -> tuple[Any, Any]:
    """v0.16.0 版 apply_rotary_pos_emb 适配：传入 cu_seqlens 参数。

    逻辑与 integrations/megatron_runtime._apply_positioned_rope 相同，
    但增加 v0.16.0 要求的 cu_seqlens 参数。
    """
    from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

    positions = kept_position_ids.to(device=query.device, dtype=torch.long)
    max_needed = positions.max().item() + 1

    q_pos_emb = _extend_pos_emb(q_pos_emb, max_needed, query.device, query.dtype)
    k_pos_emb = _extend_pos_emb(k_pos_emb, max_needed, key.device, key.dtype)

    cu_seqlens_q = _extract_cu_seqlens(
        packed_seq_params, "cu_seqlens_q_padded", "cu_seqlens_q"
    )
    cu_seqlens_kv = _extract_cu_seqlens(
        packed_seq_params, "cu_seqlens_kv_padded", "cu_seqlens_kv"
    )

    if q_pos_emb is not None:
        q_freqs = q_pos_emb.index_select(0, positions)
        query = apply_rotary_pos_emb(
            query.unsqueeze(1),
            q_freqs,
            config=attention_module.config,
            cu_seqlens=cu_seqlens_q,
            mscale=mscale,
            cp_group=attention_module.pg_collection.cp,
        ).squeeze(1)
    if k_pos_emb is not None:
        k_freqs = k_pos_emb.index_select(0, positions)
        key = apply_rotary_pos_emb(
            key.unsqueeze(1),
            k_freqs,
            config=attention_module.config,
            cu_seqlens=cu_seqlens_kv,
            mscale=mscale,
            cp_group=attention_module.pg_collection.cp,
        ).squeeze(1)
    return query, key


def _extend_pos_emb(pos_emb, max_needed, device, dtype):
    """线性扩展 pos_emb 到 max_needed 长度（RoPE 线性性质）。"""
    if pos_emb is None or max_needed <= pos_emb.shape[0]:
        return pos_emb
    dim_half = pos_emb.shape[-1] // 2
    step = pos_emb[1:2, :, :, :dim_half] - pos_emb[0:1, :, :, :dim_half]
    extra_positions = torch.arange(
        pos_emb.shape[0], max_needed, device=device, dtype=dtype
    )
    extra_angles = extra_positions[:, None, None, None] * step
    extra_emb = torch.cat([extra_angles, extra_angles], dim=-1)
    return torch.cat([pos_emb, extra_emb], dim=0)


def _extract_cu_seqlens(packed_seq_params, primary_attr, fallback_attr):
    """从 packed_seq_params 提取 cu_seqlens，优先 padded 版本。"""
    if packed_seq_params is None:
        return None
    val = getattr(packed_seq_params, primary_attr, None)
    if val is None:
        val = getattr(packed_seq_params, fallback_attr, None)
    return val


# ═══════════════════════════════════════
# Batch 适配
# ═══════════════════════════════════════

def extract_sequences_from_batch(
    batch: Any,
) -> tuple[list[list[int]], str]:
    """从 verl batch 中提取 token 序列，自动识别数据格式。

    Returns: (sequences, format) — format 为 "nested" 或 "plain"
    """
    input_ids = batch["input_ids"]

    if isinstance(input_ids, torch.nested.NestedTensor):
        offsets = input_ids.offsets()
        values = input_ids.values()
        lengths = offsets.diff().tolist()
        sequences = [
            values[offsets[i]:offsets[i + 1]].detach().cpu().tolist()
            for i in range(len(lengths))
        ]
        return sequences, "nested"

    # plain tensor（旧版 verl）
    attention_mask = batch["attention_mask"].to(bool)
    valid_indices = [
        attention_mask[row].nonzero(as_tuple=False).flatten()
        for row in range(input_ids.shape[0])
    ]
    sequences = [
        input_ids[row, idx].detach().cpu().tolist()
        for row, idx in enumerate(valid_indices)
    ]
    return sequences, "plain"


def trim_batch(batch: Any, plan: Any, format: str) -> Any:
    """根据 format 类型裁剪 batch。plan 包含 input_keep_ranges。"""
    if format == "nested":
        return _trim_nested_batch(batch, plan)
    return _trim_plain_batch(batch, plan)


def _trim_nested_batch(batch, plan):
    """裁剪 NestedTensor batch（verl 0.8.x jagged layout）。"""
    input_ids = batch["input_ids"]
    offsets = input_ids.offsets()
    values = input_ids.values()

    trimmed_sequences = []
    for i in range(len(plan.input_keep_ranges)):
        keep_start, keep_end = plan.input_keep_ranges[i]
        trimmed_sequences.append(
            values[offsets[i] + keep_start: offsets[i + 1] + keep_end]
        )
    new_input_ids = torch.nested.nested_tensor(
        trimmed_sequences, layout=torch.jagged
    )

    position_ids = batch["position_ids"]
    if isinstance(position_ids, torch.nested.NestedTensor):
        pos_values = position_ids.values()
        pos_offsets = position_ids.offsets()
        trimmed_positions = []
        for i in range(len(plan.input_keep_ranges)):
            keep_start, keep_end = plan.input_keep_ranges[i]
            trimmed_positions.append(
                pos_values[pos_offsets[i] + keep_start: pos_offsets[i + 1] + keep_end]
            )
        new_position_ids = torch.nested.nested_tensor(
            trimmed_positions, layout=torch.jagged
        )
    else:
        new_position_ids = position_ids

    trimmed_batch = batch.clone()
    trimmed_batch["input_ids"] = new_input_ids
    trimmed_batch["position_ids"] = new_position_ids
    return trimmed_batch


def _trim_plain_batch(batch, plan):
    """裁剪 plain tensor batch（旧版 verl 数据格式）。"""
    attention_mask = batch["attention_mask"].to(bool)
    position_ids = batch["position_ids"]

    new_attention_mask = attention_mask.clone()
    new_attention_mask[:] = False
    kept_position_rows = []

    valid_indices = [
        attention_mask[row].nonzero(as_tuple=False).flatten()
        for row in range(attention_mask.shape[0])
    ]

    for row, indices in enumerate(valid_indices):
        keep_start, keep_end = plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        new_attention_mask[row, kept_indices] = True
        kept_position_rows.append(position_ids[row, kept_indices])

    trimmed_batch = batch.clone()
    trimmed_batch["attention_mask"] = new_attention_mask
    return trimmed_batch


def compute_packed_cu_seqlens(
    batch: Any, plan: Any, format: str
) -> list[int]:
    """计算 THD packed layout 的 cu_seqlens（含 TP/CP 对齐 padding）。"""
    if format == "nested":
        input_ids = batch["input_ids"]
        offsets = input_ids.offsets()
        return offsets.tolist()

    # plain tensor
    attention_mask = batch["attention_mask"].to(bool)
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    try:
        from megatron.core import parallel_state as mpu
        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
    except Exception:
        tp_size = 1
        cp_size = 1
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    pad_sizes = (align_size - seqlens % align_size) % align_size
    seqlens_padded = seqlens + pad_sizes
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens_padded, dim=0)
    return [int(v) for v in cu_seqlens.tolist()]


def collect_kept_position_ids(
    batch: Any, plan: Any, format: str
) -> Any:
    """从 batch 中收集各序列 kept 区段的 position_ids，拼接为 1D tensor。"""
    if format == "nested":
        position_ids = batch["position_ids"]
        pos_values = position_ids.values()
        pos_offsets = position_ids.offsets()
        pieces = []
        for i in range(len(plan.input_keep_ranges)):
            keep_start, keep_end = plan.input_keep_ranges[i]
            pieces.append(
                pos_values[pos_offsets[i] + keep_start: pos_offsets[i + 1] + keep_end]
            )
        return torch.cat(pieces, dim=0)

    # plain tensor
    attention_mask = batch["attention_mask"].to(bool)
    position_ids = batch["position_ids"]
    valid_indices = [
        attention_mask[row].nonzero(as_tuple=False).flatten()
        for row in range(position_ids.shape[0])
    ]
    pieces = []
    for row, indices in enumerate(valid_indices):
        keep_start, keep_end = plan.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        pieces.append(position_ids[row, kept_indices])
    return torch.cat(pieces, dim=0)


# ═══════════════════════════════════════
# Logprob restore 适配
# ═══════════════════════════════════════

def restore_logprobs(
    logits: Any,
    labels: Any,
    log_probs: Any,
    vocab_parallel_log_probs_fn: Any,
    prefix_last_restore_indices: Any,
) -> Any:
    """THD packed 1D 格式的 logprob restore。

    从 provider 的 prefix-last logits 位置重新计算 vocab_parallel_log_probs，
    写入 reuser 的 suffix-first logprob 位置。
    """
    if not prefix_last_restore_indices:
        return log_probs

    restored = log_probs.clone()
    for index in prefix_last_restore_indices:
        provider_logits = logits[
            0:1, index.provider_1d_pos: index.provider_1d_pos + 1, :
        ].clone()
        reuse_label = labels[
            0:1, index.reuse_1d_pos: index.reuse_1d_pos + 1,
        ]
        restored_value = vocab_parallel_log_probs_fn(provider_logits, reuse_label)
        restored[0, index.reuse_1d_pos] = restored_value.reshape(())
    return restored