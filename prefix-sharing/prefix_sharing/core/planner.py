"""Prefix-sharing planning: turn detection output into batch metadata.

This module sits between **prefix detection** (:mod:`prefix_sharing.core.prefix_detector`)
and **downstream execution** (mapping, attention backends, RoPE). It does not
discover shared prefixes; it consumes :class:`~prefix_sharing.core.prefix_detector.PrefixDetectionResult`
and materializes per-row layout: what to keep on the Q path, cumulative sequence
lengths for packed tensors, position offsets, and optional prefix-last restore
hints for RL / logprob alignment.

Core Responsibilities:
    1. **Per-row layout after detection**: for each batch index, decide how that
       row participates in packed execution—whether the Q path uses the full
       sequence or only the post-prefix suffix, how long the row occupies on
       the KV path, and which contiguous token spans to keep for inputs, labels,
       and loss masks. Inputs are only ``provider_index``, ``prefix_lens``, and
       per-row original length; the result is encoded as lengths, cumulative
       seqlens, and keep ranges inside ``PrefixSharingPlan``.
    2. Set ``q_position_offsets`` so reuser rows preserve correct absolute positions
       when the Q path only packs the suffix after a shared prefix.
    3. Emit :class:`PrefixRestoreSpec` entries when a reuser has both a
       non-zero prefix and a non-empty suffix.

Key Concepts:
    - Reuser (planner view): ``provider_index[i] != i`` and ``prefix_lens[i] > 0``.
      The Q path keeps ``input_ids[i][prefix_len:original_len)``; KV expanded
      length stays the full original length for backend conventions.
    - Provider or standalone row: full sequence on the Q path; offsets zero.

Key Components:
    - :class:`PrefixSharingPlanner`: Optional embedded :class:`~prefix_sharing.core.prefix_detector.TriePrefixDetector`;
      :meth:`~PrefixSharingPlanner.plan` runs detect-then-plan;
      :meth:`~PrefixSharingPlanner.plan_from_detection` accepts precomputed detection
      (tests or alternate detectors).
    - :func:`_cumsum`: Computes cumulative lengths for ``cu_seqlens_q`` and ``cu_seqlens_kv``.

Design Principles:
    - Single layout source: all ranges and lengths derive from detector fields
      plus original lengths; planner does not reinterpret group structure.
    - Detector-agnostic planning: only ``PrefixDetectionResult`` + ``input_ids``
      lengths matter in :meth:`~PrefixSharingPlanner.plan_from_detection`.
    - Correlation hooks: optional ``forward_id`` and ``micro_batch_id`` for logging,
      cache keys, and pipeline tracing (auto-assigned when omitted).

Example:
    >>> from prefix_sharing.core.config import PrefixSharingConfig
    >>> planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    >>> prefix_sharing_plan = planner.plan([[1, 2, 3, 4], [1, 2, 3, 5]], forward_id=1, micro_batch_id=1)
    >>> prefix_sharing_plan.has_sharing
    True
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Sequence

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.prefix_detector import PrefixDetectionResult, PrefixReuseSpec, TriePrefixDetector


@dataclass(frozen=True)
class PrefixRestoreSpec:
    """Plan for one reuse row's logprob restore.

    Two kinds of restore exist:

    * **Prefix-last restore** (``restore_type='restore_prefix_last'``, default):
      the reuser's first suffix token is predicted from the shared prefix-last
      position. Since different reusers may have different first suffix labels,
      the provider's full logits at ``prefix_len - 1`` must be stored and the
      logprob computed per reuser using that reuser's label.

    * **Shared-prefix interior restore** (``restore_type='restore_prefix_interior'``):
      a token that lives strictly inside the shared prefix. Its
      logprob is ``log_softmax(logits[pos-1])[label=pos]``, where both the
      logits position and the label belong to the shared prefix. The logprob
      is therefore **identical for all reusers** and can be computed once from
      the provider's logits, stored as a non-detached scalar tensor, and
      reused for every reuser without re-reading logits.
    """

    reuse_idx_in_batch: int
    provider_idx_in_batch: int
    provider_predict_pos: int
    """Logits position in provider used for this restore (logits[p] predicts
    token at p+1). Equals ``target_2d_pos`` numerically; kept as a separate
    field to document the provider-side lookup semantic and to drive provider
    chain resolution in ``context.py``."""
    reuse_first_suffix_label_pos: int
    group_id: int
    restore_type: str = "restore_prefix_last"
    target_2d_pos: int = -1
    """Absolute 2D position in output tensor where restored logprob belongs.
    
    This is the label position in the original (untrimmed) sequence:
      log_probs[i] = log_softmax(logits[i])[label[i]]
    For shared-prefix interior: ``target_2d_pos = prefix_label_pos - 1`` (label index).
    For prefix-last:    ``target_2d_pos = prefix_len - 1`` (first-suffix label index).
    """
    label_value: int = -1
    """The actual token ID used as label for logprob computation.
    
    For shared-prefix interior: token at ``prefix_label_pos`` (shared prefix, same for provider/reuser).
    For prefix-last: token at ``prefix_len`` (reuser's first suffix token).
    This value is needed because trimmed packed labels don't contain these positions.
    """


Range = tuple[int, int]


@dataclass(frozen=True)
class PrefixSharingPlan:
    """Complete framework-independent plan for a single micro-batch."""

    # 基础标识信息
    forward_id: int                              # 前向传播唯一标识
    micro_batch_id: int                          # micro-batch序号
    batch_size: int                              # 批次内序列数量
    original_lengths: list[int]                   # 各序列原始长度

    # 前缀复用关系
    reuse_specs: list[PrefixReuseSpec]           # 序列间复用关系规范
    group_ids: list[int]                        # 各序列所属前缀组ID
    is_provider: list[bool]                      # 各序列是否为provider（被复用方）
    provider_index: list[int]                    # 各序列的provider在batch中的索引
    prefix_lens: list[int]                      # 各序列前缀长度（可共享部分）
    suffix_lens: list[int]                      # 各序列后缀长度（需独立计算部分）

    # THD格式下的序列长度管理
    kept_lengths_q: list[int]                   # 裁剪后各序列的Q长度
    expanded_lengths_kv: list[int]             # 扩展后各序列的KV长度（包含共享前缀）
    cu_seqlens_q: list[int]                     # Q的累积序列长度（用于THD索引）
    cu_seqlens_kv: list[int]                   # KV的累积序列长度（用于THD索引）
    max_seqlen_q: int                          # Q的最大序列长度
    max_seqlen_kv: int                         # KV的最大序列长度

    # 位置偏移（用于恢复原始位置信息）
    q_position_offsets: list[int]               # Q相对于原始序列的位置偏移
    kv_position_offsets: list[int]             # KV相对于原始序列的位置偏移

    # 裁剪保留范围（start, end）
    input_keep_ranges: list[Range]              # input_ids保留范围
    label_keep_ranges: list[Range]             # labels保留范围
    loss_mask_keep_ranges: list[Range]         # loss_mask保留范围

    # 恢复点信息（用于logprob恢复）
    prefix_restore_specs: list[PrefixRestoreSpec] = field(default_factory=list)  # reuser的prefix相关位置恢复规范

    def __post_init__(self) -> None:
        expected = self.batch_size
        fields: Sequence[tuple[str, list[object]]] = (
            ("original_lengths", self.original_lengths),
            ("group_ids", self.group_ids),
            ("is_provider", self.is_provider),
            ("provider_index", self.provider_index),
            ("prefix_lens", self.prefix_lens),
            ("suffix_lens", self.suffix_lens),
            ("kept_lengths_q", self.kept_lengths_q),
            ("expanded_lengths_kv", self.expanded_lengths_kv),
            ("q_position_offsets", self.q_position_offsets),
            ("kv_position_offsets", self.kv_position_offsets),
            ("input_keep_ranges", self.input_keep_ranges),
            ("label_keep_ranges", self.label_keep_ranges),
            ("loss_mask_keep_ranges", self.loss_mask_keep_ranges),
        )
        for name, value in fields:
            if len(value) != expected:
                raise ValueError(f"{name} length must equal batch_size={expected}")
        if len(self.cu_seqlens_q) != expected + 1:
            raise ValueError("cu_seqlens_q length must equal batch_size + 1")
        if len(self.cu_seqlens_kv) != expected + 1:
            raise ValueError("cu_seqlens_kv length must equal batch_size + 1")

    @property
    def has_sharing(self) -> bool:
        return bool(self.reuse_specs)

    def is_reuser(self, idx_in_batch: int) -> bool:
        return self.provider_index[idx_in_batch] != idx_in_batch and self.prefix_lens[idx_in_batch] > 0

    def q_range_for_batch(self, idx_in_batch: int) -> Range:
        return self.cu_seqlens_q[idx_in_batch], self.cu_seqlens_q[idx_in_batch + 1]

    def kv_range_for_batch(self, idx_in_batch: int) -> Range:
        return self.cu_seqlens_kv[idx_in_batch], self.cu_seqlens_kv[idx_in_batch + 1]

    def restore_for_reuse(self, idx_in_batch: int) -> PrefixRestoreSpec | None:
        for spec in self.prefix_restore_specs:
            if spec.reuse_idx_in_batch == idx_in_batch:
                return spec
        return None


_forward_ids = itertools.count(1)


def _cumsum(lengths: Sequence[int]) -> list[int]:
    values = [0]
    total = 0
    for length in lengths:
        total += int(length)
        values.append(total)
    return values


@dataclass
class PrefixSharingPlanner:
    config: PrefixSharingConfig
    detector: TriePrefixDetector | None = None
    _micro_batch_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.detector is None:
            self.detector = TriePrefixDetector(
                min_prefix_len=self.config.min_prefix_len,
                min_group_size=self.config.min_group_size,
            )

    def plan(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        forward_id: int | None = None,
        micro_batch_id: int | None = None,
    ) -> PrefixSharingPlan:
        detect_result = self.detector.detect(sequences)
        return self.plan_from_detection(
            sequences,
            detect_result,
            forward_id=forward_id,
            micro_batch_id=micro_batch_id,
        )

    def plan_from_detection(
        self,
        sequences: Sequence[Sequence[int]],
        detect_result: PrefixDetectionResult,
        *,
        forward_id: int | None = None,
        micro_batch_id: int | None = None,
    ) -> PrefixSharingPlan:
        if len(sequences) != detect_result.batch_size:
            raise ValueError("sequences batch size does not match detection result")
        if forward_id is None:
            forward_id = next(_forward_ids)
        if micro_batch_id is None:
            self._micro_batch_counter += 1
            micro_batch_id = self._micro_batch_counter

        # 来自 PrefixDetectionResult 的信息
        batch_size = len(sequences)
        original_lengths = [len(seq) for seq in sequences]
        group_ids = list(detect_result.group_ids)
        is_provider = list(detect_result.is_provider)
        provider_index = list(detect_result.provider_index)
        prefix_lens = list(detect_result.prefix_lens)
        reuse_specs = list(detect_result.reuse_specs)

        # PrefixSharingPlan 需要的额外信息
        suffix_lens: list[int] = []
        kept_lengths_q: list[int] = []
        expanded_lengths_kv: list[int] = []
        q_position_offsets: list[int] = []
        kv_position_offsets: list[int] = []
        input_keep_ranges: list[tuple[int, int]] = []
        label_keep_ranges: list[tuple[int, int]] = []
        loss_mask_keep_ranges: list[tuple[int, int]] = []
        restore_specs: list[PrefixRestoreSpec] = []

        for seq_idx, original_len in enumerate(original_lengths):
            prefix_len = prefix_lens[seq_idx]
            if prefix_len > original_len:
                raise ValueError(f"prefix_len exceeds sequence length for batch index {seq_idx}")
            is_reuser = provider_index[seq_idx] != seq_idx and prefix_len > 0
            suffix_len = original_len - prefix_len if is_reuser else original_len
            suffix_lens.append(suffix_len)
            expanded_lengths_kv.append(original_len)

            if is_reuser:
                # 1 - 为输入数据的预处理做准备：输入阶段要对 reuser 进行序列裁剪
                #   - 只需保留suffix部分
                #   - prefix 部分
                #       - KV 激活值：在 attention 时候读取缓存并进行拼接即可
                #       - logp：可以在后处理阶段直接从provider复制得到
                keep_start, keep_end = prefix_len, original_len
                kept_len = suffix_len
                q_offset = prefix_len

                # 2 - 为输出数据的后处理做准备：输出阶段要为 reuser 恢复没有计算的前缀部分的 logp 和 entropy
                # --- Shared-prefix interior token restore ---
                # Response tokens inside the shared prefix (positions
                # 1 .. prefix_len-1) are trimmed from the Q path but
                # still need logprob entries for PPO loss. Their labels are
                # inside the shared prefix so the logprob is identical for
                # provider and reuser: compute once from provider logits,
                # store as a non-detached scalar tensor.
                # Note: prompt positions (0..prompt_len-1) are also
                # restored, but downstream label_mask is False for them
                # so they don't affect loss — restoring the whole prefix
                # column uniformly keeps planning simple.
                for prefix_label_pos in range(1, prefix_len):
                    # Shared-prefix interior token logprob: computed from
                    # logits[prefix_label_pos-1] predicting token at
                    # prefix_label_pos. Writes to 2D position
                    # prefix_label_pos - 1 (the label position).
                    restore_specs.append(
                        PrefixRestoreSpec(
                            reuse_idx_in_batch=seq_idx,
                            provider_idx_in_batch=provider_index[seq_idx],
                            provider_predict_pos=prefix_label_pos - 1,
                            reuse_first_suffix_label_pos=prefix_label_pos,
                            group_id=group_ids[seq_idx],
                            restore_type="restore_prefix_interior",
                            target_2d_pos=prefix_label_pos - 1,
                            label_value=sequences[seq_idx][prefix_label_pos],
                        )
                    )

                # --- Prefix-last restore (first suffix token) ---
                # The logits at position prefix_len-1 predict the first suffix
                # token whose label is sequences[prefix_len] (differs per
                # reuser). The restored logprob is written to 2D position
                # prefix_len - 1 (the label slot for first-suffix prediction).
                if prefix_len > 0 and suffix_len > 0:
                    restore_specs.append(
                        PrefixRestoreSpec(
                            reuse_idx_in_batch=seq_idx,
                            provider_idx_in_batch=provider_index[seq_idx],
                            provider_predict_pos=prefix_len - 1,
                            reuse_first_suffix_label_pos=prefix_len,
                            group_id=group_ids[seq_idx],
                            target_2d_pos=prefix_len - 1,
                            label_value=sequences[seq_idx][prefix_len],
                        )
                    )
            else:
                keep_start, keep_end = 0, original_len
                kept_len = original_len
                q_offset = 0

            kept_lengths_q.append(kept_len)
            q_position_offsets.append(q_offset)
            kv_position_offsets.append(0)
            keep_range = (keep_start, keep_end)
            input_keep_ranges.append(keep_range)
            label_keep_ranges.append(keep_range)
            loss_mask_keep_ranges.append(keep_range)

        cu_seqlens_q = _cumsum(kept_lengths_q)
        cu_seqlens_kv = _cumsum(expanded_lengths_kv)

        return PrefixSharingPlan(
            forward_id=forward_id,
            micro_batch_id=micro_batch_id,
            batch_size=batch_size,
            original_lengths=original_lengths,
            reuse_specs=reuse_specs,
            group_ids=group_ids,
            is_provider=is_provider,
            provider_index=provider_index,
            prefix_lens=prefix_lens,
            suffix_lens=suffix_lens,
            kept_lengths_q=kept_lengths_q,
            expanded_lengths_kv=expanded_lengths_kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max(kept_lengths_q, default=0),
            max_seqlen_kv=max(expanded_lengths_kv, default=0),
            q_position_offsets=q_position_offsets,
            kv_position_offsets=kv_position_offsets,
            input_keep_ranges=input_keep_ranges,
            label_keep_ranges=label_keep_ranges,
            loss_mask_keep_ranges=loss_mask_keep_ranges,
            prefix_restore_specs=restore_specs,
        )
