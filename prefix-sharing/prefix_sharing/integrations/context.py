"""Runtime context passed from framework preprocess into patched attention."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.core.observability import PrefixSharingStats
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.prefix_store import PrefixAttentionStore
from prefix_sharing.integrations.parallel_info import MegatronParallelInfo


_current_context: ContextVar["PrefixSharingRuntimeContext | None"] = ContextVar(
    "prefix_sharing_context",
    default=None,
)


@dataclass
class PackedPrefixLastRestoreIndex:
    reuse_idx_in_batch: int
    provider_idx_in_batch: int
    provider_1d_pos: int
    reuse_1d_pos: int
    is_shared_prefix_interior: bool = False
    target_2d_pos: int = -1
    """Absolute 2D position in output where restored logprob is written."""
    label_value: int = -1
    """Actual token ID used as label (needed when label isn't in trimmed packed region)."""


@dataclass(init=False)
class PrefixSharingRuntimeContext:
    prefix_sharing_plan: PrefixSharingPlan
    packed_batch_layout: PackedBatchLayout
    parallel_info: MegatronParallelInfo
    store: PrefixAttentionStore
    attention_backend: Any | None = None
    kept_position_ids: Any | None = None
    prefix_last_restore_indices: list[PackedPrefixLastRestoreIndex] = field(default_factory=list)
    prefix_last_logits_saved: dict[tuple[int, int], Any] = field(default_factory=dict)
    """Saved provider packed logits for prefix-last logprob recompute in 2D space.

    Keyed by (reuse_idx_in_batch, target_2d_pos). Each value is [1, V//tp]
    cloned from packed logits after temperature scaling, before any
    in-place modification by entropy/logprob computation.

    Only populated for non-interior (prefix-last) restore specs.
    Interior specs use direct 2D copy instead.
    """
    valid_indices: list | None = None
    """Per-row tensor positions of valid tokens in the original 2D tensors.
    Used to map planner's valid-space target_2d_pos to tensor-space columns."""
    stats: PrefixSharingStats | None = None

    def __init__(self, runtime_state: Any, store: PrefixAttentionStore) -> None:
        self.prefix_sharing_plan = runtime_state.prefix_sharing_plan
        self.packed_batch_layout = runtime_state.packed_batch_layout
        self.parallel_info = runtime_state.parallel_info
        self.store = store
        self.attention_backend = runtime_state.attention_backend
        self.kept_position_ids = getattr(runtime_state, "kept_position_ids", None)
        self.prefix_last_restore_indices = _build_prefix_last_restore_indices(
            runtime_state.prefix_sharing_plan,
            runtime_state.packed_batch_layout,
        )
        # Provider packed logits saved for prefix-last logprob recompute in 2D
        # space.  Populated lazily by the verl vocab-logprobs patch for each
        # non-interior (prefix-last) restore spec; read by
        # restore_reuser_prefix_columns_2d.  Always present (possibly empty).
        self.prefix_last_logits_saved: dict[tuple[int, int], Any] = {}
        # valid_indices maps planner's valid-space target_2d_pos to tensor
        # columns; used by the 2D restore path in verl_mcore.  May be absent
        # on minimal runtime states (e.g. unit-test fixtures).
        self.valid_indices = getattr(runtime_state, "valid_indices", None)
        # stats drives the per-micro-batch audit log.  Prefer an explicit
        # stats object carried by the runtime state; otherwise derive one
        # from the plan so the audit summary is always available.
        self.stats = getattr(runtime_state, "stats", None)
        if self.stats is None:
            try:
                self.stats = PrefixSharingStats.from_plan(
                    self.prefix_sharing_plan, self.packed_batch_layout
                )
            except Exception:
                self.stats = None


def current_prefix_sharing_context() -> PrefixSharingRuntimeContext | None:
    return _current_context.get()


def _resolve_provider_for_position(
    plan: PrefixSharingPlan,
    provider_idx: int,
    target_pos: int,
) -> int:
    """Walk up the provider chain to find the nearest ancestor whose
    packed layout contains ``target_pos`` (absolute original position).

    In chain-reuse scenarios (row 0 → row 1 → row 2), intermediate
    providers are reusers with truncated packed layouts.  The root
    provider may also be shorter than the reuser's prefix — in that
    case intermediate providers contribute the extended range.

    The range is extended left by one position (keep_start - 1) for
    reusers: that position holds the logprob for the token at
    keep_start which is still inside the shared prefix.  Its value was
    computed via prefix-last recompute with the correct shared label
    and is safe to copy from.
    """
    while True:
        keep_start, keep_end = plan.input_keep_ranges[provider_idx]
        if keep_start - 1 <= target_pos < keep_end:
            return provider_idx
        if plan.is_reuser(provider_idx):
            provider_idx = plan.provider_index[provider_idx]
        else:
            # Can't walk further; return current (should not happen
            # with valid detection, but guard)
            return provider_idx


def _build_prefix_last_restore_indices(
    prefix_sharing_plan: PrefixSharingPlan,
    packed_batch_layout: PackedBatchLayout,
) -> list[PackedPrefixLastRestoreIndex]:
    indices = []
    for spec in prefix_sharing_plan.prefix_last_restore:
        reuse_idx = spec.reuse_idx_in_batch
        # Resolve through chain reuse to the nearest provider whose
        # packed layout contains provider_predict_pos.
        target_pos = spec.provider_predict_pos
        resolved_provider = _resolve_provider_for_position(
            prefix_sharing_plan, spec.provider_idx_in_batch, target_pos
        )

        # Interior: provider and reuser share the same prefix tokens,
        # so the logprob at target_pos is identical — just copy.
        # 2D restore uses provider_idx_in_batch + valid_indices mapping,
        # no packed-index lookup needed.  1d_pos_in_provider is only used
        # for prefix-last entries to fetch saved provider logits.
        provider_offset = (
            target_pos - prefix_sharing_plan.input_keep_ranges[resolved_provider][0]
        )
        if provider_offset >= 0:
            # target_pos is inside the resolved provider's packed region.
            pos_1d_in_provider = packed_batch_layout.packed_index(resolved_provider, provider_offset)
        else:
            # Interior spec resolved to an intermediate reuser's keep_start-1
            pos_1d_in_provider = -1

        reuse_1d = -1  # sentinel: no slot in reuser packed region

        indices.append(
            PackedPrefixLastRestoreIndex(
                reuse_idx_in_batch=reuse_idx,
                provider_idx_in_batch=resolved_provider,
                provider_1d_pos=pos_1d_in_provider,
                reuse_1d_pos=reuse_1d,
                is_shared_prefix_interior=spec.is_shared_prefix_interior,
                target_2d_pos=spec.target_2d_pos,
                label_value=spec.label_value,
            )
        )
    return indices


@contextmanager
def prefix_sharing_runtime_context(
    prefix_sharing_runtime_state: Any | None,
) -> Iterator[PrefixSharingRuntimeContext | None]:
    if prefix_sharing_runtime_state is None:
        yield None
        return

    store = PrefixAttentionStore()
    ctx = PrefixSharingRuntimeContext(prefix_sharing_runtime_state, store)
    token = _current_context.set(ctx)
    try:
        yield ctx
    finally:
        _current_context.reset(token)
        _log_prefix_sharing_audit(ctx)
        ctx.store.close()


def _log_prefix_sharing_audit(ctx: PrefixSharingRuntimeContext) -> None:
    stats = ctx.stats
    if stats is None:
        return
    global_rank, tp_rank, tp_size = _read_parallel_rank_info()
    print(
        f"[PS][audit][global_rank={global_rank} tp_rank={tp_rank}/tp_size={tp_size}] summary: "
        f"forward_id={stats.forward_id} micro_batch_id={stats.micro_batch_id} batch_size={stats.batch_size} "
        f"original_tokens={stats.original_tokens} kept_valid_tokens={stats.kept_valid_tokens} kept_padded_tokens={stats.kept_padded_tokens} "
        f"reused_valid_tokens={stats.reused_valid_tokens} reused_valid_token_ratio={stats.reused_valid_token_ratio:.4f} "
        f"provider_count={stats.provider_count} reuser_count={stats.reuser_count} sharing_group_count={stats.sharing_group_count} "
        f"expected_reused_counts_per_layer={stats.expected_reused_counts_per_layer} expected_reused_prefix_tokens_per_layer={stats.expected_reused_prefix_tokens_per_layer} "
        f"expected_restore_count={stats.expected_restore_count} actual_restore_count={stats.actual_restore_count}"
    )
    if not stats.layers and stats.expected_reused_counts_per_layer > 0:
        print(
            f"[PS][audit][global_rank={global_rank} tp_rank={tp_rank}/tp_size={tp_size}] runtime_missing: "
            f"expected_reused_counts_per_layer={stats.expected_reused_counts_per_layer} expected_reused_prefix_tokens_per_layer={stats.expected_reused_prefix_tokens_per_layer}"
        )
    for layer_id in sorted(stats.layers):
        layer = stats.layers[layer_id]
        print(
            f"[PS][audit][global_rank={global_rank} tp_rank={tp_rank}/tp_size={tp_size} layer={layer_id}] runtime: "
            f"store_count={layer.store_count} reuse_count={layer.reuse_count} reuse_hit_count={layer.reuse_hit_count} reuse_miss_count={layer.reuse_miss_count} "
            f"stored_tokens={layer.stored_tokens} reused_prefix_tokens={layer.reused_prefix_tokens} expanded_kv_tokens={layer.expanded_kv_tokens} "
            f"valid_q_tokens={layer.valid_q_tokens} padded_q_tokens={layer.padded_q_tokens} matches_expected={stats.layer_matches_expected(layer_id)}"
        )


def _read_parallel_rank_info() -> tuple[int | str, int, int]:
    global_rank: int | str = "unknown"
    tp_rank = 0
    tp_size = 1

    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            global_rank = int(dist.get_rank())
    except Exception:
        pass

    try:
        from megatron.core import parallel_state as mpu

        tp_size = int(mpu.get_tensor_model_parallel_world_size())
        if hasattr(mpu, "get_tensor_model_parallel_rank"):
            tp_rank = int(mpu.get_tensor_model_parallel_rank())
    except (ImportError, RuntimeError, AssertionError, AttributeError):
        pass

    return global_rank, tp_rank, tp_size
