"""verl Megatron actor integration helpers.

This module has two layers:

* ``VerlMCoreBatchAdapter`` is framework-light and testable locally. It turns a
  verl-style micro-batch payload into prefix-sharing metadata plus trimmed
  inputs/labels/masks, and it assembles restored logprobs after forward.
* ``VerlMCoreIntegration`` installs the Megatron attention patch. The real
  Megatron QKV rewiring still requires the framework runtime and remains guarded
  by optional integration tests.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence, TypeVar

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.mapping import (
    TrimmedBatch,
    restore_prefix_last_logprobs,
    trim_inputs,
    trim_labels,
    trim_loss_masks,
)
from prefix_sharing.core.metadata import PrefixSharingBatchMeta
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.integrations.context import (
    PrefixSharingRuntimeContext,
    current_prefix_sharing_context,
    prefix_sharing_context,
)
from prefix_sharing.integrations.megatron_attention import IntegrationUnavailable, MegatronAttentionIntegration
from prefix_sharing.integrations.patch_manager import PatchHandle


T = TypeVar("T")


@dataclass(frozen=True)
class DensePrefixLastRestoreSpec:
    reuse_idx_in_batch: int
    provider_idx_in_batch: int
    provider_dense_pos: int
    reuse_dense_pos: int


@dataclass(frozen=True)
class MegatronActorPreparedMicroBatch:
    meta: PrefixSharingBatchMeta
    backend: Any
    kept_position_ids: Any
    restore_positions: list[DensePrefixLastRestoreSpec]


@dataclass(frozen=True)
class VerlMCorePreparedBatch:
    """Framework-independent materialization of one verl actor micro-batch."""

    meta: PrefixSharingBatchMeta
    input_ids: TrimmedBatch[int]
    labels: TrimmedBatch[Any] | None = None
    loss_masks: TrimmedBatch[Any] | None = None


@dataclass
class VerlMCoreBatchAdapter:
    """Prepare and restore verl Megatron actor micro-batches.

    The adapter is intentionally tensor-agnostic for Phase 1 local tests. A real
    verl integration can map the returned ``TrimmedBatch.flattened`` and
    ``meta.cu_seqlens_q`` fields to torch tensors without changing core
    semantics.
    """

    config: PrefixSharingConfig
    planner: PrefixSharingPlanner | None = None

    def __post_init__(self) -> None:
        if self.planner is None:
            self.planner = PrefixSharingPlanner(self.config)

    def prepare_micro_batch(
        self,
        input_ids: Sequence[Sequence[int]],
        *,
        labels: Sequence[Sequence[T]] | None = None,
        loss_masks: Sequence[Sequence[T]] | None = None,
        forward_id: int | None = None,
        micro_batch_id: int | None = None,
    ) -> VerlMCorePreparedBatch:
        """Plan prefix sharing and apply mapping trims to a micro-batch."""

        assert self.planner is not None
        meta = self.planner.plan(
            input_ids,
            forward_id=forward_id,
            micro_batch_id=micro_batch_id,
        )
        trimmed_inputs = trim_inputs(input_ids, meta)
        trimmed_labels = trim_labels(labels, meta) if labels is not None else None
        trimmed_loss_masks = trim_loss_masks(loss_masks, meta) if loss_masks is not None else None
        return VerlMCorePreparedBatch(
            meta=meta,
            input_ids=trimmed_inputs,
            labels=trimmed_labels,
            loss_masks=trimmed_loss_masks,
        )

    @contextmanager
    def prepared_context(
        self,
        prepared: VerlMCorePreparedBatch,
    ) -> Iterator[PrefixSharingRuntimeContext]:
        """Open the runtime context consumed by patched attention."""

        with prefix_sharing_context(prepared.meta, backend=TorchReferenceBackend()) as ctx:
            yield ctx

    def restore_logprobs(
        self,
        suffix_logprobs: Sequence[Sequence[float]],
        provider_prefix_last_logprobs: Sequence[float],
        meta: PrefixSharingBatchMeta,
    ) -> list[list[float]]:
        """Assemble per-row logprobs with Prefix-Last Restore."""

        return restore_prefix_last_logprobs(
            suffix_logprobs,
            provider_prefix_last_logprobs,
            meta,
        )


@dataclass
class VerlMCoreIntegration:
    config: PrefixSharingConfig
    backend: Any | None = None
    batch_adapter: VerlMCoreBatchAdapter = field(init=False)

    def __post_init__(self) -> None:
        self.batch_adapter = VerlMCoreBatchAdapter(self.config)

    def install(self, model_config: Any | None = None) -> PatchHandle:
        self.config.validate(model_config=model_config, integrate_mode="verl_megatron_actor")
        self._ensure_verl_importable()
        backend = self.backend or TorchReferenceBackend()
        return MegatronAttentionIntegration(config=self.config, backend=backend).install(
            model_config=model_config
        )

    @staticmethod
    def _ensure_verl_importable() -> None:
        try:
            importlib.import_module("verl")
        except ModuleNotFoundError as exc:
            raise IntegrationUnavailable("verl is not importable in this environment") from exc


def enable_prefix_sharing(
    config: PrefixSharingConfig,
    *,
    model_config: Any | None = None,
    backend: Any | None = None,
) -> PatchHandle:
    """Install Phase 1 prefix-sharing patches for the verl + Megatron path."""

    return VerlMCoreIntegration(config=config, backend=backend).install(model_config=model_config)


@contextmanager
def prefix_sharing_enabled(
    config: PrefixSharingConfig,
    *,
    model_config: Any | None = None,
    backend: Any | None = None,
) -> Iterator[PatchHandle]:
    """Context manager wrapper around :func:`enable_prefix_sharing`."""

    handle = enable_prefix_sharing(config, model_config=model_config, backend=backend)
    try:
        yield handle
    finally:
        handle.disable()


def prepare_megatron_actor_micro_batch(
    batch: Any,
    actor_config: Any,
    model_config: Any,
    *,
    backend: Any | None = None,
) -> tuple[Any, MegatronActorPreparedMicroBatch | None]:
    """Trim one verl Megatron actor micro-batch in-place for prefix sharing.

    The framework-facing contract is intentionally small: dependency/verl calls
    this once after obtaining the micro-batch and then opens
    :func:`megatron_actor_prefix_sharing_context` around its existing forward
    call. Unsupported or disabled cases return ``(batch, None)``.
    """

    config = prefix_sharing_config_from_verl(actor_config)
    if not config.enabled:
        return batch, None
    config.validate(model_config=model_config, integrate_mode="verl_megatron_actor")
    if not _read_actor_bool(actor_config, "megatron.use_remove_padding", False):
        raise RuntimeError("prefix sharing phase 1 requires verl megatron.use_remove_padding=True")
    if "multi_modal_inputs" in batch:
        raise RuntimeError("prefix sharing phase 1 supports only text-only actor micro-batches")

    attention_mask = batch["attention_mask"].to(bool)
    input_ids = batch["input_ids"]
    position_ids = batch["position_ids"]
    if attention_mask.dim() != 2 or input_ids.dim() != 2 or position_ids.dim() != 2:
        raise RuntimeError("prefix sharing phase 1 expects 2D input_ids/attention_mask/position_ids")

    valid_indices = [attention_mask[row].nonzero(as_tuple=False).flatten() for row in range(input_ids.shape[0])]
    sequences = [input_ids[row, indices].detach().cpu().tolist() for row, indices in enumerate(valid_indices)]
    meta = PrefixSharingPlanner(config).plan(sequences)
    if not meta.has_sharing:
        return batch, None

    prepared_batch = _clone_batch(batch)
    new_attention_mask = attention_mask.clone()
    new_attention_mask[:] = False
    new_input_ids = input_ids.clone()
    new_position_ids = position_ids.clone()
    kept_position_rows = []

    for row, indices in enumerate(valid_indices):
        keep_start, keep_end = meta.input_keep_ranges[row]
        kept_indices = indices[keep_start:keep_end]
        new_attention_mask[row, kept_indices] = True
        kept_position_rows.append(position_ids[row, kept_indices])

    prepared_batch["input_ids"] = new_input_ids
    prepared_batch["attention_mask"] = new_attention_mask
    prepared_batch["position_ids"] = new_position_ids

    restore_positions = []
    for spec in meta.prefix_last_restore:
        provider_indices = valid_indices[spec.provider_idx_in_batch]
        reuse_indices = valid_indices[spec.reuse_idx_in_batch]
        restore_positions.append(
            DensePrefixLastRestoreSpec(
                reuse_idx_in_batch=spec.reuse_idx_in_batch,
                provider_idx_in_batch=spec.provider_idx_in_batch,
                provider_dense_pos=int(provider_indices[spec.provider_prefix_last_pos].item()),
                reuse_dense_pos=int(reuse_indices[spec.reuse_first_suffix_label_pos].item()),
            )
        )

    kept_position_ids = _concat_tensors(kept_position_rows)
    return prepared_batch, MegatronActorPreparedMicroBatch(
        meta=meta,
        backend=backend or TorchReferenceBackend(),
        kept_position_ids=kept_position_ids,
        restore_positions=restore_positions,
    )


@contextmanager
def megatron_actor_prefix_sharing_context(
    prepared: MegatronActorPreparedMicroBatch | None,
) -> Iterator[PrefixSharingRuntimeContext | None]:
    if prepared is None:
        with nullcontext(None) as ctx:
            yield ctx
        return
    with prefix_sharing_context(
        prepared.meta,
        backend=prepared.backend,
        kept_position_ids=prepared.kept_position_ids,
        restore_positions=prepared.restore_positions,
    ) as ctx:
        yield ctx


def restore_megatron_actor_log_probs(
    logits: Any,
    labels: Any,
    log_probs: Any,
    vocab_parallel_log_probs_fn: Any,
) -> Any:
    """Restore reuser first-suffix logprob from provider prefix-last logits."""

    ctx = current_prefix_sharing_context()
    if ctx is None or not ctx.restore_positions:
        return log_probs
    restored = log_probs.clone()
    for spec in ctx.restore_positions:
        provider_logits = logits[
            spec.provider_idx_in_batch : spec.provider_idx_in_batch + 1,
            spec.provider_dense_pos : spec.provider_dense_pos + 1,
            :,
        ]
        reuse_label = labels[
            spec.reuse_idx_in_batch : spec.reuse_idx_in_batch + 1,
            spec.reuse_dense_pos : spec.reuse_dense_pos + 1,
        ]
        restored_value = vocab_parallel_log_probs_fn(provider_logits, reuse_label)
        restored[spec.reuse_idx_in_batch, spec.reuse_dense_pos] = restored_value.reshape(())
    return restored


def prefix_sharing_config_from_verl(actor_config: Any) -> PrefixSharingConfig:
    raw = _read_actor_value(actor_config, "prefix_sharing", None)
    if raw is None:
        raw = _read_actor_value(actor_config, "prefix_sharing_config", None)
    if raw is None or raw is False:
        return PrefixSharingConfig(enabled=False)
    if raw is True:
        return PrefixSharingConfig(enabled=True)
    values = _to_plain_mapping(raw)
    return PrefixSharingConfig(**values)


def _clone_batch(batch: Any) -> Any:
    if hasattr(batch, "clone"):
        return batch.clone()
    if hasattr(batch, "copy"):
        return batch.copy()
    return dict(batch)


def _concat_tensors(tensors: Sequence[Any]) -> Any:
    if not tensors:
        raise RuntimeError("prefix sharing produced an empty packed query")
    first = tensors[0]
    torch = importlib.import_module("torch")
    return torch.cat([tensor.to(first.device) for tensor in tensors], dim=0)


def _read_actor_bool(config: Any, dotted_name: str, default: bool) -> bool:
    value = _read_actor_value(config, dotted_name, default)
    return bool(value)


def _read_actor_value(config: Any, dotted_name: str, default: Any) -> Any:
    current = config
    for part in dotted_name.split("."):
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(part, default)
        else:
            getter = getattr(current, "get", None)
            if callable(getter):
                current = getter(part, default)
            else:
                current = getattr(current, part, default)
    return current


def _to_plain_mapping(raw: Any) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(raw):
            return dict(OmegaConf.to_container(raw, resolve=True))
    except ModuleNotFoundError:
        pass
    if isinstance(raw, Mapping):
        return dict(raw)
    if hasattr(raw, "__dict__"):
        return {
            key: value
            for key, value in vars(raw).items()
            if not key.startswith("_")
        }
    raise TypeError("prefix_sharing config must be a bool, mapping, or OmegaConf object")
