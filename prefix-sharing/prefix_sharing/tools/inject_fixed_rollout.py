"""Inject fixed rollout data from a JSON file, replacing generate_sequences output.

Usage in ray_trainer.py's fit() method, right before the training loop:

    from prefix_sharing.tools.inject_fixed_rollout import patch_fixed_rollout
    patch_fixed_rollout(self, json_path="/path/to/your_data.json")

The JSON format expected:
{
    "outputs": {
        "input_ids":  [[...], [...], ...],
        "attention_mask": [[...], ...],
        "position_ids": [[...], ...],
        "responses": [[...], ...],
        "prompts": [[...], ...],
        "token_level_rewards": [[...], ...],
        "response_mask": [[...], ...],
        "rm_scores": [[...], ...],
        "rollout_log_probs": [[...], ...]
    }
}
"""

import json
import logging

import torch

logger = logging.getLogger(__name__)


def _load_json_to_dataproto(json_path: str):
    """Load a JSON file and convert to DataProto."""
    from verl.protocol import DataProto

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"[FixedRollout] JSON file not found: {json_path}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[FixedRollout] Invalid JSON in {json_path}: {e}")

    if "outputs" not in raw:
        raise RuntimeError(f"[FixedRollout] Missing key 'outputs' in {json_path}")

    outputs = raw["outputs"]

    # Values may be JSON strings like "[[1,2],[3,4]]" instead of actual lists
    def _ensure_list(val):
        if isinstance(val, str):
            return json.loads(val)
        return val

    outputs = {k: _ensure_list(v) for k, v in outputs.items()}

    def _pad_long(seqs, pad_id=0):
        """Pad list of int lists to rectangular LongTensor."""
        max_len = max(len(s) for s in seqs)
        tensor = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len, dtype=torch.long)
        for i, s in enumerate(seqs):
            tensor[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return tensor

    def _pad_float(seqs, pad_val=0.0):
        """Pad list of float lists to rectangular FloatTensor."""
        max_len = max(len(s) for s in seqs)
        tensor = torch.full((len(seqs), max_len), pad_val, dtype=torch.float32)
        for i, s in enumerate(seqs):
            tensor[i, : len(s)] = torch.tensor(s, dtype=torch.float32)
        return tensor

    batch = {
        "input_ids": _pad_long(outputs["input_ids"]),
        "attention_mask": _pad_long(outputs["attention_mask"]),
        "position_ids": _pad_long(outputs["position_ids"]),
        "responses": _pad_long(outputs["responses"]),
        "prompts": _pad_long(outputs["prompts"]),
    }

    # Optional float fields
    for key in ("token_level_rewards", "response_mask", "rm_scores", "rollout_log_probs"):
        if key in outputs:
            batch[key] = _pad_float(outputs[key])

    # Build sequences = prompts + responses if not present
    if "sequences" not in outputs:
        batch["sequences"] = torch.cat([batch["prompts"], batch["responses"]], dim=1)

    data = DataProto.from_dict(batch)
    logger.warning(
        f"[FixedRollout] Loaded {data.batch['input_ids'].shape[0]} samples from {json_path}"
    )
    return data


def patch_fixed_rollout(trainer, json_path: str):
    """Monkey-patch generate_sequences to return fixed data.

    Patches both actor_rollout_wg.generate_sequences and
    async_rollout_manager.generate_sequences (if present), since the
    trainer runs with async_rollout_mode=True by default.

    Args:
        trainer: The RayPPOTrainer / RayMegatronTrainer instance (self in fit()).
        json_path: Absolute path to the JSON file.
    """
    fixed_data = _load_json_to_dataproto(json_path)

    def _patched(batch, **kwargs):
        logger.warning("[FixedRollout] Returning fixed rollout data, skipping generation.")
        fixed_data.meta_info["timing"] = {}
        return fixed_data

    trainer.actor_rollout_wg.generate_sequences = _patched
    logger.warning("[FixedRollout] Patched actor_rollout_wg.generate_sequences.")

    if hasattr(trainer, "async_rollout_manager") and trainer.async_rollout_manager is not None:
        trainer.async_rollout_manager.generate_sequences = _patched
        logger.warning("[FixedRollout] Patched async_rollout_manager.generate_sequences.")
