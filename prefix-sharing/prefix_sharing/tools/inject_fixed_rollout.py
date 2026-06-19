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

import torch


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

    # verl>=0.8.0 的 trainer.fit() 会硬索引 batch.non_tensor_batch["multi_modal_inputs"]
    # （ray_trainer.py:1483，纯文本场景也走这行）。纯文本/虚拟注入没有这个字段会 KeyError。
    # v070 trainer 不碰这个字段，无需添加。这里只在 verl>=0.8.0 时填一个空字典占位
    # （下游 'image_grid_thw' 检查会对空 dict continue 跳过，行为正确）。
    non_tensors = None
    try:
        import verl
        from packaging.version import parse as parse_version

        if parse_version(verl.__version__) >= parse_version("0.8.0"):
            import numpy as np

            n_samples = batch["input_ids"].shape[0]
            non_tensors = {
                "multi_modal_inputs": np.array([{}] * n_samples, dtype=object)
            }
            print(
                f"[FixedRollout] verl={verl.__version__}, filled empty "
                f"'multi_modal_inputs' placeholder for {n_samples} text-only samples."
            )
    except Exception as e:  # import 失败或版本探测失败，退回到不填占位
        print(f"[FixedRollout] skip 'multi_modal_inputs' placeholder: {e}")

    data = DataProto.from_dict(batch, non_tensors=non_tensors)
    print(
        f"[FixedRollout] Loaded {data.batch['input_ids'].shape[0]} samples from {json_path}"
    )
    return data


def patch_fixed_rollout(trainer, json_path: str, num_workers: int = 8):
    """Monkey-patch generate_sequences to return fixed data.

    Patches both actor_rollout_wg.generate_sequences and
    async_rollout_manager.generate_sequences (if present), since the
    trainer runs with async_rollout_mode=True by default.

    Args:
        trainer: The RayPPOTrainer / RayMegatronTrainer instance (self in fit()).
        json_path: Absolute path to the JSON file.
        num_workers: Number of agent loop workers (default 8, matching verl's
            ``actor_rollout_ref.rollout.agent.num_workers``). The fixed data
            will be auto-padded to a multiple of this value to avoid
            DataProto.chunk() equal-division assertion errors.
    """
    fixed_data = _load_json_to_dataproto(json_path)

    # Pad to make divisible by num_workers (prevent chunk() AssertionError)
    n = len(fixed_data)
    remainder = n % num_workers
    if remainder != 0:
        pad_size = num_workers - remainder
        fixed_data.padding(pad_size, "last")
        print(
            f"[FixedRollout] Padded from {n} to {n + pad_size} samples"
            f" (divisible by {num_workers})."
        )

    def _patched(batch, **kwargs):
        print("[FixedRollout] Returning fixed rollout data, skipping generation.")
        fixed_data.meta_info["timing"] = {}
        return fixed_data

    trainer.actor_rollout_wg.generate_sequences = _patched
    print("[FixedRollout] Patched actor_rollout_wg.generate_sequences.")

    if hasattr(trainer, "async_rollout_manager") and trainer.async_rollout_manager is not None:
        trainer.async_rollout_manager.generate_sequences = _patched
        print("[FixedRollout] Patched async_rollout_manager.generate_sequences.")
