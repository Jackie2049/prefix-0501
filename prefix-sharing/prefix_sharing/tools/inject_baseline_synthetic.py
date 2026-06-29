"""Inject independent synthetic sequences for GEMM precision baseline.

Creates ``num_seq`` distinct sequences from a JSON token source, optionally
shuffles each sequence's internal token order with a deterministic seed, then
stacks each sequence ``stack`` times for batch-size-controlled comparison.

Usage::

    # Env vars:
    PREFIX_SHARING_BASELINE_SYNTHETIC=/path/to/data.json
    PREFIX_SHARING_BASELINE_NUM_SEQ=4      # distinct sequences
    PREFIX_SHARING_BASELINE_STACK=3        # copies per sequence
    PREFIX_SHARING_BASELINE_SEED=42        # shuffle seed

    # Single-copy run:
    PREFIX_SHARING_BASELINE_STACK=1 python train.py ...

    # Stacked run (same data, same shuffle → identical copies):
    PREFIX_SHARING_BASELINE_STACK=3 python train.py ...
"""

import json
import random

import torch


def _load_base_tokens(json_path: str) -> list[int]:
    """Load JSON and return the longest unpadded token sequence."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"[BaselineSynthetic] JSON not found: {json_path}")

    outputs = raw.get("outputs", raw)

    def _ensure_list(v):
        return json.loads(v) if isinstance(v, str) else v

    for k in list(outputs.keys()):
        outputs[k] = _ensure_list(outputs[k])

    ids = outputs.get("input_ids")
    if not ids:
        raise RuntimeError("[BaselineSynthetic] No 'input_ids' in JSON")
    pos = outputs.get("position_ids")
    if pos is None:
        raise RuntimeError("[BaselineSynthetic] No 'position_ids' in JSON")

    def _valid_len(p):
        return max(p) + 1 if p else 0

    best = max(range(len(ids)), key=lambda i: _valid_len(pos[i]))
    best_pos = pos[best]

    try:
        first_one = next(i for i, p in enumerate(best_pos) if p == 1)
    except StopIteration:
        raise RuntimeError("[BaselineSynthetic] Sample has no position_id=1")

    start = first_one - 1
    end = max(i for i, p in enumerate(best_pos) if p > 0) + 1
    return ids[best][start:end]


def _shuffle_tokens(tokens: list[int], seed: int) -> list[int]:
    """Deterministically shuffle token order using ``seed``."""
    rng = random.Random(seed)
    indices = list(range(len(tokens)))
    rng.shuffle(indices)
    return [tokens[i] for i in indices]


def _build_baseline_batch(
    base_tokens: list[int],
    num_seq: int,
    stack: int,
    max_prompt_length: int,
    max_response_length: int,
    seed: int = 42,
    pad_id: int = 151643,
) -> dict:
    """Create a synthetic batch with independent sequences.

    Each of the ``num_seq`` sequences gets a deterministic shuffle (seed =
    ``seed + seq_index``), then is stacked ``stack`` times.

    Total batch size = ``num_seq * stack``.
    """
    R = max_response_length
    P = max_prompt_length
    seq_len = P + R
    total_tokens_needed = num_seq * seq_len

    if len(base_tokens) < total_tokens_needed:
        raise RuntimeError(
            f"[BaselineSynthetic] Need {total_tokens_needed} tokens, "
            f"have {len(base_tokens)}."
        )

    # Split into num_seq segments, shuffle each
    sequences: list[list[int]] = []
    for i in range(num_seq):
        seg = base_tokens[i * seq_len : (i + 1) * seq_len]
        seg = _shuffle_tokens(seg, seed + i)
        sequences.append(seg)

    total_bs = num_seq * stack
    print(
        f"[BaselineSynthetic] num_seq={num_seq} stack={stack} "
        f"total_bs={total_bs} P={P} R={R} seed={seed}"
    )

    # Build 2D tensors [total_bs, P+R]
    input_ids = torch.full((total_bs, seq_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(total_bs, seq_len, dtype=torch.long)
    position_ids = torch.zeros(total_bs, seq_len, dtype=torch.long)
    prompts = torch.full((total_bs, P), pad_id, dtype=torch.long)
    responses = torch.full((total_bs, R), pad_id, dtype=torch.long)
    response_mask = torch.zeros(total_bs, R, dtype=torch.float32)

    for si in range(num_seq):
        seq_data = sequences[si]
        pl = P  # seq_len
        rl = R
        tokens_p = seq_data[:pl]
        tokens_r = seq_data[pl : pl + rl]
        for ci in range(stack):
            bi = si * stack + ci  # flat batch index
            # Prompt: right-padded (left-aligned in verl convention)
            prompts[bi, :len(tokens_p)] = torch.tensor(tokens_p, dtype=torch.long)
            # Response: right-padded
            responses[bi, :len(tokens_r)] = torch.tensor(tokens_r, dtype=torch.long)
            # Combined input_ids
            input_ids[bi, :len(tokens_p)] = torch.tensor(tokens_p, dtype=torch.long)
            input_ids[bi, P : P + len(tokens_r)] = torch.tensor(tokens_r, dtype=torch.long)
            # Attention mask
            attention_mask[bi, :len(tokens_p)] = 1
            attention_mask[bi, P : P + len(tokens_r)] = 1
            # Position IDs: prompt=0..pl-1, response=pl..pl+rl-1
            position_ids[bi, :len(tokens_p)] = torch.arange(len(tokens_p))
            position_ids[bi, P : P + len(tokens_r)] = torch.arange(len(tokens_r)) + pl
            # Response mask
            response_mask[bi, :len(tokens_r)] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prompts": prompts,
        "responses": responses,
        "response_mask": response_mask,
        "token_level_rewards": response_mask.clone(),
        "rollout_log_probs": torch.zeros(total_bs, R, dtype=torch.float32),
        "rm_scores": torch.ones(total_bs, 1, dtype=torch.float32),
    }


def patch_baseline_synthetic(
    trainer,
    json_path: str,
    batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    num_seq: int = 1,
    stack: int = 1,
    seed: int = 42,
    num_workers: int = 8,
):
    """Monkey-patch generate_sequences to return baseline synthetic data.

    Args:
        trainer: RayPPOTrainer instance.
        json_path: Path to JSON with at least one long input_ids entry.
        batch_size: gen_batch_size from config.
        max_prompt_length: Prompt length.
        max_response_length: Response length.
        num_seq: Number of distinct sequences (env: BASELINE_NUM_SEQ).
        stack: How many times to stack each sequence (env: BASELINE_STACK).
        seed: Random seed for shuffle (env: BASELINE_SEED).
        num_workers: Agent loop workers for chunk() divisibility.
    """
    from verl.protocol import DataProto

    base_tokens = _load_base_tokens(json_path)
    batch = _build_baseline_batch(
        base_tokens=base_tokens,
        num_seq=num_seq,
        stack=stack,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        seed=seed,
    )

    # Multi-modal placeholder (verl >= 0.8.0 compatibility)
    non_tensors = None
    try:
        import verl
        from packaging.version import parse as parse_version

        if parse_version(verl.__version__) > parse_version("0.7.99"):
            import numpy as np

            n_samples = batch["input_ids"].shape[0]
            non_tensors = {
                "multi_modal_inputs": np.array([{}] * n_samples, dtype=object)
            }
    except Exception:
        pass

    fixed_data = DataProto.from_dict(batch, non_tensors=non_tensors)

    # Pad to be divisible by num_workers
    n = len(fixed_data)
    rem = n % num_workers
    if rem:
        pad_size = num_workers - rem
        fixed_data.padding(pad_size, "last")
        print(f"[BaselineSynthetic] Padded {n} -> {n + pad_size}")

    total_bs = num_seq * stack
    def _patched(batch, **kwargs):
        print(
            f"[BaselineSynthetic] Returning synthetic baseline data "
            f"(num_seq={num_seq}, stack={stack}, total_bs={total_bs}, "
            f"P={max_prompt_length}, R={max_response_length}, seed={seed})."
        )
        fixed_data.meta_info["timing"] = {}
        return fixed_data

    trainer.actor_rollout_wg.generate_sequences = _patched
    print("[BaselineSynthetic] Patched actor_rollout_wg.generate_sequences.")

    if hasattr(trainer, "async_rollout_manager") and trainer.async_rollout_manager is not None:
        trainer.async_rollout_manager.generate_sequences = _patched
        print("[BaselineSynthetic] Patched async_rollout_manager.generate_sequences.")
