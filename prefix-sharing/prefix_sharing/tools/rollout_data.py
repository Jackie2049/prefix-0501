"""Generate GRPO-style agentic step-replay benchmark data.

Produces a deterministic, repo-committable dataset that exercises
prefix-sharing under a realistic agentic-RL workload.

Two data views are generated from the same underlying trajectories:

1. **token_mode** — raw multi-step agentic trajectories, mimicking what a
   real GRPO rollout would produce. Each *group* has one shared base
   prompt ``p1`` and ``group_size`` (= n in GRPO) trajectories, each a
   multi-step interaction::

       [p1 r1 p2 r2 p3 r3 ...]

   ``p1`` is shared across the whole group; ``r_k`` / ``p_k`` (k>=2) differ
   per trajectory.

2. **step_mode** — step-wise replay derived from the token-mode
   trajectories. Each trajectory's growing prefix becomes a separate
   training sample, so the chain of prefix sharing is explicit::

       step 0:  [p1 r1]
       step 1:  [p1 r1 p2 r2]
       step 2:  [p1 r1 p2 r2 p3 r3]

   step k's full sequence equals step k+1's prefix → natural chain reuse.
   All step-0 samples in a group share ``p1`` → group-level reuse.

3. **manifest** — documents the expected prefix-sharing structure (which
   samples are providers, expected shared-prefix lengths) so the benchmark
   analyzer can verify that prefix sharing behaved as expected.

The step_mode file follows the ``inject_fixed_rollout`` JSON contract
(keys: ``input_ids``, ``attention_mask``, ``position_ids``, ``prompts``,
``responses``, ``response_mask``).

Usage::

    python -m prefix_sharing.tools.rollout_data --out-dir prefix_sharing/tools/data
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    """Shape of the synthetic agentic dataset."""

    num_groups: int = 4
    """Number of GRPO groups (distinct base prompts / tasks)."""

    group_size: int = 8
    """Trajectories per group (n in GRPO)."""

    num_steps: int = 3
    """Interaction steps per trajectory (p1 r1 p2 r2 ... pK rK)."""

    base_prompt_len: int = 128
    """Token length of p1 (the shared task prompt)."""

    step_prompt_len: int = 48
    """Token length of p2, p3, ... (environment feedback per step)."""

    step_response_len: int = 32
    """Token length of r1, r2, ... (model responses)."""

    vocab_size: int = 32000
    vocab_min: int = 5
    """Token ids are drawn from [vocab_min, vocab_size) to avoid reserved ids
    (0=pad, 1-4=BOS/EOS/PAD variants used by some tokenizers)."""

    pad_token_id: int = 0
    seed: int = 42


# ---------------------------------------------------------------------------
# token-mode generation
# ---------------------------------------------------------------------------


def _gen_tokens(rng: random.Random, n: int, config: DatasetConfig) -> list[int]:
    return [rng.randint(config.vocab_min, config.vocab_size - 1) for _ in range(n)]


def generate_token_mode(config: DatasetConfig) -> list[dict[str, Any]]:
    """Generate raw multi-step agentic trajectories.

    Returns a list of groups. Each group carries a shared ``base_prompt``
    (p1) and ``group_size`` trajectories; each trajectory has ``num_steps``
    ``(prompt, response)`` pairs.
    """
    rng = random.Random(config.seed)
    groups: list[dict[str, Any]] = []

    for g in range(config.num_groups):
        base_prompt = _gen_tokens(rng, config.base_prompt_len, config)
        trajectories: list[dict[str, Any]] = []
        for t in range(config.group_size):
            steps: list[dict[str, list[int]]] = []
            for k in range(config.num_steps):
                if k == 0:
                    prompt = list(base_prompt)  # p1 = shared base prompt
                else:
                    prompt = _gen_tokens(rng, config.step_prompt_len, config)
                response = _gen_tokens(rng, config.step_response_len, config)
                steps.append({"prompt": prompt, "response": response})
            trajectories.append({"traj_id": t, "steps": steps})
        groups.append({"group_id": g, "base_prompt": base_prompt, "trajectories": trajectories})

    return groups


# ---------------------------------------------------------------------------
# step-wise replay
# ---------------------------------------------------------------------------


def build_step_mode(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand trajectories into step-wise replay training samples.

    For a trajectory with steps [(p1,r1), (p2,r2), (p3,r3)] produces::

        step 0: prompt=[p1],               response=[r1], seq=[p1 r1]
        step 1: prompt=[p1 r1 p2],         response=[r2], seq=[p1 r1 p2 r2]
        step 2: prompt=[p1 r1 p2 r2 p3],   response=[r3], seq=[p1 r1 p2 r2 p3 r3]

    The model trains on ``response`` (r_k); everything before it is context.
    """
    samples: list[dict[str, Any]] = []
    for group in groups:
        gid = group["group_id"]
        for traj in group["trajectories"]:
            tid = traj["traj_id"]
            context: list[int] = []
            for k, step in enumerate(traj["steps"]):
                context = context + step["prompt"]  # append p_k
                response = step["response"]  # r_k
                sequence = context + response
                samples.append(
                    {
                        "group_id": gid,
                        "traj_id": tid,
                        "step": k,
                        "sequence": sequence,
                        "prompt": list(context),
                        "response": list(response),
                    }
                )
                context = sequence  # r_k becomes part of next step's context
    return samples


# ---------------------------------------------------------------------------
# inject_fixed_rollout JSON format
# ---------------------------------------------------------------------------


def _pad_right(rows: list[list[int]], pad_val: int) -> list[list[int]]:
    max_len = max(len(r) for r in rows)
    return [r + [pad_val] * (max_len - len(r)) for r in rows]


def build_inject_outputs(
    samples: list[dict[str, Any]], config: DatasetConfig
) -> dict[str, Any]:
    """Convert step-replay samples to the inject_fixed_rollout JSON contract."""
    sequences = [s["sequence"] for s in samples]
    prompts = [s["prompt"] for s in samples]
    responses = [s["response"] for s in samples]
    max_seq_len = max(len(seq) for seq in sequences)

    input_ids = _pad_right(sequences, config.pad_token_id)
    attention_mask = _pad_right([[1] * len(seq) for seq in sequences], 0)
    position_ids = _pad_right([list(range(len(seq))) for seq in sequences], 0)

    # response_mask: 1 on response tokens, 0 elsewhere (prompt + padding)
    response_mask: list[list[int]] = []
    for s in samples:
        plen = len(s["prompt"])
        rlen = len(s["response"])
        mask = [0] * plen + [1] * rlen
        mask += [0] * (max_seq_len - len(mask))
        response_mask.append(mask)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "prompts": _pad_right(prompts, config.pad_token_id),
        "responses": _pad_right(responses, config.pad_token_id),
        "response_mask": response_mask,
    }


# ---------------------------------------------------------------------------
# manifest (expected prefix-sharing structure)
# ---------------------------------------------------------------------------


def build_manifest(
    groups: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    config: DatasetConfig,
) -> dict[str, Any]:
    """Document expected prefix-sharing structure for analyzer verification.

    Records, per trajectory, the chain of step-replay samples and the
    expected shared-prefix length between consecutive steps. Also records
    the group-level shared base-prompt length.
    """
    # Index samples by (group_id, traj_id) → list sorted by step
    by_traj: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for idx, s in enumerate(samples):
        key = (s["group_id"], s["traj_id"])
        by_traj.setdefault(key, []).append({**s, "_idx": idx})
    for key in by_traj:
        by_traj[key].sort(key=lambda x: x["step"])

    manifest_groups: list[dict[str, Any]] = []
    for group in groups:
        gid = group["group_id"]
        base_prompt_len = len(group["base_prompt"])
        chains: list[dict[str, Any]] = []
        for traj in group["trajectories"]:
            tid = traj["traj_id"]
            chain_samples: list[dict[str, Any]] = []
            traj_samples = by_traj[(gid, tid)]
            for pos, s in enumerate(traj_samples):
                entry: dict[str, Any] = {
                    "sample_idx": s["_idx"],
                    "step": s["step"],
                    "seq_len": len(s["sequence"]),
                    "prompt_len": len(s["prompt"]),
                    "response_len": len(s["response"]),
                    "is_chain_provider": pos == 0,
                }
                if pos > 0:
                    prev = chain_samples[-1]
                    entry["expected_reuse_sample_idx"] = prev["sample_idx"]
                    entry["expected_shared_prefix_len"] = prev["seq_len"]
                chain_samples.append(entry)
            chains.append({"traj_id": tid, "samples": chain_samples})
        manifest_groups.append(
            {
                "group_id": gid,
                "base_prompt_len": base_prompt_len,
                "group_shared_prefix_len": base_prompt_len,
                "chains": chains,
            }
        )

    return {
        "config": asdict(config),
        "num_groups": config.num_groups,
        "group_size": config.group_size,
        "num_steps": config.num_steps,
        "total_samples": len(samples),
        "groups": manifest_groups,
    }


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------


def validate_prefix_structure(samples: list[dict[str, Any]]) -> None:
    """Assert that chain reuse holds: step k+1's prefix == step k's sequence."""
    by_traj: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for s in samples:
        by_traj.setdefault((s["group_id"], s["traj_id"]), []).append(s)
    for key, traj_samples in by_traj.items():
        traj_samples.sort(key=lambda x: x["step"])
        for k in range(1, len(traj_samples)):
            prev_seq = traj_samples[k - 1]["sequence"]
            cur_seq = traj_samples[k]["sequence"]
            assert cur_seq[: len(prev_seq)] == prev_seq, (
                f"chain reuse broken at group={key[0]} traj={key[1]} "
                f"step={k}: prefix mismatch"
            )
    print(f"[rollout_data] validated chain reuse across {len(by_traj)} trajectories")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GRPO-style agentic step-replay benchmark data."
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for JSON files.")
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8, help="n in GRPO")
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--base-prompt-len", type=int, default=128)
    parser.add_argument("--step-prompt-len", type=int, default=48)
    parser.add_argument("--step-response-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = DatasetConfig(
        num_groups=args.num_groups,
        group_size=args.group_size,
        num_steps=args.num_steps,
        base_prompt_len=args.base_prompt_len,
        step_prompt_len=args.step_prompt_len,
        step_response_len=args.step_response_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )

    import os

    os.makedirs(args.out_dir, exist_ok=True)

    groups = generate_token_mode(config)
    samples = build_step_mode(groups)
    validate_prefix_structure(samples)

    # token_mode.json — raw trajectories
    token_mode = {
        "config": asdict(config),
        "groups": groups,
    }
    token_mode_path = os.path.join(args.out_dir, "token_mode.json")
    with open(token_mode_path, "w") as f:
        json.dump(token_mode, f)
    print(f"[rollout_data] wrote {token_mode_path} ({config.num_groups} groups x "
          f"{config.group_size} trajs x {config.num_steps} steps)")

    # step_mode.json — inject_fixed_rollout format
    outputs = build_inject_outputs(samples, config)
    inject_data = {"config": asdict(config), "outputs": outputs}
    step_mode_path = os.path.join(args.out_dir, "step_mode.json")
    with open(step_mode_path, "w") as f:
        json.dump(inject_data, f)
    max_seq = max(len(seq) for seq in outputs["input_ids"])
    print(f"[rollout_data] wrote {step_mode_path} ({len(samples)} samples, "
          f"max_seq_len={max_seq})")

    # manifest.json — expected prefix-sharing structure
    manifest = build_manifest(groups, samples, config)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[rollout_data] wrote {manifest_path}")


if __name__ == "__main__":
    main()
