#!/usr/bin/env python3
"""PrefixGrouper Competitive Analysis: Standalone Benchmark with Qwen2.5-0.5B

This script directly uses the PrefixGrouper package (pip install prefix-grouper)
to benchmark block-causal attention optimization vs standard full attention.

Setup: Qwen2.5-0.5B-Instruct model on 1 GPU (FSDP not needed - model is small),
simulating verl GRPO training pattern with shared prefixes.

Key metrics:
- Forward-only (compute_log_prob equivalent)
- Forward + backward (update_actor equivalent)
- Token savings ratio
- Precision comparison (PG ON vs OFF)

Usage: python scripts/pg_competitive_benchmark.py
"""

import os
import sys
import time
import json
import torch
import numpy as np
from dataclasses import dataclass

# ===== Configuration =====
MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen2.5-0.5B-Instruct")
N_VALUES = [4, 8]           # Number of responses per prompt (rollout.n)
PREFIX_LEN = 256             # Prompt length (realistic GRPO scenario)
SUFFIX_LEN = 128             # Response length
N_RUNS = 5                   # Number of timing runs for averaging

# ===== Load Model =====
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

print("Loading Qwen2.5-0.5B-Instruct model...")
config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=device,
    local_files_only=True,
)
model.eval()

vocab_size = config.vocab_size
hidden_size = config.hidden_size
num_layers = config.num_hidden_layers
num_attn_heads = config.num_attention_heads
num_kv_heads = config.num_key_value_heads
head_dim = hidden_size // num_attn_heads

print(f"Model: Qwen2.5-0.5B ({num_layers} layers, {hidden_size} hidden, "
      f"{num_attn_heads} attn heads, {num_kv_heads} kv heads)")
print(f"Config: prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}, n_values={N_VALUES}")

# ===== Import PrefixGrouper =====
from prefix_grouper import PrefixGrouper

# ===== Apply PG Monkey Patch =====
from verl.models.transformers.monkey_patch import apply_prefix_grouper_patch
apply_prefix_grouper_patch()
print("PrefixGrouper monkey patch applied")

# ===== Helper Functions =====

def generate_random_ids(n_seqs, total_len):
    """Generate random token IDs for benchmarking."""
    return torch.randint(0, vocab_size, (n_seqs, total_len), device=device)


def build_prefix_mask(prefix_len):
    """Build mask for prefix tokens (all 1s)."""
    return torch.ones(1, prefix_len, dtype=torch.bool, device=device)


def build_suffix_mask(n_seqs, suffix_len):
    """Build mask for suffix tokens (all 1s)."""
    return torch.ones(n_seqs, suffix_len, dtype=torch.bool, device=device)


def build_pg_inputs(n, prefix_len, suffix_len):
    """Build PrefixGrouper inputs for n responses sharing a prefix."""
    # Prefix mask: 1 sequence with prefix_len tokens
    prefix_mask = build_prefix_mask(prefix_len)

    # Suffix mask: n sequences with suffix_len tokens
    suffix_mask = build_suffix_mask(n, suffix_len)

    # Create PrefixGrouper object
    pg = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prefix_mask,
        suffix_mask=suffix_mask,
        group_sizes=[n],
        padding_mode="right",
        device=device,
    )

    # Build position_ids for PG
    num_samples = len(pg.group_info)
    max_len = pg.padding_mask.size(1)
    position_ids = torch.zeros(num_samples, max_len, dtype=torch.long, device=device)

    for i, group in enumerate(pg.group_info):
        p_len = group.prefix_len
        position_ids[i, :p_len] = torch.arange(p_len, device=device)
        cur_pos = p_len
        for s_len in group.suffix_lens:
            if s_len > 0:
                position_ids[i, cur_pos:cur_pos + s_len] = torch.arange(
                    p_len, p_len + s_len, device=device
                )
                cur_pos += s_len

    # Build concatenated input_ids
    prefix_ids = torch.randint(0, vocab_size, (1, prefix_len), device=device)
    suffix_ids = torch.randint(0, vocab_size, (n, suffix_len), device=device)
    prefix_ids_expanded = prefix_ids.expand(n, -1)  # (n, prefix_len)
    prefix_tok_mask = prefix_ids.ne(tokenizer.pad_token_id or 0)
    suffix_tok_mask = suffix_ids.ne(tokenizer.pad_token_id or 0)

    concat_input_ids = pg.concat_input(prefix_ids, prefix_tok_mask, suffix_ids, suffix_tok_mask)
    attention_mask = pg.padding_mask

    return pg, concat_input_ids, attention_mask, position_ids, suffix_ids, suffix_tok_mask


def build_no_pg_inputs(n, prefix_len, suffix_len):
    """Build standard (no PG) inputs: n sequences, each with prefix+suffix."""
    total_len = prefix_len + suffix_len
    # All n sequences share the same prefix tokens, each has different suffix
    prefix_ids = torch.randint(0, vocab_size, (1, prefix_len), device=device)
    suffix_ids = torch.randint(0, vocab_size, (n, suffix_len), device=device)

    # Concatenate prefix + suffix for each sequence
    input_ids = torch.cat([prefix_ids.expand(n, -1), suffix_ids], dim=1)  # (n, total_len)
    attention_mask = torch.ones(n, total_len, dtype=torch.long, device=device)
    position_ids = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(n, -1)

    return input_ids, attention_mask, position_ids


def measure_fn(fn, n_runs=N_RUNS):
    """Measure average time over n_runs."""
    times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        result = fn()
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
    return sum(times) / len(times), result


def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm()).item()


# ===== Warmup =====
print("\nWarmup run...")
warmup_ids = torch.randint(0, vocab_size, (4, PREFIX_LEN + SUFFIX_LEN), device=device)
warmup_mask = torch.ones(4, PREFIX_LEN + SUFFIX_LEN, dtype=torch.long, device=device)
warmup_pos = torch.arange(PREFIX_LEN + SUFFIX_LEN, dtype=torch.long, device=device).unsqueeze(0).expand(4, -1)
with torch.no_grad():
    model(warmup_ids, attention_mask=warmup_mask, position_ids=warmup_pos)
torch.cuda.synchronize()
print("Warmup done.")

# ===== Main Benchmark =====
results = {}

for n in N_VALUES:
    print(f"\n{'='*60}")
    print(f"Testing n={n} (rollout.n={n})")
    print(f"Token savings: {(n-1)/n * PREFIX_LEN/(PREFIX_LEN+SUFFIX_LEN):.1%}")
    print(f"{'='*60}")

    total_len = PREFIX_LEN + SUFFIX_LEN
    token_savings = (n - 1) / n * PREFIX_LEN / total_len

    # --- Build inputs ---
    # No PG: n sequences, each prefix+suffix
    no_pg_ids, no_pg_mask, no_pg_pos = build_no_pg_inputs(n, PREFIX_LEN, SUFFIX_LEN)

    # PG: concatenated format with block-causal attention
    pg, pg_ids, pg_mask, pg_pos, suffix_ids, suffix_tok_mask = build_pg_inputs(n, PREFIX_LEN, SUFFIX_LEN)

    # ===== 1. Forward-only (compute_log_prob equivalent) =====
    print("\n--- Forward-only (inference) ---")

    # No PG forward
    def no_pg_forward():
        with torch.no_grad():
            out = model(no_pg_ids, attention_mask=no_pg_mask, position_ids=no_pg_pos)
        return out.logits

    avg_off_fwd, logits_off = measure_fn(no_pg_forward)

    # PG forward
    def pg_forward():
        with torch.no_grad():
            out = model(pg_ids, attention_mask=pg_mask, position_ids=pg_pos, prefix_grouper=pg)
        return out.logits

    avg_pg_fwd, logits_pg = measure_fn(pg_forward)

    # Precision comparison
    # Extract suffix logits from both outputs
    # No PG: logits[:, PREFIX_LEN:, :] (suffix portion for each sequence)
    # PG: need to split_output to get suffix logits
    pg_prefix_out, pg_prefix_mask, pg_suffix_out_raw, pg_suffix_mask_raw = pg.split_output(
        logits_pg, include_prefix_last=1
    )

    no_pg_suffix_logits = logits_off[:, PREFIX_LEN:, :].float()
    pg_suffix_logits = pg_suffix_out_raw[:, :-1].float()  # Remove last token

    # Adjust lengths if different due to PG padding
    min_len = min(no_pg_suffix_logits.size(1), pg_suffix_logits.size(1))
    cos_sim_fwd = cosine_similarity(
        no_pg_suffix_logits[:, :min_len, :],
        pg_suffix_logits[:, :min_len, :]
    )

    print(f"  No PG forward: {avg_off_fwd*1000:.1f}ms")
    print(f"  PG forward:    {avg_pg_fwd*1000:.1f}ms")
    print(f"  Speedup:       {avg_off_fwd/avg_pg_fwd:.2f}x")
    print(f"  Cosine sim:    {cos_sim_fwd:.6f}")

    # ===== 2. Forward + Backward (update_actor equivalent) =====
    print("\n--- Forward + Backward (training) ---")

    # No PG: forward + backward
    def no_pg_train():
        model.train()
        out = model(no_pg_ids, attention_mask=no_pg_mask, position_ids=no_pg_pos)
        loss = out.logits.sum()  # Simple loss for benchmarking
        loss.backward()
        model.zero_grad()
        model.eval()
        return out.logits

    avg_off_train, _ = measure_fn(no_pg_train)

    # PG: forward + backward
    def pg_train():
        model.train()
        out = model(pg_ids, attention_mask=pg_mask, position_ids=pg_pos, prefix_grouper=pg)
        loss = out.logits.sum()
        loss.backward()
        model.zero_grad()
        model.eval()
        return out.logits

    avg_pg_train, _ = measure_fn(pg_train)

    print(f"  No PG train: {avg_off_train*1000:.1f}ms")
    print(f"  PG train:    {avg_pg_train*1000:.1f}ms")
    print(f"  Speedup:     {avg_off_train/avg_pg_train:.2f}x")

    # ===== 3. Memory comparison =====
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / 1024**3

    # Measure peak memory during forward
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        model(no_pg_ids, attention_mask=no_pg_mask, position_ids=no_pg_pos)
    peak_off = torch.cuda.max_memory_allocated() / 1024**3

    torch.cuda.empty_cache()
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        model(pg_ids, attention_mask=pg_mask, position_ids=pg_pos, prefix_grouper=pg)
    peak_pg = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n--- Memory ---")
    print(f"  No PG peak: {peak_off:.2f} GB")
    print(f"  PG peak:    {peak_pg:.2f} GB")
    print(f"  Memory savings: {(peak_off - peak_pg)/peak_off:.1%}")

    # Store results
    results[n] = {
        "n": n,
        "prefix_len": PREFIX_LEN,
        "suffix_len": SUFFIX_LEN,
        "total_len": total_len,
        "token_savings_pct": token_savings * 100,
        "theoretical_speedup": 1 / (1 - token_savings),
        "forward_no_pg_ms": avg_off_fwd * 1000,
        "forward_pg_ms": avg_pg_fwd * 1000,
        "forward_speedup": avg_off_fwd / avg_pg_fwd,
        "forward_cos_sim": cos_sim_fwd,
        "train_no_pg_ms": avg_off_train * 1000,
        "train_pg_ms": avg_pg_train * 1000,
        "train_speedup": avg_off_train / avg_pg_train,
        "peak_mem_no_pg_gb": peak_off,
        "peak_mem_pg_gb": peak_pg,
        "mem_savings_pct": (peak_off - peak_pg) / peak_off * 100 if peak_off > 0 else 0,
    }

    print(f"\n{'='*60}")
    print(f"Summary for n={n}:")
    print(f"  Forward speedup: {avg_off_fwd/avg_pg_fwd:.2f}x (theoretical: {1/(1-token_savings):.2f}x)")
    print(f"  Train speedup:   {avg_off_train/avg_pg_train:.2f}x")
    print(f"  Memory savings:  {(peak_off-peak_pg)/peak_off:.1%}")
    print(f"  Precision:       cos_sim={cos_sim_fwd:.6f}")
    print(f"{'='*60}")

# ===== Save Results =====
output_path = os.path.expanduser("~/pg_competitive/pg_benchmark_results.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# ===== Print Final Table =====
print(f"\n{'='*70}")
print("PREFIXGROUPER COMPETITIVE ANALYSIS - Qwen2.5-0.5B-Instruct")
print(f"{'='*70}")
print(f"{'n':>3} | {'Token Save':>10} | {'Fwd OFF(ms)':>12} | {'Fwd PG(ms)':>10} | {'Fwd Spd':>8} | "
      f"{'Train OFF':>10} | {'Train PG':>10} | {'Train Spd':>9} | {'Mem Save':>8} | {'CosSim':>8}")
print("-" * 95)
for n, r in results.items():
    print(f"{r['n']:>3} | {r['token_savings_pct']:>9.1f}% | {r['forward_no_pg_ms']:>11.1f} | "
          f"{r['forward_pg_ms']:>9.1f} | {r['forward_speedup']:>7.2f}x | "
          f"{r['train_no_pg_ms']:>9.1f} | {r['train_pg_ms']:>9.1f} | {r['train_speedup']:>8.2f}x | "
          f"{r['mem_savings_pct']:>7.1f}% | {r['forward_cos_sim']:>7.4f}")
print(f"{'='*70}")