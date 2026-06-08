#!/usr/bin/env python3
"""PR #4368 (PrefixGrouper) Competitive Analysis - Standalone Benchmark.

Directly benchmarks PrefixGrouper ON vs OFF using transformers + PrefixGrouper,
without needing the full verl pipeline (which requires vLLM for rollout).

This measures the core attention-level speedup that PrefixGrouper provides:
1. Normal forward: N sequences × (prefix + suffix) full attention
2. PG forward: 1 prefix self-attention + N suffix concat-attention

Model: Qwen2.5-0.5B-Instruct on RTX 4090

Metrics:
- Per-step timing (forward only, forward+backward)
- Peak GPU memory
- Precision alignment (PG vs normal log_probs cos_sim)
"""
import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from prefix_grouper import PrefixGrouper

MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen2.5-0.5B-Instruct")
DEVICE = "cuda:0"
SEED = 42

torch.manual_seed(SEED)

# Load model and tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model.eval()
config = AutoConfig.from_pretrained(MODEL_PATH)
vocab_size = config.vocab_size
hidden_size = config.hidden_size
num_layers = config.num_hidden_layers
num_heads = config.num_attention_heads
head_dim = hidden_size // num_heads

print(f"Model: Qwen2.5-0.5B, vocab={vocab_size}, hidden={hidden_size}, "
      f"layers={num_layers}, heads={num_heads}, head_dim={head_dim}")
print(f"GPU memory after load: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")


def generate_sequences(prefix_len, suffix_len, n, vocab_size, device):
    """Generate random token sequences for benchmarking."""
    prefix_tokens = torch.randint(10, vocab_size - 10, (1, prefix_len), device=device)
    suffix_tokens = torch.randint(10, vocab_size - 10, (n, suffix_len), device=device)
    # Full sequences: prefix + suffix
    full_sequences = torch.cat([
        prefix_tokens.expand(n, -1),
        suffix_tokens
    ], dim=1)
    return prefix_tokens, suffix_tokens, full_sequences


def run_normal_forward(model, sequences, device):
    """Normal forward: each sequence runs through full model independently."""
    with torch.no_grad():
        outputs = model(input_ids=sequences)
    logits = outputs.logits
    return logits


def run_pg_forward(model, prefix_tokens, suffix_tokens, prefix_grouper, device):
    """PrefixGrouper forward: prefix self-attn + suffix concat-attn.

    Steps:
    1. concat_input: merge prefix and suffix into grouped format
    2. Run full model forward on grouped input
    3. split_output: split back to individual suffix outputs
    """
    n = suffix_tokens.shape[0]
    prefix_len = prefix_tokens.shape[1]
    suffix_len = suffix_tokens.shape[1]

    # Create masks (no padding)
    prefix_mask = torch.ones(1, prefix_len, device=device, dtype=torch.long)
    suffix_mask = torch.ones(n, suffix_len, device=device, dtype=torch.long)

    # concat_input: creates grouped batch where each group has prefix + all suffixes
    grouped_input = prefix_grouper.concat_input(
        prefix_tokens, prefix_mask, suffix_tokens, suffix_mask
    )
    grouped_mask = prefix_grouper.padding_mask.long()  # bool -> int for transformers

    # Run model forward on grouped input
    with torch.no_grad():
        outputs = model(input_ids=grouped_input, attention_mask=grouped_mask)
    grouped_logits = outputs.logits

    # split_output: get suffix-only logits (returns prefix_logits, prefix_mask, suffix_logits, suffix_mask)
    pg_prefix_logits, _, pg_suffix_logits, _ = prefix_grouper.split_output(grouped_logits)

    return pg_suffix_logits


# Sweep configurations
configs = [
    {"prefix_len": 64, "suffix_len": 32, "n": 4, "label": "p64_s32_n4"},
    {"prefix_len": 128, "suffix_len": 32, "n": 4, "label": "p128_s32_n4"},
    {"prefix_len": 256, "suffix_len": 64, "n": 4, "label": "p256_s64_n4"},
    {"prefix_len": 256, "suffix_len": 64, "n": 8, "label": "p256_s64_n8"},
    {"prefix_len": 512, "suffix_len": 128, "n": 4, "label": "p512_s128_n4"},
]

results = []

# Warmup run to eliminate CUDA initialization overhead
print("\nWarmup run...")
warmup_prefix, warmup_suffix, warmup_full = generate_sequences(64, 32, 4, vocab_size, DEVICE)
with torch.no_grad():
    _ = model(input_ids=warmup_full)
torch.cuda.synchronize()
print("Warmup done.")

NUM_REPEATS = 5  # Average over multiple runs

for cfg in configs:
    prefix_len = cfg["prefix_len"]
    suffix_len = cfg["suffix_len"]
    n = cfg["n"]
    label = cfg["label"]
    total_len = prefix_len + suffix_len

    print(f"\n{'='*60}")
    print(f"Config: {label} (prefix={prefix_len}, suffix={suffix_len}, n={n})")
    print(f"{'='*60}")

    # Generate sequences
    prefix_tokens, suffix_tokens, full_sequences = generate_sequences(
        prefix_len, suffix_len, n, vocab_size, DEVICE
    )

    # Build PrefixGrouper
    # group_info: list of [prefix_len, suffix_len_1, suffix_len_2, ...]
    group_info = [[prefix_len] + [suffix_len] * n]
    pg = PrefixGrouper(group_info=group_info, device=DEVICE)

    # ===== Normal forward (baseline) =====
    t_normal_list = []
    peak_normal = 0
    for _ in range(NUM_REPEATS):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_start = time.time()
        normal_logits = run_normal_forward(model, full_sequences, DEVICE)
        torch.cuda.synchronize()
        t_normal_list.append(time.time() - t_start)
        peak_normal = max(peak_normal, torch.cuda.max_memory_allocated() / 1024**3)
    t_normal = sum(t_normal_list) / NUM_REPEATS

    # ===== PG forward =====
    t_pg_list = []
    peak_pg = 0
    for _ in range(NUM_REPEATS):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_start = time.time()
        pg_suffix_logits = run_pg_forward(
            model, prefix_tokens, suffix_tokens, pg, DEVICE
        )
        torch.cuda.synchronize()
        t_pg_list.append(time.time() - t_start)
        peak_pg = max(peak_pg, torch.cuda.max_memory_allocated() / 1024**3)
    t_pg = sum(t_pg_list) / NUM_REPEATS

    # ===== Precision check =====
    # Normal: logits for suffix tokens at positions prefix_len..total_len-1
    # We compare next-token prediction logits for suffix positions
    # Normal suffix logits: normal_logits[:, prefix_len-1:total_len-1, :] predicts tokens at prefix_len:total_len
    # PG suffix logits: pg_suffix_logits predicts suffix tokens

    # For next-token prediction alignment:
    # Normal: logits at position p predict token at position p+1
    # Suffix tokens start at absolute position prefix_len
    # So normal suffix prediction logits = normal_logits[:, prefix_len-1:-1, :]

    normal_suffix_logits = normal_logits[:, prefix_len-1:-1, :]  # (n, suffix_len, vocab)
    # PG suffix_logits might have different shape - let's check

    print(f"  Normal suffix logits shape: {normal_suffix_logits.shape}")
    print(f"  PG suffix logits shape: {pg_suffix_logits.shape}")

    # Compute log_probs
    normal_log_probs = F.log_softmax(normal_suffix_logits.float(), dim=-1)
    pg_log_probs = F.log_softmax(pg_suffix_logits.float(), dim=-1)

    # Compute cos_sim between log_probs
    cos_sims = []
    for i in range(n):
        cs = F.cosine_similarity(
            normal_log_probs[i].flatten(),
            pg_log_probs[i].flatten(),
            dim=0
        ).item()
        cos_sims.append(cs)
    mean_cos = sum(cos_sims) / len(cos_sims)
    max_diff = (normal_log_probs - pg_log_probs).abs().max().item()

    # Compute speedup
    speedup = t_normal / t_pg if t_pg > 0 else 0

    result = {
        "label": label,
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "n": n,
        "t_normal_ms": t_normal * 1000,
        "t_pg_ms": t_pg * 1000,
        "speedup": speedup,
        "peak_normal_gib": peak_normal,
        "peak_pg_gib": peak_pg,
        "memory_save": (peak_normal - peak_pg) / peak_normal * 100 if peak_normal > 0 else 0,
        "cos_sim": mean_cos,
        "max_diff": max_diff,
        "per_seq_cos": cos_sims,
    }
    results.append(result)

    print(f"  Normal forward: {t_normal*1000:.1f}ms, peak GPU: {peak_normal:.2f} GiB")
    print(f"  PG forward: {t_pg*1000:.1f}ms, peak GPU: {peak_pg:.2f} GiB")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Memory save: {result['memory_save']:.1f}%")
    print(f"  Precision: cos_sim={mean_cos:.6f}, max_diff={max_diff:.4f}")
    print(f"  Status: {'PASS' if mean_cos >= 0.99 else 'MISMATCH' if mean_cos >= 0.9 else 'FAIL'}")

    # Cleanup
    del normal_logits, pg_suffix_logits, normal_suffix_logits
    del normal_log_probs, pg_log_probs
    torch.cuda.empty_cache()


# ===== Summary =====
print(f"\n{'='*60}")
print(f"PR #4368 (PrefixGrouper) Competitive Analysis Summary")
print(f"{'='*60}")
print(f"Model: Qwen2.5-0.5B, GPU: RTX 4090")
print(f"")
print(f"| Config | Normal(ms) | PG(ms) | Speedup | Normal GiB | PG GiB | Mem Save% | cos_sim |")
print(f"|--------|-----------|--------|---------|-----------|--------|----------|---------|")
for r in results:
    print(f"| {r['label']} | {r['t_normal_ms']:.1f} | {r['t_pg_ms']:.1f} | {r['speedup']:.2f}x | "
          f"{r['peak_normal_gib']:.2f} | {r['peak_pg_gib']:.2f} | {r['memory_save']:.1f}% | "
          f"{r['cos_sim']:.6f} |")

print(f"\nKey findings:")
avg_speedup = sum(r['speedup'] for r in results) / len(results)
avg_cos = sum(r['cos_sim'] for r in results) / len(results)
print(f"  Average speedup: {avg_speedup:.2f}x")
print(f"  Average cos_sim: {avg_cos:.6f}")

if avg_speedup > 1.0:
    print(f"  PG is faster than normal forward (attention-level optimization)")
else:
    print(f"  PG is NOT faster than normal forward (Python overhead dominates)")

# Save results
import json
with open(os.path.expanduser("~/rollout-prefix/pg-competitive/pg_benchmark_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to ~/rollout-prefix/pg-competitive/pg_benchmark_results.json")