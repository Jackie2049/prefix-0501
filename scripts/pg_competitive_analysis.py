"""
PR #4368 PrefixGrouper Competitive Analysis - Precision Validation
==================================================================

Validates that PG block-causal attention produces identical logits to
normal mode, using SDPA with a 4D custom attention mask.

Then runs comprehensive timing benchmarks with n=4 and n=8.
"""

import argparse
import json
import os
import time
import gc
import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from prefix_grouper import PrefixGrouper


def load_model_and_tokenizer(model_path, device="cuda", attn_impl="sdpa"):
    print(f"Loading model from {model_path} (attn_impl={attn_impl})...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True, attn_implementation=attn_impl,
    )
    model.eval()
    print(f"Model loaded: {model.config.model_type}, {model.config.num_hidden_layers} layers")
    return model, tokenizer


def make_position_ids(attention_mask):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


def create_unpadded_prompts(tokenizer, n_prompts, prompt_length, device):
    """Create prompts that tokenize to exactly prompt_length tokens (no padding)."""
    prompt_texts = []
    for i in range(n_prompts):
        base = f"What is {i}+{i}? "
        reasoning = "Let me think step by step. First, I need to add these numbers. "
        reasoning += "Addition is a fundamental arithmetic operation. "
        reasoning += "I will compute the sum carefully and verify my answer. "
        text = base + reasoning * ((prompt_length // 25) + 2)
        prompt_texts.append(text)

    prompt_ids_list = []
    for text in prompt_texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) > prompt_length:
            ids = ids[:prompt_length]
        elif len(ids) < prompt_length:
            # Pad with repeated tokens to reach exact length
            while len(ids) < prompt_length:
                ids.append(ids[-1])  # repeat last token
        prompt_ids_list.append(ids)

    prompts = torch.tensor(prompt_ids_list, dtype=torch.long, device=device)
    prompt_mask = torch.ones_like(prompts)
    return prompts, prompt_mask


def build_block_causal_mask(pg, prefix_mask, n_suffixes):
    """Build a 4D block-causal attention mask using vectorized tensor ops.

    Block-causal pattern:
    - Prefix tokens: causal within prefix
    - Suffix_k tokens: see ALL real prefix + causal within own suffix_k
    - Suffix_k CANNOT see other suffixes

    Returns: (num_groups, 1, seq_len, seq_len) float mask
      0.0 = attend, -inf = mask out
    """
    num_groups = pg.padding_mask.size(0)
    seq_len = pg.padding_mask.size(1)
    device = pg.padding_mask.device

    # Step 1: Causal mask (upper triangular: j <= i)
    causal = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0).bool()
    causal = causal.unsqueeze(0).unsqueeze(0).expand(num_groups, -1, -1, -1)

    # Step 2: Suffix assignment tensor (0=prefix, 1..k=suffix index)
    suffix_id = torch.zeros(num_groups, seq_len, dtype=torch.long, device=device)
    for g in range(num_groups):
        group = pg.group_info[g]
        prefix_len = group.prefix_len
        cur_pos = prefix_len
        for s_idx, s_len in enumerate(group.suffix_lens):
            if s_len > 0:
                suffix_id[g, cur_pos:cur_pos + s_len] = s_idx + 1
                cur_pos += s_len

    # Step 3: Block-causal via tensor ops
    # i can attend to j if: causal AND (same suffix OR j is prefix)
    i_sid = suffix_id.unsqueeze(1).unsqueeze(2)   # (G,1,S,1)
    j_sid = suffix_id.unsqueeze(1).unsqueeze(3)   # (G,1,1,S)
    same_block = (i_sid == j_sid) | (j_sid == 0)  # same suffix block or j in prefix
    block_ok = same_block & causal

    # Step 4: Padding mask (only attend to valid positions)
    valid_q = pg.padding_mask.unsqueeze(1).unsqueeze(2).bool()
    valid_kv = pg.padding_mask.unsqueeze(1).unsqueeze(3).bool()

    # Step 5: Prefix real-token mask (for left-padded prefixes)
    # Build a per-position validity mask: prefix positions are real if prefix_mask=1
    pos_valid = pg.padding_mask.bool()  # (G, S) start with overall padding mask
    for g in range(num_groups):
        prefix_len = pg.group_info[g].prefix_len
        pm = prefix_mask[g]  # (prefix_len,) 1 for real, 0 for left-pad
        pos_valid[g, :prefix_len] = pm.bool()  # override prefix positions with real-token mask

    kv_valid = pos_valid.unsqueeze(1).unsqueeze(2).expand(-1, -1, seq_len, -1)
    q_valid = pos_valid.unsqueeze(1).unsqueeze(3).expand(-1, -1, -1, seq_len)

    mask_bool = block_ok & valid_q & valid_kv & kv_valid & q_valid

    # Convert to float mask matching model dtype (bf16): 0.0 = attend, -inf = mask
    mask = torch.where(mask_bool, torch.tensor(0.0, dtype=torch.bfloat16, device=device),
                       torch.tensor(float('-inf'), dtype=torch.bfloat16, device=device))
    return mask


def validate_precision(model, tokenizer, device, n=4, prompt_length=64, response_length=64):
    """Validate PG precision: compare logits from normal vs PG block-causal mode."""
    print(f"\n{'='*60}")
    print(f"PRECISION VALIDATION: n={n}, prompt_len={prompt_length}, resp_len={response_length}")
    print(f"{'='*60}")

    n_prompts = 1  # Single prompt for clean validation

    prompts, prompt_mask = create_unpadded_prompts(tokenizer, n_prompts, prompt_length, device)

    # Generate n different responses
    print("Generating responses...")
    with torch.no_grad():
        generated = model.generate(
            prompts, attention_mask=prompt_mask,
            max_new_tokens=response_length, do_sample=True, temperature=0.6,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )

    responses = generated[:, prompt_length:]  # (1, response_length)

    # Create n different responses by generating multiple times
    all_responses = []
    for _ in range(n):
        with torch.no_grad():
            gen = model.generate(
                prompts, attention_mask=prompt_mask,
                max_new_tokens=response_length, do_sample=True, temperature=0.6,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        all_responses.append(gen[:, prompt_length:])
    responses = torch.cat(all_responses, dim=0)  # (n, response_length)

    # Normal mode: forward pass on each (prompt + response_i) independently
    print("Computing normal logits...")
    normal_logits_list = []
    for i in range(n):
        full_seq = torch.cat([prompts[0:1], responses[i:i+1]], dim=1)  # (1, prompt_len + resp_len)
        full_mask = torch.ones_like(full_seq)
        full_pos = make_position_ids(full_mask)
        with torch.no_grad():
            out = model(input_ids=full_seq, attention_mask=full_mask, position_ids=full_pos)
            normal_logits_list.append(out.logits.cpu())

    normal_logits = torch.cat(normal_logits_list, dim=0)  # (n, seq_len, vocab)

    # PG mode: grouped forward pass with block-causal mask
    print("Computing PG logits...")
    responses_repeated = responses  # (n, response_length) - no repetition needed for single prompt
    response_mask = torch.ones_like(responses_repeated)

    # Mask after EOS
    for i in range(responses_repeated.size(0)):
        eos_pos = (responses_repeated[i] == tokenizer.eos_token_id).nonzero()
        if len(eos_pos) > 0:
            response_mask[i, eos_pos[0].item() + 1:] = 0

    group_sizes = [n]
    pg = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prompt_mask,
        suffix_mask=response_mask,
        group_sizes=group_sizes,
        padding_mode="right",
        device=device,
    )

    concat_input_ids = pg.concat_input(prompts, prompt_mask, responses_repeated, response_mask)
    pg_attention_mask = pg.padding_mask

    # Build position_ids
    pg_position_ids = torch.zeros(1, pg_attention_mask.size(1), dtype=torch.long, device=device)
    group = pg.group_info[0]
    prefix_len = group.prefix_len
    pg_position_ids[0, :prefix_len] = torch.arange(prefix_len, device=device)
    cur_pos = prefix_len
    for s_len in group.suffix_lens:
        if s_len > 0:
            pg_position_ids[0, cur_pos:cur_pos + s_len] = torch.arange(
                prefix_len, prefix_len + s_len, device=device
            )
            cur_pos += s_len

    # Build 4D block-causal mask
    block_causal_mask = build_block_causal_mask(pg, prompt_mask, n)

    print(f"  concat_input shape: {concat_input_ids.shape}")
    print(f"  block_causal_mask shape: {block_causal_mask.shape}")
    print(f"  prefix_len={prefix_len}, suffix_lens={group.suffix_lens}")

    with torch.no_grad():
        out = model(
            input_ids=concat_input_ids,
            attention_mask=block_causal_mask,
            position_ids=pg_position_ids,
        )
        concat_logits = out.logits

    # Split output
    split_result = pg.split_output(concat_logits)
    prefix_logits, _, suffix_logits, suffix_mask_out = split_result

    # Reconstruct per-sample logits
    # prefix_logits: (1, prefix_len, vocab), suffix_logits: (n, suffix_len, vocab)
    # For each sample: prefix_logits[0] + suffix_logits[i]
    pg_logits_per_sample = []
    for i in range(n):
        sample_logits = torch.cat([prefix_logits[0], suffix_logits[i]], dim=0)  # (prefix_len + suffix_len, vocab)
        pg_logits_per_sample.append(sample_logits.cpu())

    pg_logits = torch.stack(pg_logits_per_sample, dim=0)  # (n, seq_len, vocab)

    # Compare response logits (suffix portion)
    print(f"  normal_logits shape: {normal_logits.shape}")
    print(f"  pg_logits shape: {pg_logits.shape}")

    # Extract response portion: logits at position prompt_len-1 predict first response token
    resp_start = prompt_length - 1
    resp_len = response_length

    normal_resp_logits = normal_logits[:, resp_start:resp_start + resp_len, :]
    pg_resp_logits = pg_logits[:, resp_start:resp_start + resp_len, :]

    min_len = min(normal_resp_logits.size(1), pg_resp_logits.size(1))
    normal_resp_logits = normal_resp_logits[:, :min_len, :]
    pg_resp_logits = pg_resp_logits[:, :min_len, :]

    # Compare log probs
    normal_logprobs = F.log_softmax(normal_resp_logits.float(), dim=-1)
    pg_logprobs = F.log_softmax(pg_resp_logits.float(), dim=-1)

    # Flatten and compare
    normal_flat = normal_logprobs.reshape(-1)
    pg_flat = pg_logprobs.reshape(-1)

    cos_sim = F.cosine_similarity(normal_flat.unsqueeze(0), pg_flat.unsqueeze(0)).item()
    max_diff = (normal_flat - pg_flat).abs().max().item()
    mean_diff = (normal_flat - pg_flat).abs().mean().item()

    print(f"\n  PRECISION RESULTS:")
    print(f"  cos_sim = {cos_sim:.6f}")
    print(f"  max_diff = {max_diff:.4f}")
    print(f"  mean_diff = {mean_diff:.4f}")

    return {"cos_sim": cos_sim, "max_diff": max_diff, "mean_diff": mean_diff}


def benchmark_timing(model, tokenizer, device, n_samples, n, prompt_length, response_length, n_runs=5):
    """Timing benchmark: normal vs PG grouped forward pass."""
    print(f"\n{'='*60}")
    print(f"TIMING BENCHMARK: n_samples={n_samples}, n={n}, prompt_len={prompt_length}, resp_len={response_length}")
    print(f"{'='*60}")

    prompts, prompt_mask = create_unpadded_prompts(tokenizer, n_samples, prompt_length, device)

    # Generate responses
    print(f"Generating {n_samples} responses...")
    with torch.no_grad():
        generated = model.generate(
            prompts, attention_mask=prompt_mask,
            max_new_tokens=response_length, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    responses = generated[:, prompt_length:]

    # Create repeated batch (GRPO-style)
    prompts_repeated = prompts.repeat_interleave(n, dim=0)
    prompt_mask_repeated = prompt_mask.repeat_interleave(n, dim=0)
    responses_repeated = responses.repeat_interleave(n, dim=0)

    full_sequences = torch.cat([prompts_repeated, responses_repeated], dim=1)
    full_mask = torch.ones_like(full_sequences)
    full_position_ids = make_position_ids(full_mask)

    n_total = full_sequences.size(0)
    total_tokens_normal = n_total * full_sequences.size(1)

    print(f"Normal mode: {n_total} sequences × {full_sequences.size(1)} tokens = {total_tokens_normal} total tokens")

    # Build PG grouped input
    response_only_mask = torch.ones_like(responses_repeated)
    for i in range(responses_repeated.size(0)):
        eos_pos = (responses_repeated[i] == tokenizer.eos_token_id).nonzero()
        if len(eos_pos) > 0:
            response_only_mask[i, eos_pos[0].item() + 1:] = 0

    group_sizes = [n] * n_samples
    pg = PrefixGrouper.from_ungrouped_masks(
        prefix_mask=prompt_mask,
        suffix_mask=response_only_mask,
        group_sizes=group_sizes,
        padding_mode="right",
        device=device,
    )

    concat_input_ids = pg.concat_input(prompts, prompt_mask, responses_repeated, response_only_mask)
    pg_attention_mask = pg.padding_mask

    # Build position_ids for PG (unpadded prompts, simple case)
    pg_position_ids = torch.zeros(n_samples, pg_attention_mask.size(1), dtype=torch.long, device=device)
    for g_idx, group in enumerate(pg.group_info):
        prefix_len = group.prefix_len
        pg_position_ids[g_idx, :prefix_len] = torch.arange(prefix_len, device=device)
        cur_pos = prefix_len
        for s_len in group.suffix_lens:
            if s_len > 0:
                pg_position_ids[g_idx, cur_pos:cur_pos + s_len] = torch.arange(
                    prefix_len, prefix_len + s_len, device=device
                )
                cur_pos += s_len

    total_tokens_pg = concat_input_ids.numel()
    print(f"PG mode: {n_samples} groups × {concat_input_ids.size(1)} tokens = {total_tokens_pg} total tokens")
    print(f"Token savings: {total_tokens_normal - total_tokens_pg} ({(total_tokens_normal - total_tokens_pg)/total_tokens_normal:.1%})")

    # Warmup
    with torch.no_grad():
        _ = model(input_ids=full_sequences[:2], attention_mask=full_mask[:2])
    torch.cuda.synchronize()

    normal_times = []
    pg_times = []

    for run in range(n_runs):
        gc.collect()
        torch.cuda.empty_cache()

        # Normal mode
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model(input_ids=full_sequences, attention_mask=full_mask, position_ids=full_position_ids)
        torch.cuda.synchronize()
        normal_times.append(time.time() - t0)

        gc.collect()
        torch.cuda.empty_cache()

        # PG mode
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model(input_ids=concat_input_ids, attention_mask=pg_attention_mask, position_ids=pg_position_ids)
        torch.cuda.synchronize()
        pg_times.append(time.time() - t0)

    normal_avg = np.mean(normal_times)
    pg_avg = np.mean(pg_times)

    # Compute theoretical FLOPs savings
    # Per token FLOPs for a Transformer layer:
    # QKV: 3 * d^2, Attn: ~2*S*d (per query token), O_proj: d^2, MLP: ~2*d*4d=8d^2
    # Total per token: ~12d^2 + 2*S*d
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    seq_len = prompt_length + response_length

    # Normal mode FLOPs
    flops_per_token = 12 * d_model * d_model + 2 * seq_len * d_model  # rough estimate
    total_flops_normal = n_total * seq_len * flops_per_token * n_layers

    # PG mode: fewer tokens but longer sequences per group
    pg_seq_len = concat_input_ids.size(1)
    # Attention FLOPs are quadratic, but block-causal reduces them
    # For block-causal: prefix^2/2 + n*(suffix*prefix + suffix^2/2)
    # vs normal causal: n * (P+S)^2/2
    attn_flops_normal = n_total * seq_len * seq_len / 2 * d_model * n_layers
    attn_flops_pg_block_causal = n_samples * (
        prompt_length * prompt_length / 2 +  # prefix causal
        n * (response_length * prompt_length + response_length * response_length / 2)  # suffix blocks
    ) * d_model * n_layers

    # Non-attention FLOPs (QKV + O_proj + MLP) proportional to token count
    non_attn_flops_per_token = 12 * d_model * d_model
    non_attn_normal = n_total * seq_len * non_attn_flops_per_token * n_layers
    non_attn_pg = total_tokens_pg * non_attn_flops_per_token * n_layers

    prefix_ratio = prompt_length / (prompt_length + response_length)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Normal avg: {normal_avg*1000:.1f}ms")
    print(f"PG avg: {pg_avg*1000:.1f}ms")
    print(f"Measured speedup: {normal_avg/pg_avg:.2f}x")
    print(f"Prefix ratio: {prefix_ratio:.1%}")
    print(f"Token savings: {(total_tokens_normal - total_tokens_pg)/total_tokens_normal:.1%}")
    print(f"Attn FLOPs savings (block-causal vs causal): {(attn_flops_normal - attn_flops_pg_block_causal)/attn_flops_normal:.1%}")
    print(f"Non-attn FLOPs savings: {(non_attn_normal - non_attn_pg)/non_attn_normal:.1%}")
    print(f"Theoretical total FLOPs savings: {(total_flops_normal - (attn_flops_pg_block_causal + non_attn_pg))/total_flops_normal:.1%}")

    mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"Peak GPU memory: {mem:.2f}GB")

    return {
        "n_samples": n_samples, "n": n,
        "prompt_length": prompt_length, "response_length": response_length,
        "prefix_ratio": prefix_ratio,
        "normal_avg_ms": normal_avg * 1000,
        "pg_avg_ms": pg_avg * 1000,
        "measured_speedup": normal_avg / pg_avg,
        "total_tokens_normal": total_tokens_normal,
        "total_tokens_pg": total_tokens_pg,
        "token_savings_pct": (total_tokens_normal - total_tokens_pg) / total_tokens_normal,
        "attn_flops_savings_pct": (attn_flops_normal - attn_flops_pg_block_causal) / attn_flops_normal,
        "non_attn_flops_savings_pct": (non_attn_normal - non_attn_pg) / non_attn_normal,
        "gpu_memory_gb": mem,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/rollout-prefix/models/Qwen2.5-0.5B-Instruct"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    # Phase 1: Precision validation with SDPA + block-causal mask
    print("\n" + "="*60)
    print("PHASE 1: PRECISION VALIDATION")
    print("="*60)

    model_sdpa, tokenizer = load_model_and_tokenizer(args.model_path, device, attn_impl="sdpa")

    precision_results = []
    for n in [4, 8]:
        for p_len in [64, 128]:
            try:
                result = validate_precision(model_sdpa, tokenizer, device, n=n, prompt_length=p_len, response_length=64)
                precision_results.append({"n": n, "prompt_length": p_len, **result})
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                precision_results.append({"n": n, "prompt_length": p_len, "error": str(e)})

    del model_sdpa
    gc.collect()
    torch.cuda.empty_cache()

    # Phase 2: Timing benchmark with flash attention (faster, but no precision validation)
    print("\n" + "="*60)
    print("PHASE 2: TIMING BENCHMARK")
    print("="*60)

    model, tokenizer = load_model_and_tokenizer(args.model_path, device, attn_impl="flash_attention_2")

    timing_results = []
    configs = [
        # n=4 configurations
        {"n_samples": 4, "n": 4, "prompt_length": 64, "response_length": 128},
        {"n_samples": 4, "n": 4, "prompt_length": 128, "response_length": 128},
        {"n_samples": 4, "n": 4, "prompt_length": 256, "response_length": 128},
        # n=8 configurations
        {"n_samples": 4, "n": 8, "prompt_length": 64, "response_length": 128},
        {"n_samples": 4, "n": 8, "prompt_length": 128, "response_length": 128},
        {"n_samples": 4, "n": 8, "prompt_length": 256, "response_length": 128},
    ]

    for cfg in configs:
        result = benchmark_timing(
            model, tokenizer, device,
            cfg["n_samples"], cfg["n"], cfg["prompt_length"], cfg["response_length"],
            n_runs=5
        )
        timing_results.append(result)

    # Save all results
    all_results = {"precision": precision_results, "timing": timing_results}
    output_file = os.path.expanduser("~/rollout-prefix/pg_competitive_analysis_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"COMPETITIVE ANALYSIS SUMMARY - PR #4368 PrefixGrouper (Qwen2.5-0.5B)")
    print(f"{'='*80}")

    print(f"\n--- PRECISION VALIDATION (SDPA + block-causal mask) ---")
    print(f"{'Config':<20} {'cos_sim':<12} {'max_diff':<12} {'Status':<10}")
    for r in precision_results:
        config = f"n={r['n']}, p={r['prompt_length']}"
        if "error" in r:
            print(f"{config:<20} {'N/A':<12} {'N/A':<12} {'FAILED':<10}")
        else:
            status = "PASS" if r['cos_sim'] > 0.999 else "FAIL"
            print(f"{config:<20} {r['cos_sim']:.6f}   {r['max_diff']:.4f}     {status:<10}")

    print(f"\n--- TIMING BENCHMARK (flash_attention_2) ---")
    print(f"{'Config':<30} {'PG OFF(ms)':<12} {'PG ON(ms)':<12} {'Speedup':<10} {'Token Save':<12} {'FLOPs Save':<12}")
    print(f"{'-'*80}")
    for r in timing_results:
        config = f"n={r['n']}, p={r['prompt_length']}, r={r['response_length']} ({r['prefix_ratio']:.0%})"
        pg_off = f"{r['normal_avg_ms']:.1f}"
        pg_on = f"{r['pg_avg_ms']:.1f}"
        speedup = f"{r['measured_speedup']:.2f}x"
        token_save = f"{r['token_savings_pct']:.1%}"
        # Total FLOPs savings ≈ token savings (since attention is small fraction for 0.5B)
        flops_save = f"{r['non_attn_flops_savings_pct']:.1%}"
        print(f"{config:<30} {pg_off:<12} {pg_on:<12} {speedup:<10} {token_save:<12} {flops_save:<12}")

    # Projected speedup for larger models
    print(f"\n--- PROJECTED SPEEDUP FOR LARGER MODELS ---")
    print(f"Model         Attn%   n=4,p=33%  n=4,p=50%  n=4,p=67%  n=8,p=33%  n=8,p=50%  n=8,p=67%")
    print(f"{'-'*80}")
    for model_name, attn_pct in [("0.5B", 3), ("7B", 10), ("70B", 15)]:
        row = f"{model_name:<14} {attn_pct}%   "
        for n_val in [4, 8]:
            for p_ratio in [0.33, 0.50, 0.67]:
                # Theoretical speedup = 1 / (1 - (n-1)/n * p_ratio * (1 - attn_pct/100))
                # This assumes only non-attention computation is saved on prefix
                prefix_compute_saved = (n_val - 1) / n_val * p_ratio
                attn_penalty = 1 + (n_val - 1) * p_ratio * attn_pct / 100  # quadratic attention cost
                # More accurate: total FLOPs ratio
                # Normal: n * (P+S) * (attn_flops + non_attn_flops)
                # PG: 1*P*(attn+non_attn) + n*S*(attn+non_attn) + n*S*P*attn (suffix→prefix cross)
                # Simplified: speedup ≈ 1 / (1 - prefix_compute_saved * (1 - attn_frac))
                speedup = 1 / (1 - prefix_compute_saved * (1 - attn_pct / 100))
                row += f"{speedup:.2f}x    "
        print(row)


if __name__ == "__main__":
    main()