#!/usr/bin/env python3
"""PG suffix precision debug - compare PG ON vs OFF suffix log_probs."""
import os, sys, torch, torch.nn.functional as F
import json

VERL_CLEAN = os.path.expanduser("~/rollout-prefix/verl-clean")
sys.path.insert(0, VERL_CLEAN)

from prefix_grouper import PrefixGrouper
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.models.transformers.monkey_patch import apply_prefix_grouper_patch
from verl.utils.torch_functional import logprobs_from_logits
from verl.trainer.ppo.prefix_grouper_utils import build_pg_from_micro_batch

model_path = os.path.expanduser("~/rollout-prefix/models/Qwen2.5-0.5B-Instruct")
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

n = 4
# Use a short, identical prefix so prefix_ratio is high
prompts = ["What is 5+5? Let me think step by step. " * 5] * n  # identical

encoded = tokenizer(prompts, padding=True, truncation=True, max_length=64, return_tensors="pt")
input_ids_off = encoded["input_ids"].to(device)
attention_mask_off = encoded["attention_mask"].to(device)
prompt_len = input_ids_off.shape[1]

# Generate responses using model.generate() (same for both tests)
print(f"Input shape: {input_ids_off.shape}, prompt_len={prompt_len}")

# ========== Normal forward: compute log_probs ==========
print("Loading model OFF...")
model_off = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map=device,
    trust_remote_code=True, attn_implementation="flash_attention_2",
)
model_off.eval()

# Generate responses first (needed for both ON and OFF)
print("Generating responses...")
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    responses_full = model_off.generate(
        input_ids=input_ids_off, attention_mask=attention_mask_off,
        max_new_tokens=32, do_sample=False,  # greedy for consistency
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
response_tokens = responses_full[:, prompt_len:]
print(f"Response shape: {response_tokens.shape}")

# Build response mask
response_mask = (response_tokens != tokenizer.pad_token_id).long()
for i in range(n):
    eos_pos = (response_tokens[i] == tokenizer.eos_token_id).nonzero()
    if len(eos_pos) > 0:
        response_mask[i, eos_pos[0].item() + 1:] = 0
response_mask = response_mask.to(device)

# Normal log_probs
print("Computing normal log_probs...")
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    seq = torch.cat([input_ids_off, response_tokens], dim=1)
    seq_mask = torch.cat([attention_mask_off, response_mask.to(attention_mask_off.dtype)], dim=1)
    out_off = model_off(input_ids=seq, attention_mask=seq_mask)
    logits_off = out_off.logits

shift_logits = logits_off[:, prompt_len-1:-1, :]
shift_labels = response_tokens
lp_off_all = F.log_softmax(shift_logits.float(), dim=-1)
lp_off = lp_off_all.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
print(f"Normal log_probs shape: {lp_off.shape}")

del model_off
torch.cuda.empty_cache()

# ========== PG forward: compute log_probs with PrefixGrouper ==========
print("Applying PG patch...")
apply_prefix_grouper_patch()

print("Loading model ON (with PG patch)...")
model_on = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map=device,
    trust_remote_code=True, attn_implementation="flash_attention_2",
)
model_on.eval()

# Build PG input
uid_tensor = torch.tensor([0] * n)  # all same group
micro_batch = {
    "prompts": input_ids_off,
    "responses": response_tokens,
    "response_mask": response_mask,
    "uid": uid_tensor,
    "pad_token_id": tokenizer.pad_token_id,
}

(pg, concat_ids, pg_mask, position_ids, pg_responses, pg_response_mask) = build_pg_from_micro_batch(
    micro_batch, pad_token_id=tokenizer.pad_token_id, padding_mode="right"
)

print(f"PG concat shape: {concat_ids.shape}")
print(f"PG responses shape: {pg_responses.shape}")

# Forward with prefix_grouper
print("Computing PG log_probs...")
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out_on = model_on(input_ids=concat_ids, attention_mask=pg_mask, position_ids=position_ids,
                       prefix_grouper=pg, use_cache=False)
    logits_on_full = out_on.logits

print(f"PG logits shape: {logits_on_full.shape}")

# Split output using include_prefix_last=1
prefix_out, prefix_mask_out, suffix_out_raw, suffix_mask_raw = pg.split_output(
    logits_on_full, include_prefix_last=1
)

print(f"Suffix logits raw shape: {suffix_out_raw.shape}")
print(f"Suffix mask raw shape: {suffix_mask_raw.shape}")

# PG's log_probs computation (from pg_e2e_grpo_benchmark_v2.py)
suffix_out = suffix_out_raw[:, :-1].float()
suffix_mask = suffix_mask_raw[:, 1:]

# Convert responses to right-padded format
completion_ids_right = pg.convert_padding(response_tokens, response_mask, padding_mode="right")

# Temperature scaling (T=1.0 so no effect)
suffix_out /= 1.0

# Compute log probs using verl's function
lp_on = logprobs_from_logits(suffix_out, completion_ids_right)

# Zero out padding
padding_mask = suffix_mask == 0
lp_on = lp_on.masked_fill(padding_mask, 0.0)

# Pad to target length
target_len = response_tokens.size(1)
if lp_on.size(1) != target_len:
    full = lp_on.new_zeros(lp_on.size(0), target_len)
    full[:, :lp_on.size(1)] = lp_on
    lp_on = full

print(f"PG log_probs shape: {lp_on.shape}")

# ========== Compare log_probs ==========
print(f"\n{'='*60}")
print(f"LOG_PROB PRECISION COMPARISON")
print(f"{'='*60}")

# Only compare on valid (non-padding) positions
valid = response_mask.bool()

off_v = lp_off[valid].float()
on_v = lp_on[valid].float()

cos_sim = F.cosine_similarity(off_v.unsqueeze(0), on_v.unsqueeze(0)).item()
max_diff = (off_v - on_v).abs().max().item()
mean_diff = (off_v - on_v).abs().mean().item()

print(f"Overall (valid positions):")
print(f"  cos_sim: {cos_sim:.6f}")
print(f"  max_diff: {max_diff:.6f}")
print(f"  mean_diff: {mean_diff:.6f}")

# Per-sequence comparison
for i in range(n):
    seq_valid = response_mask[i].bool()
    off_seq = lp_off[i][seq_valid].float()
    on_seq = lp_on[i][seq_valid].float()
    cs = F.cosine_similarity(off_seq.unsqueeze(0), on_seq.unsqueeze(0)).item()
    md = (off_seq - on_seq).abs().max().item()
    print(f"  Seq {i}: cos_sim={cs:.6f}, max_diff={md:.6f}, n_valid={seq_valid.sum().item()}")

# Also compare per-position for seq 0
print(f"\nPer-position comparison (seq 0):")
for pos in range(min(10, response_tokens.size(1))):
    if response_mask[0, pos].item() == 0:
        continue
    off_val = lp_off[0, pos].item()
    on_val = lp_on[0, pos].item()
    diff = abs(off_val - on_val)
    print(f"  pos {pos}: OFF={off_val:.4f}, ON={on_val:.4f}, diff={diff:.4f}")

# Save results
results = {
    "cos_sim": cos_sim,
    "max_diff": max_diff,
    "mean_diff": mean_diff,
    "prefix_len": prompt_len,
    "response_len": response_tokens.size(1),
    "n_sequences": n,
}

result_file = "/home/zxw/rollout-prefix/pg_suffix_precision_debug.json"
with open(result_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {result_file}")