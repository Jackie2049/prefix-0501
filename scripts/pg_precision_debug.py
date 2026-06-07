#!/usr/bin/env python3
"""Quick PG precision debug - test PG ON vs OFF with identical prompts."""
import os, sys, torch, torch.nn.functional as F

VERL_CLEAN = os.path.expanduser("~/rollout-prefix/verl-clean")
sys.path.insert(0, VERL_CLEAN)

from prefix_grouper import PrefixGrouper
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.models.transformers.monkey_patch import apply_prefix_grouper_patch

model_path = os.path.expanduser("~/rollout-prefix/models/Qwen2.5-0.5B-Instruct")
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

n = 4
prompts = ["What is 5+5? " + "Let me think. " * 3] * n  # identical prompts

encoded = tokenizer(prompts, padding=True, truncation=True, max_length=64, return_tensors="pt")
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

prompt_len = input_ids.shape[1]
print(f"Input shape: {input_ids.shape}, prompt_len={prompt_len}")

# ========== Normal forward (NO patch) ==========
print("Loading model OFF...")
model_off = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map=device,
    trust_remote_code=True, attn_implementation="flash_attention_2",
)
model_off.eval()

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out_off = model_off(input_ids=input_ids, attention_mask=attention_mask)
    logits_off = out_off.logits

print(f"OFF logits shape: {logits_off.shape}")

# ========== PG forward (WITH patch) ==========
print("Applying PG patch...")
apply_prefix_grouper_patch()

print("Loading model ON (with PG patch)...")
del model_off
torch.cuda.empty_cache()

model_on = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map=device,
    trust_remote_code=True, attn_implementation="flash_attention_2",
)
model_on.eval()

# Build PG input
from verl.trainer.ppo.prefix_grouper_utils import build_pg_from_micro_batch
response_len = 32
responses = torch.randint(0, tokenizer.vocab_size, (n, response_len), device=device)
response_mask = torch.ones(n, response_len, dtype=torch.long, device=device)
uid_tensor = torch.tensor([0] * n)  # all same group

micro_batch = {
    "prompts": input_ids,
    "responses": responses,
    "response_mask": response_mask,
    "uid": uid_tensor,
    "pad_token_id": tokenizer.pad_token_id,
}

(pg, concat_ids, pg_mask, position_ids, pg_responses, pg_response_mask) = build_pg_from_micro_batch(
    micro_batch, pad_token_id=tokenizer.pad_token_id, padding_mode="right"
)

print(f"PG concat shape: {concat_ids.shape}")

# Full sequence: prompt+response for normal, grouped for PG
# Compare at prompt positions (same for all n sequences)
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    out_on = model_on(input_ids=concat_ids, attention_mask=pg_mask, position_ids=position_ids,
                       prefix_grouper=pg, use_cache=False)
    logits_on_full = out_on.logits

print(f"ON logits shape: {logits_on_full.shape}")

# Split output
prefix_out, prefix_mask_out, suffix_out, suffix_mask = pg.split_output(logits_on_full, include_prefix_last=1)
print(f"Prefix logits shape: {prefix_out.shape}")
print(f"Suffix logits shape: {suffix_out.shape}")

# Compare prefix logits (should be identical since all prompts are the same)
# prefix_out is (1, prefix_len-1, vocab) — logits at positions 0..prefix_len-2
# logits_off[0, :prefix_len-1, :] — same positions in normal forward
n_prefix_logits = prefix_out.shape[1]
prefix_logits_off = logits_off[0, :n_prefix_logits, :]

cos_sim_prefix = F.cosine_similarity(
    prefix_logits_off.float().reshape(1, -1),
    prefix_out[0].float().reshape(1, -1)
).item()
max_diff_prefix = (prefix_logits_off.float() - prefix_out[0].float()).abs().max().item()

print(f"\nPrefix comparison (positions 0..{n_prefix_logits-1}):")
print(f"  prefix_out shape: {prefix_out.shape}")
print(f"  cos_sim = {cos_sim_prefix:.6f}")
print(f"  max_diff = {max_diff_prefix:.4f}")

# Check per-position cos_sim
for pos in range(n_prefix_logits):
    cs = F.cosine_similarity(
        prefix_logits_off[pos].float().unsqueeze(0),
        prefix_out[0, pos].float().unsqueeze(0)
    ).item()
    md = (prefix_logits_off[pos].float() - prefix_out[0, pos].float()).abs().max().item()
    if pos < 5 or cs < 0.99:
        print(f"  pos {pos}: cos_sim={cs:.6f}, max_diff={md:.4f}")

# Also compare suffix logits at the first position
# suffix_out[0, 0, :] should match logits_off[0, prompt_len, :] approximately
# (because suffix token 0 follows the same prefix)
if suffix_out.shape[1] > 0:
    # suffix_out includes logits at suffix positions
    # For suffix token position j, suffix_out[0, j] = logit at prefix_len+j
    # But this only works if suffix tokens are the same for all sequences
    # Since responses are random, we can't compare suffix logits directly

    # Instead, compare the boundary: logit at position prompt_len-1
    # In normal mode, this predicts the first suffix token
    # In PG mode, this is in the prefix logits (if include_prefix_last=1)
    # or at the start of suffix logits

    print(f"\nSuffix logits shape: {suffix_out.shape}")
    print(f"  First suffix logit (seq 0, pos 0): {suffix_out[0, 0, :5].tolist()}")
    print(f"  Normal logit at prompt boundary: {logits_off[0, prompt_len-1, :5].tolist()}")

# Also check if logits_off is identical across all n sequences (since prompts are identical)
for i in range(1, n):
    cs = F.cosine_similarity(
        logits_off[0, :prompt_len, :].float().reshape(1, -1),
        logits_off[i, :prompt_len, :].float().reshape(1, -1),
    ).item()
    print(f"  OFF seq0 vs seq{i} prefix cos_sim = {cs:.6f}")

print("\nDone!")