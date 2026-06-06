#!/usr/bin/env python3
"""End-to-end test for two-pass PS forward using the helper function.

Validates that the run_two_pass_prefix_sharing_forward helper function
produces logits that match the normal forward logits (cos_sim > 0.999).

This test uses the padded model (ParallelQwen3_6ForCausalLM) with
pretrained weights and the two_pass_forward helper.

Usage: torchrun --nproc_per_node=4 scripts/test_two_pass_forward.py
"""
import os
import sys
import time

verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_sharing_path = os.path.expanduser("~/rollout-prefix/prefix-0501/prefix-sharing")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path)
sys.path.insert(0, prefix_sharing_path)
sys.path.insert(0, prefix_path)

import torch
import torch.nn.functional as F
from transformers import AutoConfig

# ===== Configuration =====
HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
TP_SIZE = 4
PREFIX_LEN = 64
SUFFIX_LEN = 64
N_SEQUENCES = 4
SEED = 42

# ===== Initialize distributed =====
torch.distributed.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker, get_cuda_rng_tracker, _MODEL_PARALLEL_RNG_TRACKER_NAME,
)
initialize_rng_tracker(use_te_rng_tracker=False, inference_rng_tracker=False)
rng_tracker = get_cuda_rng_tracker()
rng_tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed=SEED)

from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
)

tp_rank = parallel_state.get_tensor_model_parallel_rank()
device = torch.device(f"cuda:{local_rank}")

# ===== Load config =====
config = AutoConfig.from_pretrained(HF_MODEL_PATH)

from megatron.core import ModelParallelConfig
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLM

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    bf16=True,
    params_dtype=torch.bfloat16,
)

# Instantiate model
model = ParallelQwen3_6ForCausalLM(config, megatron_config)
model = model.to(device)

# Load pretrained weights
from scripts.load_weights_qwen36 import load_qwen36_weights
load_qwen36_weights(model, HF_MODEL_PATH, device, tp_rank=tp_rank)

if local_rank == 0:
    print("Model loaded with pretrained weights")

# ===== Create test token sequences =====
# For this test, we use actual token IDs to test the full pipeline
# (embed_tokens → layers → norm → lm_head)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)

# Create test prompts
prompt_text = "The quick brown fox jumps over the lazy dog. This is a test prompt for prefix-sharing."
suffix_texts = [
    " Response one: The fox was very agile.",
    " Response two: The dog was sleeping peacefully.",
    " Response three: They met in the forest.",
    " Response four: Both animals were friendly.",
]

# Tokenize
prefix_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
suffix_tokens_list = [tokenizer.encode(s, add_special_tokens=False) for s in suffix_texts]

# Adjust lengths to match PREFIX_LEN and SUFFIX_LEN
# Trim or pad prefix
if len(prefix_tokens) > PREFIX_LEN:
    prefix_tokens = prefix_tokens[:PREFIX_LEN]
else:
    prefix_tokens = prefix_tokens + [tokenizer.pad_token_id or 0] * (PREFIX_LEN - len(prefix_tokens))

# Trim or pad suffixes
for i in range(len(suffix_tokens_list)):
    if len(suffix_tokens_list[i]) > SUFFIX_LEN:
        suffix_tokens_list[i] = suffix_tokens_list[i][:SUFFIX_LEN]
    else:
        suffix_tokens_list[i] = suffix_tokens_list[i] + [tokenizer.pad_token_id or 0] * (SUFFIX_LEN - len(suffix_tokens_list[i]))

# Build full sequences (prefix + suffix_i)
full_sequences = []
for suffix in suffix_tokens_list:
    full_seq = prefix_tokens + suffix
    full_sequences.append(full_seq)

# Convert to tensors
full_input_ids = torch.tensor(full_sequences, dtype=torch.long, device=device)
suffix_input_ids = torch.tensor(suffix_tokens_list, dtype=torch.long, device=device)
prefix_input_ids = torch.tensor([prefix_tokens], dtype=torch.long, device=device)

# Attention masks (all tokens are valid in this test)
full_attention_mask = torch.ones_like(full_input_ids, dtype=torch.bool, device=device)
suffix_attention_mask = torch.ones_like(suffix_input_ids, dtype=torch.bool, device=device)
prefix_attention_mask = torch.ones_like(prefix_input_ids, dtype=torch.bool, device=device)

# Position IDs
prefix_position_ids = torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0)
suffix_position_ids = torch.arange(PREFIX_LEN, PREFIX_LEN + SUFFIX_LEN, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

# ===== Step 1: Normal forward (reference) =====
if local_rank == 0:
    print("=== Step 1: Normal forward ===")

torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    output = model(input_ids=full_input_ids, attention_mask=full_attention_mask, position_ids=torch.arange(PREFIX_LEN + SUFFIX_LEN, device=device).unsqueeze(0).expand(N_SEQUENCES, -1))
    logits_normal = output.logits
torch.cuda.synchronize()
t_normal = time.time() - t0

if local_rank == 0:
    print(f"Normal logits shape: {logits_normal.shape}, time={t_normal:.3f}s")

# ===== Step 2: Two-pass PS forward =====
if local_rank == 0:
    print("=== Step 2: Two-pass PS forward ===")

from prefix_sharing.integrations.two_pass_forward import run_two_pass_prefix_sharing_forward

torch.cuda.synchronize()
t_ps_start = time.time()
with torch.no_grad():
    logits_ps = run_two_pass_prefix_sharing_forward(
        model=model,
        prefix_input_ids=prefix_input_ids,
        prefix_attention_mask=prefix_attention_mask,
        prefix_position_ids=prefix_position_ids,
        suffix_input_ids=suffix_input_ids,
        suffix_attention_mask=suffix_attention_mask,
        suffix_position_ids=suffix_position_ids,
        prefix_len=PREFIX_LEN,
        config=config,
    )
torch.cuda.synchronize()
t_ps = time.time() - t_ps_start

if local_rank == 0:
    print(f"PS logits shape: {logits_ps.shape}, time={t_ps:.3f}s")

# ===== Step 3: Compare logits =====
if local_rank == 0:
    print("=== Step 3: Precision alignment ===")

# Extract suffix portion from normal logits
normal_suffix_logits = logits_normal[:, PREFIX_LEN:, :]  # (N, SUFFIX_LEN, vocab_size)
ps_suffix_logits = logits_ps[:, :, :]  # (N, SUFFIX_LEN, vocab_size) or (N, SUFFIX_LEN, vocab_size)

# Ensure shapes match
if normal_suffix_logits.shape != ps_suffix_logits.shape:
    if local_rank == 0:
        print(f"Shape mismatch: normal={normal_suffix_logits.shape}, ps={ps_suffix_logits.shape}")
        # Try to align
        min_len = min(normal_suffix_logits.shape[1], ps_suffix_logits.shape[1])
        normal_suffix_logits = normal_suffix_logits[:, :min_len, :]
        ps_suffix_logits = ps_suffix_logits[:, :min_len, :]

all_cos_sims = []
all_max_diffs = []

for i in range(N_SEQUENCES):
    cos_sim = F.cosine_similarity(
        ps_suffix_logits[i].float().flatten(),
        normal_suffix_logits[i].float().flatten(),
        dim=0,
    ).item()
    max_diff = (ps_suffix_logits[i].float() - normal_suffix_logits[i].float()).abs().max().item()
    mean_diff = (ps_suffix_logits[i].float() - normal_suffix_logits[i].float()).abs().mean().item()

    all_cos_sims.append(cos_sim)
    all_max_diffs.append(max_diff)

    if local_rank == 0:
        print(f"Seq {i}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

overall_cos = sum(all_cos_sims) / len(all_cos_sims)
overall_max = max(all_max_diffs)

if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"TWO-PASS PS FORWARD HELPER E2E TEST")
    print(f"{'='*60}")
    print(f"Model: Qwen3.6-27B (TP=4, bf16)")
    print(f"N={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"Overall cos_sim: {overall_cos:.6f}")
    print(f"Overall max_diff: {overall_max:.6f}")
    print(f"Normal time: {t_normal:.3f}s")
    print(f"PS time: {t_ps:.3f}s")
    if overall_cos >= 0.999:
        print("PASS: Two-pass PS forward helper precision alignment!")
    else:
        print(f"FAIL: cos_sim {overall_cos:.6f} < 0.999")
    print(f"{'='*60}")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()