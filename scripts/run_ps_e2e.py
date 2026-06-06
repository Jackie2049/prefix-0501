#!/usr/bin/env python3
"""E2E prefix-sharing precision alignment test for Qwen3.6-27B.

Tests that prefix-sharing produces numerically equivalent results
compared to normal (non-PS) forward pass, within bf16 tolerance.

Test pattern: GRPO n=8 — one prompt with 8 response suffixes.
All 8 sequences share the same prefix (the prompt).

Steps:
1. Load model with pretrained weights (TP=4 bf16)
2. Run normal forward on 8 full sequences → reference logits
3. Run prefix-sharing forward on trimmed batch → PS logits
4. Compare suffix logits: cos_sim > 0.999, max_diff < 0.01 (bf16)

PS approach:
- Build trimmed attention_mask: provider sees full seq, reusers see suffix only
- Build position_ids with offsets for reusers (start at prefix_len)
- Model forward() calls unpad_input naturally on the trimmed mask
- PS attention patch intercepts attention forward to expand KV with prefix data

Usage: torchrun --nproc_per_node=4 scripts/run_ps_e2e.py
"""
import os
import sys
import time
import json

verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path)
sys.path.insert(0, prefix_path)

import torch
import torch.distributed as dist
from transformers import AutoConfig
from safetensors.torch import load_file

# ===== Configuration =====
HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
TP_SIZE = 4
PREFIX_LEN = 32      # Length of shared prefix (prompt)
SUFFIX_LEN = 64      # Length of each suffix (response)
N_SEQUENCES = 8      # Number of sequences (GRPO n=8 pattern)
SEED = 42

# ===== Initialize distributed =====
torch.distributed.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    get_cuda_rng_tracker,
    _MODEL_PARALLEL_RNG_TRACKER_NAME,
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
print(f"[Rank {local_rank}] Config: {type(config).__name__}, hidden={config.hidden_size}, "
      f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
      f"head_dim={config.head_dim}, layers={config.num_hidden_layers}")

# ===== Instantiate model =====
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLMRmPad
from megatron.core import ModelParallelConfig

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    bf16=True,
    params_dtype=torch.bfloat16,
)

model = ParallelQwen3_6ForCausalLMRmPad(config=config, megatron_config=megatron_config)
model = model.to(device)

# ===== Load pretrained weights =====
print(f"[Rank {local_rank}] Loading pretrained weights...")
hf_state_dict = {}
for i in range(1, 12):
    shard_path = os.path.join(HF_MODEL_PATH, f"model.safetensors-{i:05d}-of-00011.safetensors")
    shard_dict = load_file(shard_path)
    hf_state_dict.update(shard_dict)

# Filter only language_model keys
hf_keys_filtered = {}
for k, v in hf_state_dict.items():
    if k.startswith("model.language_model.") or k.startswith("lm_head."):
        hf_keys_filtered[k] = v

def shard_tensor(tensor, dim, tp_size, tp_rank):
    chunks = torch.chunk(tensor, tp_size, dim=dim)
    return chunks[tp_rank].contiguous()

def split_deltanet_qkv(in_proj_qkv_weight, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    q_portion = in_proj_qkv_weight[:key_dim]
    k_portion = in_proj_qkv_weight[key_dim:key_dim*2]
    v_portion = in_proj_qkv_weight[key_dim*2:]
    q_shard = shard_tensor(q_portion, 0, tp_size, tp_rank)
    k_shard = shard_tensor(k_portion, 0, tp_size, tp_rank)
    v_shard = shard_tensor(v_portion, 0, tp_size, tp_rank)
    return q_shard, k_shard, v_shard

def shard_conv1d_weight(conv1d_weight, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    q_portion = conv1d_weight[:key_dim]
    k_portion = conv1d_weight[key_dim:key_dim*2]
    v_portion = conv1d_weight[key_dim*2:]
    q_shard = shard_tensor(q_portion, 0, tp_size, tp_rank)
    k_shard = shard_tensor(k_portion, 0, tp_size, tp_rank)
    v_shard = shard_tensor(v_portion, 0, tp_size, tp_rank)
    return torch.cat([q_shard, k_shard, v_shard], dim=0).contiguous()

def load_mlp_weights(hf_sd, mlp_module, prefix, tp_size, tp_rank):
    loaded = 0
    gate_key = f"{prefix}.mlp.gate_proj.weight"
    up_key = f"{prefix}.mlp.up_proj.weight"
    if gate_key in hf_sd and up_key in hf_sd:
        gate_shard = shard_tensor(hf_sd[gate_key], 0, tp_size, tp_rank)
        up_shard = shard_tensor(hf_sd[up_key], 0, tp_size, tp_rank)
        gate_up_shard = torch.cat([gate_shard, up_shard], dim=0).contiguous()
        mlp_module.gate_up_proj.weight.data.copy_(gate_up_shard.to(torch.bfloat16))
        loaded += 2
    down_key = f"{prefix}.mlp.down_proj.weight"
    if down_key in hf_sd:
        shard = shard_tensor(hf_sd[down_key], 1, tp_size, tp_rank)
        mlp_module.down_proj.weight.data.copy_(shard.to(torch.bfloat16))
        loaded += 1
    return loaded

layer_types = config.layer_types
loaded_count = 0

for layer_idx in range(config.num_hidden_layers):
    is_deltanet = (layer_types[layer_idx] == "linear_attention")
    decoder_layer = model.model.layers[layer_idx]
    attn_module = decoder_layer.self_attn
    mlp_module = decoder_layer.mlp
    hf_layer_prefix = f"model.language_model.layers.{layer_idx}"

    if is_deltanet:
        hf_prefix = f"{hf_layer_prefix}.linear_attn"
        key = f"{hf_prefix}.in_proj_qkv.weight"
        if key in hf_keys_filtered:
            q_s, k_s, v_s = split_deltanet_qkv(hf_keys_filtered[key], config, TP_SIZE, tp_rank)
            attn_module.in_proj_q.weight.data.copy_(q_s.to(torch.bfloat16))
            attn_module.in_proj_k.weight.data.copy_(k_s.to(torch.bfloat16))
            attn_module.in_proj_v.weight.data.copy_(v_s.to(torch.bfloat16))
            loaded_count += 3
        key = f"{hf_prefix}.conv1d.weight"
        if key in hf_keys_filtered:
            attn_module.conv1d.weight.data.copy_(
                shard_conv1d_weight(hf_keys_filtered[key], config, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for proj_name in ["in_proj_z", "in_proj_b", "in_proj_a"]:
            key = f"{hf_prefix}.{proj_name}.weight"
            if key in hf_keys_filtered:
                shard = shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank)
                getattr(attn_module, proj_name).weight.data.copy_(shard.to(torch.bfloat16))
                loaded_count += 1
        key = f"{hf_prefix}.A_log"
        if key in hf_keys_filtered:
            attn_module.A_log.data.copy_(
                shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank).to(torch.float32))
            loaded_count += 1
        key = f"{hf_prefix}.dt_bias"
        if key in hf_keys_filtered:
            attn_module.dt_bias.data.copy_(
                shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        key = f"{hf_prefix}.norm.weight"
        if key in hf_keys_filtered:
            attn_module.norm.weight.data.copy_(hf_keys_filtered[key].to(torch.bfloat16))
            loaded_count += 1
        key = f"{hf_prefix}.out_proj.weight"
        if key in hf_keys_filtered:
            attn_module.out_proj.weight.data.copy_(
                shard_tensor(hf_keys_filtered[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
    else:
        hf_prefix = f"{hf_layer_prefix}.self_attn"
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            key = f"{hf_prefix}.{proj_name}.weight"
            if key in hf_keys_filtered:
                shard = shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank)
                getattr(attn_module, proj_name).weight.data.copy_(shard.to(torch.bfloat16))
                loaded_count += 1
        key = f"{hf_prefix}.o_proj.weight"
        if key in hf_keys_filtered:
            attn_module.o_proj.weight.data.copy_(
                shard_tensor(hf_keys_filtered[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for norm_name in ["q_norm", "k_norm"]:
            key = f"{hf_prefix}.{norm_name}.weight"
            if key in hf_keys_filtered:
                getattr(attn_module, norm_name).weight.data.copy_(hf_keys_filtered[key].to(torch.bfloat16))
                loaded_count += 1

    loaded_count += load_mlp_weights(hf_keys_filtered, mlp_module, hf_layer_prefix, TP_SIZE, tp_rank)

    for ln_name in ["input_layernorm", "post_attention_layernorm"]:
        key = f"{hf_layer_prefix}.{ln_name}.weight"
        if key in hf_keys_filtered:
            getattr(decoder_layer, ln_name).weight.data.copy_(hf_keys_filtered[key].to(torch.bfloat16))
            loaded_count += 1

# Non-layer weights
embed_key = "model.language_model.embed_tokens.weight"
if embed_key in hf_keys_filtered:
    model.model.embed_tokens.weight.data.copy_(
        shard_tensor(hf_keys_filtered[embed_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
    loaded_count += 1
norm_key = "model.language_model.norm.weight"
if norm_key in hf_keys_filtered:
    model.model.norm.weight.data.copy_(hf_keys_filtered[norm_key].to(torch.bfloat16))
    loaded_count += 1
lm_key = "lm_head.weight"
if lm_key in hf_keys_filtered:
    model.lm_head.weight.data.copy_(
        shard_tensor(hf_keys_filtered[lm_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
    loaded_count += 1

# Free HF state_dict
hf_state_dict.clear()
hf_keys_filtered.clear()

print(f"[Rank {local_rank}] Loaded {loaded_count} weights, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ===== Create test batch: GRPO n=8 pattern =====
vocab_size = config.vocab_size
torch.manual_seed(SEED)
prefix_ids = torch.randint(0, vocab_size, (PREFIX_LEN,), device="cpu").tolist()

sequences = []
for i in range(N_SEQUENCES):
    suffix_ids = torch.randint(0, vocab_size, (SUFFIX_LEN,), device="cpu").tolist()
    full_seq = prefix_ids + suffix_ids
    sequences.append(full_seq)

total_len = PREFIX_LEN + SUFFIX_LEN
print(f"[Rank {local_rank}] Test: N={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}, total={total_len}")

# ===== Step 1: Normal forward (no prefix-sharing) =====
print(f"[Rank {local_rank}] === Step 1: Normal forward (no PS) ===")

input_ids_normal = torch.tensor(sequences, dtype=torch.long, device=device)
attention_mask_normal = torch.ones((N_SEQUENCES, total_len), dtype=torch.long, device=device)
position_ids_normal = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

torch.cuda.synchronize()
t_start = time.time()
with torch.no_grad():
    output_normal = model(
        input_ids=input_ids_normal,
        attention_mask=attention_mask_normal,
        position_ids=position_ids_normal,
    )
torch.cuda.synchronize()
t_normal = time.time() - t_start

# Get reference logits (already gathered in model._forward_head)
logits_normal = output_normal.logits  # (N, total_len, vocab_size) — full vocab gathered
# Note: the model's _forward_head does gather_from_tensor_model_parallel_region,
# so logits_normal already has full vocab_size

if local_rank == 0:
    print(f"[Rank 0] Normal forward: {t_normal:.3f}s, logits shape={logits_normal.shape}")
    print(f"[Rank 0] Normal logits: mean={logits_normal.mean().item():.4f}, "
          f"std={logits_normal.std().item():.4f}")
    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"[Rank 0] GPU memory: {alloc_gb:.2f} GB")

# ===== Step 2: Prefix-sharing forward =====
print(f"[Rank {local_rank}] === Step 2: Prefix-sharing forward ===")

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.model_spec import QWEN3_6_27B

ps_config = PrefixSharingConfig(
    enable_prefix_sharing=True,
    detector="trie",
    backend="torch_ref",
    min_prefix_len=1,
    min_group_size=2,
    boundary_strategy="prefix_last_restore",
)

planner = PrefixSharingPlanner(ps_config)
prefix_sharing_plan = planner.plan(sequences)

print(f"[Rank {local_rank}] PS plan: has_sharing={prefix_sharing_plan.has_sharing}, "
      f"tokens_saved={prefix_sharing_plan.tokens_saved}/{prefix_sharing_plan.total_original_tokens}, "
      f"savings_ratio={prefix_sharing_plan.savings_ratio:.1%}")

# Build trimmed batch:
# - input_ids: same full sequences (the model reads all token IDs)
# - attention_mask: provider (seq 0) = full mask, reusers = mask only suffix tokens
# - position_ids: provider = [0..total_len], reusers = [prefix_len..total_len] for suffix,
#                 [0..prefix_len] masked out (padding)
input_ids_ps = input_ids_normal.clone()  # Same input_ids — model reads all tokens
attention_mask_ps = torch.zeros((N_SEQUENCES, total_len), dtype=torch.long, device=device)
position_ids_ps = torch.zeros((N_SEQUENCES, total_len), dtype=torch.long, device=device)

for i in range(N_SEQUENCES):
    keep_start, keep_end = prefix_sharing_plan.input_keep_ranges[i]
    # attention_mask: 1 for kept positions, 0 for trimmed prefix positions
    attention_mask_ps[i, keep_start:keep_end] = 1
    # position_ids: proper positions for kept tokens
    offset = prefix_sharing_plan.q_position_offsets[i]
    kept_len = keep_end - keep_start
    position_ids_ps[i, keep_start:keep_end] = torch.arange(
        offset, offset + kept_len, dtype=torch.long, device=device
    )
    # For padded positions (trimmed prefix for reusers): position_ids = 0 (padding)
    # This is fine because unpad_input will strip them

print(f"[Rank {local_rank}] PS trimmed mask: seq 0 (provider) keeps full, seq 1-7 (reusers) "
      f"keep suffix only [prefix_len={PREFIX_LEN}..{total_len}]")

# Install prefix-sharing patches on attention modules
from prefix_sharing.integrations.verl_attention import VerlQwen3_6Integration
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.packed_layout import PackedBatchLayout

ps_integration = VerlQwen3_6Integration(ps_config)
patch_handle = ps_integration.install(model_config=config)

# Build runtime state for PS context
# The runtime context provides prefix_sharing_plan, backend, store, etc. to the patched attention
backend = get_backend_instance(ps_config)

# Build packed_batch_layout from kept_lengths
# This will be used by the PS attention patch to know how each sequence maps in the packed (THD) format
kept_lengths = prefix_sharing_plan.kept_lengths_q
align_size = TP_SIZE

# We need to compute what the model's unpad_input will produce for the trimmed mask
# unpad_input flattens all non-zero attention_mask positions into a packed 1D tensor
# For our trimmed mask:
#   seq 0 (provider): total_len non-zero positions → total_len tokens in packed
#   seq 1-7 (reusers): suffix_len non-zero positions → suffix_len tokens in packed
# Total packed length = total_len + 7 * suffix_len

# The model will compute its own cu_seqlens from unpad_input.
# We need packed_batch_layout to match that format.
# PackedBatchLayout.from_valid_lengths will compute padded lengths with alignment

packed_batch_layout = PackedBatchLayout.from_valid_lengths(kept_lengths, align_size=int(align_size))

print(f"[Rank {local_rank}] PackedBatchLayout: valid_lengths={packed_batch_layout.valid_lengths}, "
      f"padded_lengths={packed_batch_layout.padded_lengths}, "
      f"cu_seqlens={packed_batch_layout.cu_seqlens}")

runtime_state = PrefixSharingRuntimeState(
    prefix_sharing_plan=prefix_sharing_plan,
    backend=backend,
    packed_batch_layout=packed_batch_layout,
    model_spec=QWEN3_6_27B,
)

# Run PS forward within runtime context
torch.cuda.synchronize()
t_ps_start = time.time()

with prefix_sharing_runtime_context(runtime_state) as ctx:
    with torch.no_grad():
        output_ps = model(
            input_ids=input_ids_ps,
            attention_mask=attention_mask_ps,
            position_ids=position_ids_ps,
        )

torch.cuda.synchronize()
t_ps = time.time() - t_ps_start

# Remove patches
patch_handle.disable()

logits_ps = output_ps.logits  # (N, total_len, vocab_size)

if local_rank == 0:
    print(f"[Rank 0] PS forward: {t_ps:.3f}s")
    print(f"[Rank 0] PS logits shape: {logits_ps.shape}")
    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"[Rank 0] GPU memory: {alloc_gb:.2f} GB")

# ===== Step 3: Precision alignment check =====
print(f"[Rank {local_rank}] === Step 3: Precision alignment check ===")

# Compare suffix logits between normal and PS forward.
# Normal: logits at all positions (prefix + suffix) for every sequence
# PS: logits at kept positions only — 0 at trimmed positions (padding)
# For reusers: PS logits at prefix positions are 0 (from pad_input restore),
#              PS logits at suffix positions should match normal suffix logits
# For provider: PS logits at all positions should match normal logits

results = {
    "config": {
        "n_sequences": N_SEQUENCES,
        "prefix_len": PREFIX_LEN,
        "suffix_len": SUFFIX_LEN,
        "total_len": total_len,
        "tp_size": TP_SIZE,
    },
    "normal_time": t_normal,
    "ps_time": t_ps,
    "per_sequence": {},
}

all_cos_sims = []
all_max_diffs = []
all_mean_diffs = []

for seq_idx in range(N_SEQUENCES):
    # Extract suffix logits for comparison
    # For reusers: only suffix positions have valid logits in PS output
    # For provider: all positions have valid logits

    if prefix_sharing_plan.is_reuser(seq_idx):
        # Compare suffix logits only (prefix positions in PS are padded/zero)
        normal_suffix = logits_normal[seq_idx, PREFIX_LEN:total_len, :]
        ps_suffix = logits_ps[seq_idx, PREFIX_LEN:total_len, :]
        compare_len = SUFFIX_LEN
    else:
        # Provider: compare suffix portion (prefix is handled normally)
        normal_suffix = logits_normal[seq_idx, PREFIX_LEN:total_len, :]
        ps_suffix = logits_ps[seq_idx, PREFIX_LEN:total_len, :]
        compare_len = SUFFIX_LEN

    # Compute alignment metrics
    per_pos_cos = torch.nn.functional.cosine_similarity(
        ps_suffix.float(), normal_suffix.float(), dim=-1
    )
    mean_cos = per_pos_cos.mean().item()
    min_cos = per_pos_cos.min().item()

    diff = (ps_suffix.float() - normal_suffix.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    all_cos_sims.append(mean_cos)
    all_max_diffs.append(max_diff)
    all_mean_diffs.append(mean_diff)

    results["per_sequence"][seq_idx] = {
        "is_reuser": prefix_sharing_plan.is_reuser(seq_idx),
        "prefix_len": prefix_sharing_plan.prefix_lens[seq_idx],
        "suffix_len": prefix_sharing_plan.suffix_lens[seq_idx],
        "mean_cos_sim": mean_cos,
        "min_cos_sim": min_cos,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
    }

    if local_rank == 0:
        role = "reuser" if prefix_sharing_plan.is_reuser(seq_idx) else "provider"
        print(f"[Rank 0] Seq {seq_idx} ({role}): cos_sim={mean_cos:.6f} (min={min_cos:.6f}), "
              f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

# ===== Summary =====
overall_cos_sim = sum(all_cos_sims) / len(all_cos_sims)
overall_max_diff = max(all_max_diffs)
overall_mean_diff = sum(all_mean_diffs) / len(all_mean_diffs)

results["overall"] = {
    "mean_cos_sim": overall_cos_sim,
    "max_diff": overall_max_diff,
    "mean_diff": overall_mean_diff,
}

COS_SIM_THRESHOLD = 0.999
MAX_DIFF_THRESHOLD = 0.01

passed = overall_cos_sim >= COS_SIM_THRESHOLD and overall_max_diff <= MAX_DIFF_THRESHOLD

if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"E2E PREFIX-SHARING PRECISION ALIGNMENT RESULTS")
    print(f"{'='*60}")
    print(f"Model: Qwen3.6-27B (TP={TP_SIZE}, bf16)")
    print(f"Pattern: GRPO n={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"PS savings: {prefix_sharing_plan.tokens_saved}/{prefix_sharing_plan.total_original_tokens} "
          f"({prefix_sharing_plan.savings_ratio:.1%})")
    print(f"Normal forward: {t_normal:.3f}s")
    print(f"PS forward: {t_ps:.3f}s")
    if t_ps > 0:
        print(f"Speedup ratio: {t_normal/t_ps:.2f}x")
    print(f"{'='*60}")
    print(f"Overall cos_sim: {overall_cos_sim:.6f} (threshold: {COS_SIM_THRESHOLD})")
    print(f"Overall max_diff: {overall_max_diff:.6f} (threshold: {MAX_DIFF_THRESHOLD})")
    print(f"Overall mean_diff: {overall_mean_diff:.6f}")
    print(f"{'='*60}")
    if passed:
        print("PASS: Prefix-sharing precision alignment within bf16 tolerance!")
    else:
        print("FAIL: Precision misalignment detected!")
        if overall_cos_sim < COS_SIM_THRESHOLD:
            print(f"  - cos_sim {overall_cos_sim:.6f} < {COS_SIM_THRESHOLD}")
        if overall_max_diff > MAX_DIFF_THRESHOLD:
            print(f"  - max_diff {overall_max_diff:.6f} > {MAX_DIFF_THRESHOLD}")
    print(f"{'='*60}")

    results["passed"] = passed
    results_path = os.path.expanduser("~/rollout-prefix/ps_e2e_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()