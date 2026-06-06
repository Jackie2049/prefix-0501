#!/usr/bin/env python3
"""Direct attention-layer prefix-sharing test for Qwen3.6-27B.

Tests the PS attention mechanism at a single attention layer level,
without the full model pipeline complexity.

This test:
1. Creates mock hidden_states, cu_seqlens for a GRPO n=4 batch
2. Runs a single full-attention module WITHOUT PS → reference output
3. Runs the same module WITH PS patch → PS output
4. Compares suffix outputs for precision alignment

Usage: torchrun --nproc_per_node=4 scripts/test_ps_attention_direct.py
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
import torch.distributed as dist
from transformers import AutoConfig

# ===== Configuration =====
HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
TP_SIZE = 4
PREFIX_LEN = 16      # Short prefix for faster testing
SUFFIX_LEN = 32      # Short suffix
N_SEQUENCES = 4      # n=4 for GRPO
SEED = 42
LAYER_IDX = 3        # First full attention layer (idx 3, 7, 11, ...)

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
from verl.models.qwen3_6.megatron.layers.parallel_attention import ParallelQwen3_6AttentionRmPad

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    bf16=True,
    params_dtype=torch.bfloat16,
)

# Instantiate just one attention layer
attn = ParallelQwen3_6AttentionRmPad(config=config, megatron_config=megatron_config)
attn = attn.to(device)

# Initialize weights with small random values (not pretrained — focus on PS mechanism)
for name, param in attn.named_parameters():
    if param.dtype == torch.bfloat16:
        param.data.normal_(0, 0.01)
    elif param.dtype == torch.float32:
        param.data.normal_(0, 0.01)

print(f"[Rank {local_rank}] Attention layer instantiated: {type(attn).__name__}")
print(f"[Rank {local_rank}] num_heads_per_tp={attn.num_heads_per_tp}, "
      f"num_key_value_heads_per_tp={attn.num_key_value_heads_per_tp}, "
      f"head_dim={attn.head_dim}")
print(f"[Rank {local_rank}] q_proj output={attn.q_proj.output_size}, "
      f"k_proj output={attn.k_proj.output_size}, "
      f"v_proj output={attn.v_proj.output_size}")

# ===== Create test batch =====
total_len = PREFIX_LEN + SUFFIX_LEN
hidden_size = config.hidden_size  # 5120

# Create hidden_states for n=4 sequences, each of total_len
# Format: (total_nnz, 1, hidden_size) — THD packed format
# But we'll first test the full (padded) format, then the trimmed format

# Full sequences: each has prefix_len + suffix_len tokens
torch.manual_seed(SEED + 100)
hidden_full = torch.randn(N_SEQUENCES * total_len, 1, hidden_size,
                          dtype=torch.bfloat16, device=device)

# Build cu_seqlens for full sequences
cu_seqlens_full = torch.tensor(
    [0, total_len, 2*total_len, 3*total_len, 4*total_len],
    dtype=torch.int32, device=device
)
max_seqlen_full = total_len

# Position IDs for full sequences
position_ids_full = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

# ===== Step 1: Normal forward (no PS) =====
print(f"[Rank {local_rank}] === Step 1: Normal forward ===")

# Run the attention module on full sequences
# We need to build the "indices" parameter for unpad_input-style call
# For the RmPad forward, the attention module expects:
#   hidden_states: (total_nnz, 1, hidden)
#   position_ids: (N, total_len) or not needed for THD
#   cu_seqlens: cumulative sequence lengths
#   max_seqlen_in_batch: max sequence length
#   sequence_length: total sequence length (for RoPE)

torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    output_normal = attn(
        hidden_states=hidden_full,
        position_ids=position_ids_full,
        sequence_length=total_len,
        cu_seqlens=cu_seqlens_full,
        max_seqlen_in_batch=max_seqlen_full,
    )
torch.cuda.synchronize()
t_normal = time.time() - t0

print(f"[Rank {local_rank}] Normal output shape: {output_normal.shape}, time={t_normal:.3f}s")

# ===== Step 2: PS forward with trimmed batch =====
print(f"[Rank {local_rank}] === Step 2: PS forward ===")

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.model_spec import QWEN3_6_27B
from prefix_sharing.integrations.verl_attention import VerlQwen3_6Integration
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.backends.factory import get_backend_instance
from prefix_sharing.backends.packed_layout import PackedBatchLayout

ps_config = PrefixSharingConfig(
    enable_prefix_sharing=True,
    detector="trie",
    backend="torch_ref",
    min_prefix_len=1,
    min_group_size=2,
    boundary_strategy="prefix_last_restore",
    model_type="qwen3",  # Match HF config's model_type
)

# Create sequences for planning (same prefix, different suffixes)
sequences = []
torch.manual_seed(SEED)
prefix_ids = torch.randint(0, config.vocab_size, (PREFIX_LEN,)).tolist()
for i in range(N_SEQUENCES):
    suffix_ids = torch.randint(0, config.vocab_size, (SUFFIX_LEN,)).tolist()
    sequences.append(prefix_ids + suffix_ids)

planner = PrefixSharingPlanner(ps_config)
prefix_sharing_plan = planner.plan(sequences)

print(f"[Rank {local_rank}] PS plan: has_sharing={prefix_sharing_plan.has_sharing}, "
      f"savings_ratio={prefix_sharing_plan.savings_ratio:.1%}")
print(f"[Rank {local_rank}] kept_lengths_q={prefix_sharing_plan.kept_lengths_q}")
print(f"[Rank {local_rank}] expanded_lengths_kv={prefix_sharing_plan.expanded_lengths_kv}")

# Build trimmed hidden_states for PS
# Provider (seq 0): keeps ALL tokens (total_len)
# Reusers (seq 1-3): keeps ONLY suffix tokens (suffix_len)
kept_lengths = prefix_sharing_plan.kept_lengths_q  # [total_len, suffix_len, suffix_len, suffix_len]

# Build THD packed hidden_states for the trimmed batch
# Provider: hidden_full[0:total_len]  (all tokens)
# Reuser 1: hidden_full[total_len+PREFIX_LEN:2*total_len]  (suffix only)
# Reuser 2: hidden_full[2*total_len+PREFIX_LEN:3*total_len]  (suffix only)
# Reuser 3: hidden_full[3*total_len+PREFIX_LEN:4*total_len]  (suffix only)

trimmed_hidden_rows = []
for i in range(N_SEQUENCES):
    keep_start, keep_end = prefix_sharing_plan.input_keep_ranges[i]
    seq_start = i * total_len
    # Extract the kept portion of hidden_states for this sequence
    # Note: keep_start/keep_end are offsets within the sequence
    trimmed_row = hidden_full[seq_start + keep_start:seq_start + keep_end, :, :]
    trimmed_hidden_rows.append(trimmed_row)

# Pad each row to TP alignment and pack
align_size = TP_SIZE
padded_lengths = []
for length in kept_lengths:
    padded = ((length + align_size - 1) // align_size) * align_size
    padded_lengths.append(padded)

packed_rows = []
for i, row in enumerate(trimmed_hidden_rows):
    pad_len = padded_lengths[i] - kept_lengths[i]
    if pad_len > 0:
        padding = torch.zeros(pad_len, 1, hidden_size, dtype=torch.bfloat16, device=device)
        packed_rows.append(torch.cat([row, padding], dim=0))
    else:
        packed_rows.append(row)

trimmed_hidden = torch.cat(packed_rows, dim=0)
total_padded = sum(padded_lengths)

# Build cu_seqlens for trimmed batch
cu_seqlens_trimmed = torch.tensor(
    [0] + [sum(padded_lengths[:i+1]) for i in range(N_SEQUENCES)],
    dtype=torch.int32, device=device
)
max_seqlen_trimmed = max(padded_lengths)

# Position IDs for trimmed batch (only suffix tokens for reusers)
# Provider: positions [0..total_len-1]
# Reusers: positions [prefix_len..total_len-1]
position_ids_trimmed = torch.zeros(N_SEQUENCES, max_seqlen_trimmed, dtype=torch.long, device=device)
for i in range(N_SEQUENCES):
    offset = prefix_sharing_plan.q_position_offsets[i]
    kept_len = kept_lengths[i]
    position_ids_trimmed[i, :kept_len] = torch.arange(offset, offset + kept_len, dtype=torch.long, device=device)

print(f"[Rank {local_rank}] Trimmed: total_padded={total_padded}, max_seqlen={max_seqlen_trimmed}")
print(f"[Rank {local_rank}] cu_seqlens_trimmed={cu_seqlens_trimmed.tolist()}")

# Install PS patch
ps_integration = VerlQwen3_6Integration(ps_config)
patch_handle = ps_integration.install(model_config=config)

# Build runtime state
backend = get_backend_instance(ps_config)

# Build packed_batch_layout from kept position rows with TP alignment
# Build position rows for each sequence (only suffix tokens for reusers)
kept_position_rows = []
for i in range(N_SEQUENCES):
    offset = prefix_sharing_plan.q_position_offsets[i]
    kept_len = kept_lengths[i]
    row = torch.arange(offset, offset + kept_len, dtype=torch.long, device=device)
    kept_position_rows.append(row)

packed_batch_layout = PackedBatchLayout.from_kept_position_rows(
    kept_position_rows,
    align_size=int(align_size),
)

# Add layer_idx to the attention module (needed by PS patch)
attn.layer_idx = LAYER_IDX

runtime_state = PrefixSharingRuntimeState(
    prefix_sharing_plan=prefix_sharing_plan,
    backend=backend,
    packed_batch_layout=packed_batch_layout,
    model_spec=QWEN3_6_27B,
)

# Run PS forward
torch.cuda.synchronize()
t1 = time.time()
with prefix_sharing_runtime_context(runtime_state) as ctx:
    with torch.no_grad():
        output_ps = attn(
            hidden_states=trimmed_hidden,
            position_ids=position_ids_trimmed,
            sequence_length=max_seqlen_trimmed,
            cu_seqlens=cu_seqlens_trimmed,
            max_seqlen_in_batch=max_seqlen_trimmed,
        )
torch.cuda.synchronize()
t_ps = time.time() - t1

patch_handle.disable()

print(f"[Rank {local_rank}] PS output shape: {output_ps.shape}, time={t_ps:.3f}s")

# ===== Step 3: Compare outputs =====
print(f"[Rank {local_rank}] === Step 3: Precision alignment ===")

# Normal output: (N*total_len, 1, hidden_size) in packed format
# Extract per-sequence outputs from normal
normal_outputs = []
for i in range(N_SEQUENCES):
    start = i * total_len
    end = start + total_len
    normal_outputs.append(output_normal[start:end, 0, :])  # (total_len, hidden)

# PS output: (total_padded, 1, hidden_size) in packed format
# Extract per-sequence outputs from PS
ps_outputs = []
for i in range(N_SEQUENCES):
    start = cu_seqlens_trimmed[i].item()
    kept_len = kept_lengths[i]
    ps_outputs.append(output_ps[start:start + kept_len, 0, :])  # (kept_len, hidden)

# Compare suffix outputs
results = {}
all_cos_sims = []
all_max_diffs = []

for i in range(N_SEQUENCES):
    if prefix_sharing_plan.is_reuser(i):
        # Reuser: compare suffix only
        normal_suffix = normal_outputs[i][PREFIX_LEN:, :]  # (suffix_len, hidden)
        ps_suffix = ps_outputs[i]  # (suffix_len, hidden) — already suffix only
    else:
        # Provider: compare suffix portion
        normal_suffix = normal_outputs[i][PREFIX_LEN:, :]  # (suffix_len, hidden)
        ps_suffix = ps_outputs[i][PREFIX_LEN:, :]  # (suffix_len, hidden)

    cos_sim = torch.nn.functional.cosine_similarity(
        ps_suffix.float(), normal_suffix.float(), dim=-1
    ).mean().item()

    max_diff = (ps_suffix.float() - normal_suffix.float()).abs().max().item()
    mean_diff = (ps_suffix.float() - normal_suffix.float()).abs().mean().item()

    all_cos_sims.append(cos_sim)
    all_max_diffs.append(max_diff)

    role = "reuser" if prefix_sharing_plan.is_reuser(i) else "provider"
    if local_rank == 0:
        print(f"[Rank 0] Seq {i} ({role}): cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}, "
              f"mean_diff={mean_diff:.6f}")

    results[i] = {"role": role, "cos_sim": cos_sim, "max_diff": max_diff, "mean_diff": mean_diff}

overall_cos = sum(all_cos_sims) / len(all_cos_sims)
overall_max = max(all_max_diffs)

if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"ATTENTION-LAYER PS PRECISION ALIGNMENT")
    print(f"{'='*60}")
    print(f"Layer {LAYER_IDX} (full attention)")
    print(f"N={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"Overall cos_sim: {overall_cos:.6f}")
    print(f"Overall max_diff: {overall_max:.6f}")
    print(f"Normal time: {t_normal:.3f}s")
    print(f"PS time: {t_ps:.3f}s")
    if overall_cos >= 0.999:
        print("PASS: PS precision alignment within bf16 tolerance!")
    else:
        print(f"FAIL: cos_sim {overall_cos:.6f} < 0.999")
    print(f"{'='*60}")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()