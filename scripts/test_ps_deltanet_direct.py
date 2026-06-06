#!/usr/bin/env python3
"""Per-layer DeltaNet prefix-sharing precision test for Qwen3.6-27B.

Validates that DeltaNet state injection produces numerically equivalent
suffix outputs compared to processing the full sequence.

Test pattern:
1. Process full sequence (prefix + suffix) through DeltaNet → reference output
2. Process prefix only → extract recurrent state
3. Process suffix only with prefix state as initial_state → PS output
4. Compare: PS suffix output should match reference suffix output

This proves that DeltaNet prefix-sharing via state injection is feasible
and achieves precision alignment within bf16 tolerance.

Usage: torchrun --nproc_per_node=4 scripts/test_ps_deltanet_direct.py
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
import torch.nn.functional as F
from transformers import AutoConfig
from verl.models.qwen3_6.megatron.layers.parallel_deltanet import (
    ParallelQwen3_6GatedDeltaNet,
    torch_chunk_gated_delta_rule,
)

# ===== Configuration =====
HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
TP_SIZE = 4
PREFIX_LEN = 64      # Must be >= chunk_size (64) for proper chunk boundary alignment
SUFFIX_LEN = 64      # Suffix length (same size as prefix for balanced test)
N_SEQUENCES = 4      # n=4 for GRPO
SEED = 42
LAYER_IDX = 0        # First DeltaNet layer (idx 0,1,2,...)

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
from verl.models.qwen3_6.megatron.layers.parallel_deltanet import (
    ParallelQwen3_6GatedDeltaNet,
)

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    bf16=True,
    params_dtype=torch.bfloat16,
)

# Instantiate just one DeltaNet layer
deltanet = ParallelQwen3_6GatedDeltaNet(config=config, megatron_config=megatron_config)
deltanet = deltanet.to(device)

# Initialize weights with small random values (focus on PS mechanism, not pretrained)
for name, param in deltanet.named_parameters():
    if param.dtype == torch.bfloat16:
        param.data.normal_(0, 0.01)
    elif param.dtype == torch.float32:
        param.data.normal_(0, 0.01)

print(f"[Rank {local_rank}] DeltaNet layer instantiated: {type(deltanet).__name__}")
print(f"[Rank {local_rank}] num_k_heads_per_tp={deltanet.num_k_heads_per_tp}, "
      f"num_v_heads_per_tp={deltanet.num_v_heads_per_tp}, "
      f"head_k_dim={deltanet.head_k_dim}, head_v_dim={deltanet.head_v_dim}")
print(f"[Rank {local_rank}] key_dim_per_tp={deltanet.key_dim_per_tp}, "
      f"value_dim_per_tp={deltanet.value_dim_per_tp}")

hidden_size = config.hidden_size  # 5120

# ===== Create test batch =====
total_len = PREFIX_LEN + SUFFIX_LEN

# CRITICAL: prefix hidden_states must be identical across all sequences
torch.manual_seed(SEED + 100)
prefix_hidden = torch.randn(1, PREFIX_LEN, hidden_size, dtype=torch.bfloat16, device=device)
suffix_hiddens = [torch.randn(1, SUFFIX_LEN, hidden_size, dtype=torch.bfloat16, device=device)
                  for _ in range(N_SEQUENCES)]

# Build full hidden_states: for each sequence, [prefix, suffix_i]
full_sequences = []
for i in range(N_SEQUENCES):
    # Each sequence: (1, total_len, hidden_size)
    seq_hidden = torch.cat([prefix_hidden, suffix_hiddens[i]], dim=1)
    full_sequences.append(seq_hidden)
# Stack into batch: (N, total_len, hidden_size)
hidden_full = torch.cat(full_sequences, dim=0)

# ===== Step 1: Normal forward (full sequences) =====
print(f"[Rank {local_rank}] === Step 1: Normal forward ===")

torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    output_normal = deltanet(hidden_full)  # (N, total_len, hidden_size)
torch.cuda.synchronize()
t_normal = time.time() - t0

print(f"[Rank {local_rank}] Normal output shape: {output_normal.shape}, time={t_normal:.3f}s")

# ===== Step 2: Prefix-only forward → extract recurrent state =====
print(f"[Rank {local_rank}] === Step 2: Prefix-only forward → extract state ===")

torch.cuda.synchronize()
t1 = time.time()
with torch.no_grad():
    # Process just the prefix (1 sequence, PREFIX_LEN tokens)
    prefix_output, prefix_state = deltanet(
        prefix_hidden,  # (1, PREFIX_LEN, hidden_size)
        initial_state=None,
        output_final_state=True,
    )
torch.cuda.synchronize()
t_prefix = time.time() - t1

print(f"[Rank {local_rank}] Prefix output shape: {prefix_output.shape}")
print(f"[Rank {local_rank}] Prefix state shape: {prefix_state.shape}")
print(f"[Rank {local_rank}] Prefix time: {t_prefix:.3f}s")

# prefix_state shape: (1, num_v_heads_per_tp, head_k_dim, head_v_dim)
# For state injection, we need to expand this to N_SEQUENCES (all reusers share same prefix state)
# The state is the accumulated recurrent state after processing the prefix tokens
prefix_state_expanded = prefix_state.expand(N_SEQUENCES, -1, -1, -1)  # (N, v_heads, k_dim, v_dim)

# ===== Step 3: Suffix-only forward with injected prefix state =====
print(f"[Rank {local_rank}] === Step 3: Suffix-only forward with injected state ===")

# CRITICAL: conv1d context — the causal conv1d (kernel_size=4) needs the
# last kernel_size-1=3 prefix tokens as context for the first suffix tokens.
# But the overlap tokens must NOT be included in the chunk_gated_delta_rule
# computation, because they're already accounted for in the prefix recurrent
# state. Including them in the chunk computation would misalign chunk boundaries.
#
# Solution: Separate conv1d context injection from chunk computation.
# 1. Compute QKV projections for suffix AND overlap prefix tokens
# 2. Apply conv1d with proper causal context (overlap provides prefix context)
# 3. After conv1d: extract only the suffix portion (skip overlap tokens)
# 4. Pass suffix-only (conv1d-processed) QKV to chunk computation with initial_state
conv_overlap = deltanet.conv_kernel_size - 1  # 3

# 3a. Compute prefix QKV for conv1d context
with torch.no_grad():
    prefix_q = deltanet.in_proj_q(prefix_hidden)[0]  # (1, PREFIX_LEN, key_dim_per_tp)
    prefix_k = deltanet.in_proj_k(prefix_hidden)[0]
    prefix_v = deltanet.in_proj_v(prefix_hidden)[0]

# Extract last conv_overlap prefix QKV values for conv1d context
prefix_qkv_tail = torch.cat([
    prefix_q[:, -conv_overlap:, :],   # (1, 3, key_dim_per_tp)
    prefix_k[:, -conv_overlap:, :],   # (1, 3, key_dim_per_tp)
    prefix_v[:, -conv_overlap:, :],   # (1, 3, value_dim_per_tp)
], dim=-1)  # (1, 3, conv_dim_per_tp)

# 3b. Compute suffix QKV
suffix_batch = torch.cat(suffix_hiddens, dim=0)  # (N, SUFFIX_LEN, hidden_size)

with torch.no_grad():
    suffix_q = deltanet.in_proj_q(suffix_batch)[0]  # (N, SUFFIX_LEN, key_dim_per_tp)
    suffix_k = deltanet.in_proj_k(suffix_batch)[0]
    suffix_v = deltanet.in_proj_v(suffix_batch)[0]

# Concatenate suffix QKV for conv1d
suffix_mixed_qkv = torch.cat([suffix_q, suffix_k, suffix_v], dim=-1)  # (N, SUFFIX_LEN, conv_dim_per_tp)

# 3c. Apply conv1d with prefix context
# Prepend prefix QKV tail to suffix QKV for conv1d context
# Each sequence gets: [prefix_qkv_tail, suffix_mixed_qkv] → (1, 3+SUFFIX_LEN, conv_dim_per_tp)
extended_qkv_seqs = []
for i in range(N_SEQUENCES):
    seq_qkv = torch.cat([prefix_qkv_tail, suffix_mixed_qkv[i:i+1]], dim=1)  # (1, 3+64, conv_dim_per_tp)
    extended_qkv_seqs.append(seq_qkv)
extended_qkv = torch.cat(extended_qkv_seqs, dim=0)  # (N, 3+64, conv_dim_per_tp)

# Apply conv1d on extended QKV (with prefix context)
extended_qkv_t = extended_qkv.transpose(1, 2)  # (N, conv_dim_per_tp, 3+64)
conv1d_weight = deltanet.conv1d.weight.to(extended_qkv_t.dtype)
extended_qkv_conv = F.conv1d(
    extended_qkv_t, conv1d_weight, None,
    stride=1, padding=deltanet.conv_kernel_size - 1,
    groups=deltanet.conv_dim_per_tp,
)
total_qkv_len = conv_overlap + SUFFIX_LEN
extended_qkv_conv = extended_qkv_conv[:, :, :total_qkv_len]  # trim causal padding
extended_qkv_conv = F.silu(extended_qkv_conv)
extended_qkv_conv = extended_qkv_conv.transpose(1, 2)  # back to (N, 3+64, conv_dim_per_tp)

# Extract only the suffix portion (skip overlap tokens)
suffix_qkv_conv = extended_qkv_conv[:, conv_overlap:, :]  # (N, 64, conv_dim_per_tp)

# Split back into query, key, value (suffix only, with correct conv1d context)
query_conv, key_conv, value_conv = torch.split(
    suffix_qkv_conv,
    [deltanet.key_dim_per_tp, deltanet.key_dim_per_tp, deltanet.value_dim_per_tp],
    dim=-1,
)

# Reshape to per-head format
query_conv = query_conv.reshape(N_SEQUENCES, SUFFIX_LEN, deltanet.num_k_heads_per_tp, deltanet.head_k_dim)
key_conv = key_conv.reshape(N_SEQUENCES, SUFFIX_LEN, deltanet.num_k_heads_per_tp, deltanet.head_k_dim)
value_conv = value_conv.reshape(N_SEQUENCES, SUFFIX_LEN, deltanet.num_v_heads_per_tp, deltanet.head_v_dim)

# 3d. Compute beta, g, z from suffix hidden_states only
with torch.no_grad():
    b_suffix = deltanet.in_proj_b(suffix_batch)[0]  # (N, SUFFIX_LEN, num_v_heads_per_tp)
    a_suffix = deltanet.in_proj_a(suffix_batch)[0]
    z_suffix = deltanet.in_proj_z(suffix_batch)[0]  # (N, SUFFIX_LEN, value_dim_per_tp)

beta_suffix = b_suffix.sigmoid()  # (N, SUFFIX_LEN, num_v_heads_per_tp)
g_suffix = -deltanet.A_log.float().exp() * F.softplus(a_suffix.float() + deltanet.dt_bias)
g_suffix = g_suffix.to(suffix_batch.dtype)

# GQA expansion
if deltanet.num_v_per_k > 1:
    query_conv = query_conv.repeat_interleave(deltanet.num_v_per_k, dim=2)
    key_conv = key_conv.repeat_interleave(deltanet.num_v_per_k, dim=2)

# Reshape z
z_suffix = z_suffix.reshape(N_SEQUENCES, SUFFIX_LEN, deltanet.num_v_heads_per_tp, deltanet.head_v_dim)

# Transpose to (N, heads, seq, dim) for chunk computation
query_t = query_conv.transpose(1, 2)  # (N, v_heads_per_tp, seq, head_k_dim)
key_t = key_conv.transpose(1, 2)
value_t = value_conv.transpose(1, 2)
beta_t = beta_suffix.transpose(1, 2)
g_t = g_suffix.transpose(1, 2)

# 3e. Run chunk computation with injected prefix state
torch.cuda.synchronize()
t2 = time.time()
with torch.no_grad():
    core_attn_out, _ = torch_chunk_gated_delta_rule(
        query_t, key_t, value_t, g_t, beta_t,
        chunk_size=64,
        initial_state=prefix_state_expanded,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
torch.cuda.synchronize()
t_suffix = time.time() - t2

# 3f. Apply RMSNormGated + out_proj
core_attn_out = deltanet.norm(core_attn_out, z_suffix)
core_attn_out = core_attn_out.reshape(N_SEQUENCES, SUFFIX_LEN, deltanet.value_dim_per_tp)
output_ps = deltanet.out_proj(core_attn_out)[0]  # (N, SUFFIX_LEN, hidden_size)

print(f"[Rank {local_rank}] PS output shape: {output_ps.shape}, time={t_suffix:.3f}s")

# ===== Step 4: Compare outputs =====
print(f"[Rank {local_rank}] === Step 4: Precision alignment ===")

# Normal output: (N, total_len, hidden_size)
# Extract suffix portion from normal output
normal_suffix = output_normal[:, PREFIX_LEN:, :]  # (N, SUFFIX_LEN, hidden_size)
ps_suffix = output_ps  # (N, SUFFIX_LEN, hidden_size)

results = {}
all_cos_sims = []
all_max_diffs = []

for i in range(N_SEQUENCES):
    # Per-position cos_sim
    per_pos_cos = torch.nn.functional.cosine_similarity(
        ps_suffix[i].float(), normal_suffix[i].float(), dim=-1
    )  # (SUFFIX_LEN,)

    cos_sim = per_pos_cos.mean().item()
    max_diff = (ps_suffix[i].float() - normal_suffix[i].float()).abs().max().item()
    mean_diff = (ps_suffix[i].float() - normal_suffix[i].float()).abs().mean().item()

    all_cos_sims.append(cos_sim)
    all_max_diffs.append(max_diff)

    if local_rank == 0:
        print(f"[Rank 0] Seq {i}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}, "
              f"mean_diff={mean_diff:.6f}")
        # Print per-position cos_sim for first 10 and last 5 positions
        first_10 = [f"{v:.4f}" for v in per_pos_cos[:10].tolist()]
        last_5 = [f"{v:.4f}" for v in per_pos_cos[-5:].tolist()]
        print(f"  Per-pos cos_sim: first10={first_10}, last5={last_5}")

    results[i] = {"cos_sim": cos_sim, "max_diff": max_diff, "mean_diff": mean_diff}

overall_cos = sum(all_cos_sims) / len(all_cos_sims)
overall_max = max(all_max_diffs)

if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"DELTANET LAYER PS PRECISION ALIGNMENT")
    print(f"{'='*60}")
    print(f"Layer {LAYER_IDX} (GatedDeltaNet)")
    print(f"N={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"Overall cos_sim: {overall_cos:.6f}")
    print(f"Overall max_diff: {overall_max:.6f}")
    print(f"Normal time: {t_normal:.3f}s")
    print(f"Prefix-only time: {t_prefix:.3f}s")
    print(f"Suffix-only time: {t_suffix:.3f}s")
    print(f"Total PS time: {t_prefix + t_suffix:.3f}s")
    print(f"PS computation savings: {(N_SEQUENCES * total_len) / (PREFIX_LEN + N_SEQUENCES * SUFFIX_LEN):.2f}x")
    if overall_cos >= 0.999:
        print("PASS: DeltaNet state injection precision alignment within bf16 tolerance!")
    else:
        print(f"FAIL: cos_sim {overall_cos:.6f} < 0.999")
    print(f"{'='*60}")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()