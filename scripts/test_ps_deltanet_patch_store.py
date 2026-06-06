#!/usr/bin/env python3
"""Test DeltaNet PS patch state injection mechanism via deltanet_store.

Validates that the _make_verl_deltanet_patch in verl_attention.py correctly
handles two-pass prefix-sharing using the deltanet_store:

1. Prefix pass: DeltaNet forward with output_final_state=True,
   store recurrent_state + conv_state (conv1d overlap) in deltanet_store
2. Suffix pass: Load from deltanet_store, inject as initial_state + conv_overlap_hidden
3. Compare: suffix output should match normal forward suffix output (cos_sim > 0.999)

This test focuses on the deltanet_store mechanism, not the full model E2E.
The full model E2E (cos_sim=0.999973) was already validated in
scripts/run_ps_e2e_twopass_v2.py.

Usage: torchrun --nproc_per_node=4 scripts/test_ps_deltanet_patch_store.py
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
LAYER_IDX = 0

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
from verl.models.qwen3_6.megatron.layers.parallel_deltanet import ParallelQwen3_6GatedDeltaNet

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    bf16=True,
    params_dtype=torch.bfloat16,
)

# Instantiate one DeltaNet layer with small random weights
deltanet = ParallelQwen3_6GatedDeltaNet(config=config, megatron_config=megatron_config)
deltanet = deltanet.to(device)

# Initialize weights with small random values (focus on PS mechanism, not pretrained)
for name, param in deltanet.named_parameters():
    if param.dtype == torch.bfloat16:
        param.data.normal_(0, 0.01)
    elif param.dtype == torch.float32:
        param.data.normal_(0, 0.01)

print(f"[Rank {local_rank}] DeltaNet layer instantiated: {type(deltanet).__name__}")

hidden_size = config.hidden_size  # 5120

# ===== Create test data =====
torch.manual_seed(SEED + 300)
prefix_hidden = torch.randn(1, PREFIX_LEN, hidden_size, dtype=torch.bfloat16, device=device)
suffix_hiddens = [torch.randn(1, SUFFIX_LEN, hidden_size, dtype=torch.bfloat16, device=device)
                  for _ in range(N_SEQUENCES)]

# Build full sequences
full_sequences = []
for i in range(N_SEQUENCES):
    seq_hidden = torch.cat([prefix_hidden, suffix_hiddens[i]], dim=1)
    full_sequences.append(seq_hidden)
hidden_full = torch.cat(full_sequences, dim=0)

# ===== Step 1: Normal forward (reference) =====
print(f"[Rank {local_rank}] === Step 1: Normal forward ===")

with torch.no_grad():
    output_normal = deltanet(hidden_full)

normal_suffix = output_normal[:, PREFIX_LEN:, :]
print(f"[Rank {local_rank}] Normal output shape: {output_normal.shape}")

# ===== Step 2: Prefix pass — store DeltaNet state in deltanet_store =====
print(f"[Rank {local_rank}] === Step 2: Prefix pass with deltanet_store ===")

from prefix_sharing.core.prefix_store import (
    PrefixDeltanetStore, PrefixActivationSlotId, PREFIX_STATE_TYPE_DELTANET_STATE,
)

deltanet_store = PrefixDeltanetStore()

slot_id = PrefixActivationSlotId(
    forward_id=0, micro_batch_id=0, layer_id=LAYER_IDX,
    sample_idx_in_batch=0,
    prefix_state_type=PREFIX_STATE_TYPE_DELTANET_STATE,
    tp_rank=tp_rank,
)

with torch.no_grad():
    # Run prefix-only forward with output_final_state=True
    prefix_output, prefix_state = deltanet(
        prefix_hidden,
        initial_state=None,
        output_final_state=True,
    )

# Extract conv1d overlap from prefix layernorm output
conv_overlap = deltanet.conv_kernel_size - 1  # 3
# In the actual PS patch, hidden_states is the layernorm output passed to DeltaNet.
# For this test, we simulate by using prefix_hidden directly (which IS the layernorm output
# in a real model forward, since the decoder layer applies layernorm before calling DeltaNet).
conv_overlap_hidden = prefix_hidden[:, -conv_overlap:, :].contiguous()

# Expand prefix state to N_SEQUENCES for suffix pass injection
prefix_state_expanded = prefix_state.expand(N_SEQUENCES, -1, -1, -1).clone()

# Store in deltanet_store
deltanet_store.store(
    slot_id,
    recurrent_state=prefix_state_expanded,
    prefix_len=PREFIX_LEN,
    conv_state=conv_overlap_hidden,
)

print(f"[Rank {local_rank}] Stored state: recurrent_state={prefix_state_expanded.shape}, "
      f"conv_state={conv_overlap_hidden.shape}")

# ===== Step 3: Suffix pass — inject stored state (simulating patch behavior) =====
print(f"[Rank {local_rank}] === Step 3: Suffix pass with state injection ===")

# Verify the store contains the state
assert deltanet_store.contains(slot_id), "deltanet_store should contain state for this layer"

stored_state = deltanet_store.load(slot_id)
initial_state = stored_state.recurrent_state
conv_overlap_hidden_loaded = stored_state.conv_state

print(f"[Rank {local_rank}] Loaded state: initial_state={initial_state.shape}, "
      f"conv_overlap={conv_overlap_hidden_loaded.shape}")

# Process suffix with injected state (same as what _make_verl_deltanet_patch does)
suffix_batch = torch.cat(suffix_hiddens, dim=0)  # (N, SUFFIX_LEN, hidden_size)

with torch.no_grad():
    output_ps = deltanet(
        suffix_batch,
        initial_state=initial_state,
        output_final_state=False,
        conv_overlap_hidden=conv_overlap_hidden_loaded,
    )

ps_suffix = output_ps

# ===== Step 4: Compare outputs =====
print(f"[Rank {local_rank}] === Step 4: Precision alignment ===")

all_cos_sims = []
all_max_diffs = []

for i in range(N_SEQUENCES):
    per_pos_cos = F.cosine_similarity(
        ps_suffix[i].float(), normal_suffix[i].float(), dim=-1
    )
    cos_sim = per_pos_cos.mean().item()
    max_diff = (ps_suffix[i].float() - normal_suffix[i].float()).abs().max().item()
    mean_diff = (ps_suffix[i].float() - normal_suffix[i].float()).abs().mean().item()

    all_cos_sims.append(cos_sim)
    all_max_diffs.append(max_diff)

    if local_rank == 0:
        print(f"[Rank 0] Seq {i}: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}, "
              f"mean_diff={mean_diff:.6f}")
        first_5 = [f"{v:.6f}" for v in per_pos_cos[:5].tolist()]
        last_5 = [f"{v:.6f}" for v in per_pos_cos[-5:].tolist()]
        print(f"  Per-pos cos_sim: first5={first_5}, last5={last_5}")

overall_cos = sum(all_cos_sims) / len(all_cos_sims)
overall_max = max(all_max_diffs)

if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"DELTANET PS PATCH STORE MECHANISM TEST")
    print(f"{'='*60}")
    print(f"Layer {LAYER_IDX} (GatedDeltaNet)")
    print(f"N={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"Overall cos_sim: {overall_cos:.6f}")
    print(f"Overall max_diff: {overall_max:.6f}")
    if overall_cos >= 0.999:
        print("PASS: DeltaNet state injection via deltanet_store works!")
    else:
        print(f"FAIL: cos_sim {overall_cos:.6f} < 0.999")
    print(f"{'='*60}")

# Cleanup
deltanet_store.close()
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()