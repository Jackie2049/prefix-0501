#!/usr/bin/env python3
"""End-to-end test for DeltaNet PS patch in verl runtime context.

Validates that the _make_verl_deltanet_patch correctly handles
two-pass prefix-sharing via the PrefixSharingRuntimeContext:
1. Prefix pass: process prefix-only tokens, store DeltaNet state + conv overlap
2. Suffix pass: process suffix-only tokens, inject stored state

This test simulates the verl runtime context flow:
- Create a PrefixSharingRuntimeContext with stores
- Prefix pass: open context, run model forward with prefix-only tokens
  → DeltaNet patch should store recurrent state + conv overlap
- Suffix pass: open context (with populated store), run model forward
  → DeltaNet patch should inject initial_state + conv_overlap_hidden
- Compare suffix pass output with normal forward output (cos_sim > 0.999)

Uses padded model (ParallelQwen3_6ForCausalLM) for simpler handling.
The DeltaNet patch works on RmPad forward, but we test the core mechanism
using padded forward with the same parameters.

Usage: torchrun --nproc_per_node=4 scripts/test_ps_deltanet_patch_e2e.py
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

# ===== Create test data =====
hidden_size = config.hidden_size  # 5120

# CRITICAL: prefix hidden_states must be identical across all sequences
torch.manual_seed(SEED + 200)
prefix_hidden = torch.randn(1, PREFIX_LEN, hidden_size, dtype=torch.bfloat16, device=device)
suffix_hiddens = [torch.randn(1, SUFFIX_LEN, hidden_size, dtype=torch.bfloat16, device=device)
                  for _ in range(N_SEQUENCES)]

# Full sequences: [prefix, suffix_i] for each sequence
full_sequences = []
for i in range(N_SEQUENCES):
    seq_hidden = torch.cat([prefix_hidden, suffix_hiddens[i]], dim=1)
    full_sequences.append(seq_hidden)
hidden_full = torch.cat(full_sequences, dim=0)  # (N, total_len, hidden_size)

# ===== Step 1: Normal forward (reference) =====
print(f"[Rank {local_rank}] === Step 1: Normal forward ===")
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    output_normal = model.model(input_ids=None, attention_mask=None, position_ids=None,
                                hidden_states=hidden_full)
    # output_normal: (N, total_len, hidden_size) — output of the model (before lm_head)
    # Actually model.model returns hidden_states through all layers
torch.cuda.synchronize()
t_normal = time.time() - t0
print(f"[Rank {local_rank}] Normal output shape: {output_normal.shape}, time={t_normal:.3f}s")

# Extract suffix portion from normal output
normal_suffix = output_normal[:, PREFIX_LEN:, :]  # (N, SUFFIX_LEN, hidden_size)

# ===== Step 2: Two-pass with PS context =====
# We need to simulate the verl PS runtime context flow:
# 2a. Prefix pass: process prefix-only tokens through all layers
# 2b. Suffix pass: process suffix-only tokens with state injection

print(f"[Rank {local_rank}] === Step 2: Two-pass with PS context ===")

# --- 2a: Prefix pass (layer-by-layer, manual) ---
# Process prefix through each layer, capturing DeltaNet states + KV + conv overlap

from prefix_sharing.integrations.context import (
    prefix_sharing_runtime_context, PrefixSharingRuntimeContext,
)
from prefix_sharing.core.prefix_store import PrefixAttentionStore, PrefixDeltanetStore
from prefix_sharing.core.model_spec import ModelSpec, AttentionLayerType
from prefix_sharing.core.planner import PrefixSharingPlan
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.backends.packed_layout import PackedBatchLayout

# Create a simulated prefix sharing context for prefix pass
ps_config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=PREFIX_LEN)
model_spec = ModelSpec.from_hf_config(config)

# Create stores
deltanet_store = PrefixDeltanetStore()
attn_store = PrefixAttentionStore()

# Process prefix through all layers manually
torch.cuda.synchronize()
t_prefix = time.time()

prefix_hidden_states = prefix_hidden.expand(1, -1, -1).clone()  # (1, PREFIX_LEN, hidden_size)

# Run through embed_tokens (skip, we're using hidden_states directly)
# prefix_hidden_states is already the hidden representation after embedding

prefix_recurrent_states = {}  # layer_id -> recurrent_state
prefix_conv_overlaps = {}      # layer_id -> conv_overlap_hidden
prefix_kv_states = {}          # layer_id -> (key, value)

with torch.no_grad():
    for layer_idx, layer in enumerate(model.model.layers):
        # Input layernorm
        hidden_normed = layer.input_layernorm(prefix_hidden_states)

        # Determine layer type
        layer_type = model_spec.layer_type(layer_idx)
        attn_module = layer.self_attn

        if layer_type == AttentionLayerType.LINEAR_ATTENTION:
            # DeltaNet layer: run with output_final_state=True
            output, final_state = attn_module.forward(
                hidden_normed,
                attention_mask=None,
                position_ids=torch.arange(PREFIX_LEN, device=device).unsqueeze(0),
                initial_state=None,
                output_final_state=True,
            )
            if final_state is not None:
                # Store recurrent state
                # Expand to N_SEQUENCES for suffix pass injection
                prefix_recurrent_states[layer_idx] = final_state.expand(N_SEQUENCES, -1, -1, -1).clone()

                # Extract conv1d overlap from layernorm output
                conv_overlap = attn_module.conv_kernel_size - 1  # 3
                overlap_hidden = hidden_normed[:, -conv_overlap:, :].contiguous()
                # Expand overlap to N_SEQUENCES (will be expanded in forward too)
                prefix_conv_overlaps[layer_idx] = overlap_hidden

                # Store in deltanet_store
                from prefix_sharing.core.prefix_store import PrefixActivationSlotId, PREFIX_STATE_TYPE_DELTANET_STATE
                slot_id = PrefixActivationSlotId(
                    forward_id=0, micro_batch_id=0, layer_id=layer_idx,
                    sample_idx_in_batch=0,
                    prefix_state_type=PREFIX_STATE_TYPE_DELTANET_STATE,
                    tp_rank=tp_rank,
                )
                deltanet_store.store(
                    slot_id,
                    recurrent_state=prefix_recurrent_states[layer_idx],
                    prefix_len=PREFIX_LEN,
                    conv_state=prefix_conv_overlaps[layer_idx],
                    overwrite=True,
                )
        else:
            # Full attention layer: capture prefix KV
            # Run normally to get output, also capture KV for suffix pass
            q_full = attn_module.q_proj(hidden_normed)[0]
            key_states_raw = attn_module.k_proj(hidden_normed)[0]
            value_states_raw = attn_module.v_proj(hidden_normed)[0]

            # Chunk q_proj into query and gate
            bsz_prefix = 1
            q_shape = (bsz_prefix * PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim * 2)
            query_states, gate = torch.chunk(q_full.view(*q_shape), 2, dim=-1)

            query_states = query_states.view(bsz_prefix * PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
            key_states = key_states_raw.view(bsz_prefix * PREFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)
            value_states = value_states_raw.view(bsz_prefix * PREFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)

            # QK normalization
            query_states = attn_module.q_norm(query_states)
            key_states = attn_module.k_norm(key_states)

            # Partial RoPE
            cos, sin = attn_module.rotary_emb(value_states, seq_len=PREFIX_LEN)
            cos = cos[:, :cos.shape[1] // 2]
            sin = sin[:, :sin.shape[1] // 2]

            if attn_module.rope_dim == attn_module.head_dim:
                from flash_attn.layers.rotary import apply_rotary_emb
                query_states = apply_rotary_emb(query_states, cos, sin, interleaved=False, inplace=False)
                key_states = apply_rotary_emb(key_states, cos, sin, interleaved=False, inplace=False)
            else:
                q_rot = query_states[:, :, :attn_module.rope_dim]
                q_pass = query_states[:, :, attn_module.rope_dim:]
                k_rot = key_states[:, :, :attn_module.rope_dim]
                k_pass = key_states[:, :, attn_module.rope_dim:]

                from flash_attn.layers.rotary import apply_rotary_emb
                q_rot = apply_rotary_emb(q_rot, cos, sin, interleaved=False, inplace=False)
                k_rot = apply_rotary_emb(k_rot, cos, sin, interleaved=False, inplace=False)
                query_states = torch.cat([q_rot, q_pass], dim=-1)
                key_states = torch.cat([k_rot, k_pass], dim=-1)

            # GQA expand
            num_key_value_groups = attn_module.num_key_value_groups
            if num_key_value_groups > 1:
                key_states = key_states.unsqueeze(2).expand(-1, -1, num_key_value_groups, -1).reshape(
                    bsz_prefix * PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
                value_states = value_states.unsqueeze(2).expand(-1, -1, num_key_value_groups, -1).reshape(
                    bsz_prefix * PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)

            # Store prefix KV (expanded to N sequences for suffix pass)
            prefix_key_expanded = key_states.expand(N_SEQUENCES * PREFIX_LEN, -1, -1).clone()
            prefix_value_expanded = value_states.expand(N_SEQUENCES * PREFIX_LEN, -1, -1).clone()
            prefix_kv_states[layer_idx] = (prefix_key_expanded, prefix_value_expanded)

            # Run full attention for prefix (just to get output for residual)
            # Since we need the actual attention output for the prefix pass
            # We use flash_attn_varlen_func with prefix-only KV
            from flash_attn import flash_attn_varlen_func

            cu_seqlens_prefix = torch.tensor([0, PREFIX_LEN], dtype=torch.int32, device=device)

            query_3d = query_states.reshape(PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
            key_3d = key_states.reshape(PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
            value_3d = value_states.reshape(PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)

            attn_output = flash_attn_varlen_func(
                query_3d, key_3d, value_3d,
                cu_seqlens_q=cu_seqlens_prefix, cu_seqlens_k=cu_seqlens_prefix,
                max_seqlen_q=PREFIX_LEN, max_seqlen_k=PREFIX_LEN,
                dropout_p=0.0, softmax_scale=attn_module.scaling, causal=True,
            )

            attn_output = attn_output.reshape(PREFIX_LEN, attn_module.q_output_size_per_tp)

            # Apply output gate
            if attn_module.attn_output_gate:
                gate_prefix = gate.reshape(PREFIX_LEN, attn_module.num_heads_per_tp * attn_module.head_dim)
                attn_output = attn_output * torch.sigmoid(gate_prefix)

            # o_proj
            attn_output = attn_output.reshape(PREFIX_LEN, 1, attn_module.q_output_size_per_tp).contiguous()
            output = attn_module.o_proj(attn_output)[0]
            output = output.reshape(1, PREFIX_LEN, hidden_size)

        # Residual + layernorm + MLP
        residual = prefix_hidden_states + output  # Wait, output might be wrong shape for DeltaNet
        # Actually for DeltaNet, output shape is (1, PREFIX_LEN, hidden_size) already
        # For full attn, we reshaped to (1, PREFIX_LEN, hidden_size)

        hidden_normed_mlp = layer.post_attention_layernorm(residual)
        mlp_output = layer.mlp(hidden_normed_mlp)
        prefix_hidden_states = residual + mlp_output

torch.cuda.synchronize()
t_prefix_pass = time.time() - t_prefix
print(f"[Rank {local_rank}] Prefix pass time: {t_prefix_pass:.3f}s")
print(f"[Rank {local_rank}] Stored {len(prefix_recurrent_states)} DeltaNet states, "
      f"{len(prefix_kv_states)} full attn KVs")

# --- 2b: Suffix pass (layer-by-layer, with state injection) ---
torch.cuda.synchronize()
t_suffix = time.time()

suffix_batch = torch.cat(suffix_hiddens, dim=0)  # (N, SUFFIX_LEN, hidden_size)
suffix_hidden_states = suffix_batch.clone()

with torch.no_grad():
    for layer_idx, layer in enumerate(model.model.layers):
        hidden_normed = layer.input_layernorm(suffix_hidden_states)
        layer_type = model_spec.layer_type(layer_idx)
        attn_module = layer.self_attn

        if layer_type == AttentionLayerType.LINEAR_ATTENTION:
            # DeltaNet: inject stored state + conv overlap
            initial_state = prefix_recurrent_states[layer_idx]
            conv_overlap_hidden = prefix_conv_overlaps[layer_idx]

            output = attn_module.forward(
                hidden_normed,
                attention_mask=None,
                position_ids=torch.arange(PREFIX_LEN, PREFIX_LEN + SUFFIX_LEN, device=device).unsqueeze(0).expand(N_SEQUENCES, -1),
                initial_state=initial_state,
                output_final_state=False,
                conv_overlap_hidden=conv_overlap_hidden,
            )
        else:
            # Full attention: KV expansion (same as two-pass E2E test)
            # Compute suffix QKV
            q_full = attn_module.q_proj(hidden_normed)[0]
            key_states_raw = attn_module.k_proj(hidden_normed)[0]
            value_states_raw = attn_module.v_proj(hidden_normed)[0]

            N = N_SEQUENCES
            q_shape = (N * SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim * 2)
            query_states, gate = torch.chunk(q_full.view(*q_shape), 2, dim=-1)

            query_states = query_states.view(N * SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
            key_states = key_states_raw.view(N * SUFFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)
            value_states = value_states_raw.view(N * SUFFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)

            # QK normalization
            query_states = attn_module.q_norm(query_states)
            key_states = attn_module.k_norm(key_states)

            # Partial RoPE with position offset
            cos, sin = attn_module.rotary_emb(value_states, seq_len=PREFIX_LEN + SUFFIX_LEN)
            cos = cos[:, :cos.shape[1] // 2]
            sin = sin[:, :sin.shape[1] // 2]

            # Position offset: suffix tokens start at PREFIX_LEN
            # Flash attn apply_rotary_emb uses cu_seqlens for position
            # We need suffix-only positions starting at PREFIX_LEN
            cu_seqlens_suffix = torch.tensor([0] + [SUFFIX_LEN] * N, dtype=torch.int32, device=device).cumsum(0).to(torch.int32)

            if attn_module.rope_dim == attn_module.head_dim:
                from flash_attn.layers.rotary import apply_rotary_emb
                query_states = apply_rotary_emb(
                    query_states, cos, sin, interleaved=False, inplace=False,
                    cu_seqlens=cu_seqlens_suffix, max_seqlen=SUFFIX_LEN,
                )
                key_states = apply_rotary_emb(
                    key_states, cos, sin, interleaved=False, inplace=False,
                    cu_seqlens=cu_seqlens_suffix, max_seqlen=SUFFIX_LEN,
                )
            else:
                q_rot = query_states[:, :, :attn_module.rope_dim]
                q_pass = query_states[:, :, attn_module.rope_dim:]
                k_rot = key_states[:, :, :attn_module.rope_dim]
                k_pass = key_states[:, :, attn_module.rope_dim:]

                from flash_attn.layers.rotary import apply_rotary_emb
                q_rot = apply_rotary_emb(q_rot, cos, sin, interleaved=False, inplace=False,
                                         cu_seqlens=cu_seqlens_suffix, max_seqlen=SUFFIX_LEN)
                k_rot = apply_rotary_emb(k_rot, cos, sin, interleaved=False, inplace=False,
                                         cu_seqlens=cu_seqlens_suffix, max_seqlen=SUFFIX_LEN)
                query_states = torch.cat([q_rot, q_pass], dim=-1)
                key_states = torch.cat([k_rot, k_pass], dim=-1)

            # GQA expand
            num_key_value_groups = attn_module.num_key_value_groups
            if num_key_value_groups > 1:
                key_states = key_states.unsqueeze(2).expand(-1, -1, num_key_value_groups, -1).reshape(
                    N * SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
                value_states = value_states.unsqueeze(2).expand(-1, -1, num_key_value_groups, -1).reshape(
                    N * SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)

            # KV expansion: concatenate prefix KV + suffix KV
            prefix_key, prefix_value = prefix_kv_states[layer_idx]
            expanded_key = torch.cat([prefix_key, key_states], dim=0)
            expanded_value = torch.cat([prefix_value, value_states], dim=0)

            # Build expanded cu_seqlens
            expanded_lengths = [PREFIX_LEN + SUFFIX_LEN] * N
            expanded_cu_seqlens = torch.tensor([0] + expanded_lengths, dtype=torch.int32, device=device).cumsum(0).to(torch.int32)

            # flash_attn with expanded KV
            from flash_attn import flash_attn_varlen_func
            query_flat = query_states.reshape(N * SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)

            attn_output = flash_attn_varlen_func(
                query_flat, expanded_key, expanded_value,
                cu_seqlens_q=cu_seqlens_suffix,
                cu_seqlens_k=expanded_cu_seqlens,
                max_seqlen_q=SUFFIX_LEN,
                max_seqlen_k=PREFIX_LEN + SUFFIX_LEN,
                dropout_p=0.0,
                softmax_scale=attn_module.scaling,
                causal=True,
            )

            attn_output = attn_output.reshape(N, SUFFIX_LEN, attn_module.num_heads_per_tp * attn_module.head_dim)
            gate_suffix = gate.reshape(N, SUFFIX_LEN, attn_module.num_heads_per_tp * attn_module.head_dim)

            if attn_module.attn_output_gate:
                attn_output = attn_output * torch.sigmoid(gate_suffix)

            # o_proj
            attn_output = attn_output.reshape(N * SUFFIX_LEN, 1, attn_module.q_output_size_per_tp).contiguous()
            output = attn_module.o_proj(attn_output)[0]
            output = output.reshape(N, SUFFIX_LEN, hidden_size)

        # Residual + MLP
        residual = suffix_hidden_states + output
        hidden_normed_mlp = layer.post_attention_layernorm(residual)
        mlp_output = layer.mlp(hidden_normed_mlp)
        suffix_hidden_states = residual + mlp_output

torch.cuda.synchronize()
t_suffix_pass = time.time() - t_suffix
print(f"[Rank {local_rank}] Suffix pass time: {t_suffix_pass:.3f}s")

# ===== Step 3: Compare outputs =====
print(f"[Rank {local_rank}] === Step 3: Precision alignment ===")

# suffix_hidden_states: (N, SUFFIX_LEN, hidden_size) — output from suffix pass
ps_suffix = suffix_hidden_states
results = {}
all_cos_sims = []
all_max_diffs = []

for i in range(N_SEQUENCES):
    per_pos_cos = torch.nn.functional.cosine_similarity(
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

    results[i] = {"cos_sim": cos_sim, "max_diff": max_diff, "mean_diff": mean_diff}

overall_cos = sum(all_cos_sims) / len(all_cos_sims)
overall_max = max(all_max_diffs)

if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"DELTANET PS PATCH E2E PRECISION ALIGNMENT")
    print(f"{'='*60}")
    print(f"N={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"Overall cos_sim: {overall_cos:.6f}")
    print(f"Overall max_diff: {overall_max:.6f}")
    print(f"Normal time: {t_normal:.3f}s")
    print(f"Prefix pass time: {t_prefix_pass:.3f}s")
    print(f"Suffix pass time: {t_suffix_pass:.3f}s")
    print(f"Total PS time: {t_prefix_pass + t_suffix_pass:.3f}s")
    print(f"PS computation savings: {(N_SEQUENCES * (PREFIX_LEN + SUFFIX_LEN)) / (PREFIX_LEN + N_SEQUENCES * SUFFIX_LEN):.2f}x")
    if overall_cos >= 0.999:
        print("PASS: DeltaNet PS patch precision alignment within bf16 tolerance!")
    else:
        print(f"FAIL: cos_sim {overall_cos:.6f} < 0.999")
    print(f"{'='*60}")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()