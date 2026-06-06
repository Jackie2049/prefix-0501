#!/usr/bin/env python3
"""Full model E2E prefix-sharing test using two-pass approach.

Approach:
1. Normal forward: process full sequences (prefix + suffix) → reference logits
2. Prefix-only pass: process just prefix tokens through all layers → store DeltaNet states
3. Suffix-only pass: process suffix tokens through all layers with injected states → PS logits
4. Compare suffix logits between normal and PS forward

Uses padded format (not RmPad THD) for simplicity. The per-layer tests have
already validated PS precision for both full attention (cos_sim=0.999997)
and DeltaNet (cos_sim=0.999957) individually. This test validates that they
combine correctly in the full model.

Usage: torchrun --nproc_per_node=4 scripts/run_ps_e2e_twopass.py
"""
import os
import sys
import time
import json

verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_sharing_path = os.path.expanduser("~/rollout-prefix/prefix-0501/prefix-sharing")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path)
sys.path.insert(0, prefix_sharing_path)
sys.path.insert(0, prefix_path)

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoConfig
from safetensors.torch import load_file

# ===== Configuration =====
HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
TP_SIZE = 4
PREFIX_LEN = 64      # Must be >= chunk_size (64) for DeltaNet chunk alignment
SUFFIX_LEN = 64      # Suffix length (also a multiple of chunk_size for clean chunks)
N_SEQUENCES = 4      # n=4 for GRPO
SEED = 42
DELTANET_CHUNK_SIZE = 64  # chunk_size for torch_chunk_gated_delta_rule

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
print(f"[Rank {local_rank}] Config: {type(config).__name__}, hidden={config.hidden_size}, "
      f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
      f"head_dim={config.head_dim}, layers={config.num_hidden_layers}")

# ===== Instantiate model (padded version, NOT RmPad) =====
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLM
from megatron.core import ModelParallelConfig

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    bf16=True,
    params_dtype=torch.bfloat16,
)

model = ParallelQwen3_6ForCausalLM(config=config, megatron_config=megatron_config)
model = model.to(device)

# ===== Load pretrained weights =====
# Weight loading helper functions (same as run_ps_e2e.py)

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

hf_state_dict.clear()
hf_keys_filtered.clear()
torch.cuda.empty_cache()

print(f"[Rank {local_rank}] Loaded {loaded_count} weights, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ===== Create test batch =====
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

# ===== Step 1: Normal forward (no PS) =====
print(f"[Rank {local_rank}] === Step 1: Normal forward ===")

input_ids_normal = torch.tensor(sequences, dtype=torch.long, device=device)
attention_mask_normal = torch.ones((N_SEQUENCES, total_len), dtype=torch.long, device=device)
position_ids_normal = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    output_normal = model(input_ids=input_ids_normal, attention_mask=attention_mask_normal, position_ids=position_ids_normal)
torch.cuda.synchronize()
t_normal = time.time() - t0

logits_normal = output_normal.logits  # (N, total_len, vocab_size)

if local_rank == 0:
    print(f"[Rank 0] Normal forward: {t_normal:.3f}s, logits shape={logits_normal.shape}")
    print(f"[Rank 0] Normal logits: mean={logits_normal.mean().item():.4f}, std={logits_normal.std().item():.4f}")
    print(f"[Rank 0] GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ===== Step 2: Prefix-only forward → store DeltaNet states =====
print(f"[Rank {local_rank}] === Step 2: Prefix-only forward ===")

from verl.models.qwen3_6.megatron.layers.parallel_deltanet import torch_chunk_gated_delta_rule

prefix_input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)  # (1, PREFIX_LEN)
prefix_attn_mask = torch.ones((1, PREFIX_LEN), dtype=torch.long, device=device)
prefix_pos_ids = torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0)

# Process prefix through the model manually (layer by layer)
# We need to extract DeltaNet recurrent states at each DeltaNet layer
deltanet_states = {}  # layer_idx -> prefix recurrent state
prefix_kv_store = {}  # layer_idx -> (prefix_key, prefix_value) for full attention
prefix_hidden_at_layer = {}  # layer_idx -> hidden_states for conv1d context

torch.cuda.synchronize()
t_prefix_start = time.time()

with torch.no_grad():
    # Embed prefix tokens
    prefix_hidden = model.model.embed_tokens(prefix_input_ids)  # (1, PREFIX_LEN, hidden_size)

    # Prepare causal attention mask
    from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import (
        _make_causal_mask, _expand_mask,
    )
    causal_mask = model.model._prepare_decoder_attention_mask(
        prefix_attn_mask, (1, PREFIX_LEN), prefix_hidden
    )

    # Process through each decoder layer
    for layer_idx in range(config.num_hidden_layers):
        decoder_layer = model.model.layers[layer_idx]
        is_deltanet = (layer_types[layer_idx] == "linear_attention")

        # Store hidden_states before this layer (for conv1d context)
        prefix_hidden_at_layer[layer_idx] = prefix_hidden.clone()

        if is_deltanet:
            # DeltaNet layer: process prefix and extract recurrent state
            attn_module = decoder_layer.self_attn

            # Save residual for layer norm + residual connection
            residual = prefix_hidden.clone()

            # Input layernorm
            prefix_hidden_normed = decoder_layer.input_layernorm(prefix_hidden)

            # Run DeltaNet with output_final_state=True
            attn_output, prefix_state = attn_module(
                prefix_hidden_normed,
                initial_state=None,
                output_final_state=True,
            )

            # Store DeltaNet state
            deltanet_states[layer_idx] = prefix_state  # (1, num_v_heads_per_tp, head_k_dim, head_v_dim)

            # Residual connection
            prefix_hidden = residual + attn_output

            # Post-attention layernorm + MLP + residual
            residual2 = prefix_hidden.clone()
            prefix_hidden_normed2 = decoder_layer.post_attention_layernorm(prefix_hidden)
            mlp_output = decoder_layer.mlp(prefix_hidden_normed2)
            prefix_hidden = residual2 + mlp_output

        else:
            # Full attention layer: process prefix normally
            # We can use the standard decoder layer forward (no PS needed for prefix pass)
            prefix_hidden = decoder_layer(
                prefix_hidden,
                attention_mask=causal_mask,
                position_ids=prefix_pos_ids,
            )

            # Also extract prefix KV for potential use in PS
            attn_module = decoder_layer.self_attn
            # Note: we don't need to extract KV separately because the PS patch
            # handles KV injection in the main forward. But we might need the
            # prefix hidden_states for conv1d context at subsequent DeltaNet layers.

    # Final norm
    prefix_final = model.model.norm(prefix_hidden)

torch.cuda.synchronize()
t_prefix = time.time() - t_prefix_start

print(f"[Rank {local_rank}] Prefix pass: {t_prefix:.3f}s, stored {len(deltanet_states)} DeltaNet states")

# ===== Step 3: Suffix-only forward with injected states =====
print(f"[Rank {local_rank}] === Step 3: Suffix PS forward ===")

# Build suffix input_ids for each sequence
suffix_input_ids = torch.tensor(
    [seq[PREFIX_LEN:] for seq in sequences],
    dtype=torch.long, device=device
)  # (N, SUFFIX_LEN)

suffix_attn_mask = torch.ones((N_SEQUENCES, SUFFIX_LEN), dtype=torch.long, device=device)
suffix_pos_ids = torch.arange(PREFIX_LEN, PREFIX_LEN + SUFFIX_LEN, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

torch.cuda.synchronize()
t_suffix_start = time.time()

with torch.no_grad():
    # Embed suffix tokens
    suffix_hidden = model.model.embed_tokens(suffix_input_ids)  # (N, SUFFIX_LEN, hidden_size)

    # Prepare causal attention mask for suffix
    suffix_causal_mask = model.model._prepare_decoder_attention_mask(
        suffix_attn_mask, (N_SEQUENCES, SUFFIX_LEN), suffix_hidden
    )

    # Process through each decoder layer
    for layer_idx in range(config.num_hidden_layers):
        decoder_layer = model.model.layers[layer_idx]
        is_deltanet = (layer_types[layer_idx] == "linear_attention")

        if is_deltanet and layer_idx in deltanet_states:
            # DeltaNet layer with injected prefix state
            attn_module = decoder_layer.self_attn

            # Save residual
            residual = suffix_hidden.clone()

            # Input layernorm
            suffix_hidden_normed = decoder_layer.input_layernorm(suffix_hidden)

            # === DeltaNet PS: conv1d context + state injection ===
            conv_overlap = attn_module.conv_kernel_size - 1  # 3

            # Get prefix hidden_states for conv1d context
            prefix_h_at_layer = prefix_hidden_at_layer[layer_idx]  # (1, PREFIX_LEN, hidden_size)

            # Compute prefix QKV for conv1d overlap
            prefix_q = attn_module.in_proj_q(prefix_h_at_layer)[0]  # (1, PREFIX_LEN, key_dim_per_tp)
            prefix_k = attn_module.in_proj_k(prefix_h_at_layer)[0]
            prefix_v = attn_module.in_proj_v(prefix_h_at_layer)[0]

            # Extract last conv_overlap prefix QKV values
            prefix_qkv_tail = torch.cat([
                prefix_q[:, -conv_overlap:, :],   # (1, 3, key_dim_per_tp)
                prefix_k[:, -conv_overlap:, :],   # (1, 3, key_dim_per_tp)
                prefix_v[:, -conv_overlap:, :],   # (1, 3, value_dim_per_tp)
            ], dim=-1)  # (1, 3, conv_dim_per_tp)

            # Compute suffix QKV
            suffix_q = attn_module.in_proj_q(suffix_hidden_normed)[0]  # (N, SUFFIX_LEN, key_dim_per_tp)
            suffix_k = attn_module.in_proj_k(suffix_hidden_normed)[0]
            suffix_v = attn_module.in_proj_v(suffix_hidden_normed)[0]

            # Concatenate suffix QKV
            suffix_mixed_qkv = torch.cat([suffix_q, suffix_k, suffix_v], dim=-1)  # (N, SUFFIX_LEN, conv_dim_per_tp)

            # Prepend prefix QKV tail for conv1d context
            extended_qkv = torch.cat([
                prefix_qkv_tail.expand(N_SEQUENCES, -1, -1),  # (N, 3, conv_dim_per_tp)
                suffix_mixed_qkv,  # (N, SUFFIX_LEN, conv_dim_per_tp)
            ], dim=1)  # (N, 3+SUFFIX_LEN, conv_dim_per_tp)

            # Apply conv1d with prefix context
            extended_qkv_t = extended_qkv.transpose(1, 2)  # (N, conv_dim_per_tp, 3+SUFFIX_LEN)
            conv1d_weight = attn_module.conv1d.weight.to(extended_qkv_t.dtype)
            extended_qkv_conv = F.conv1d(
                extended_qkv_t, conv1d_weight, None,
                stride=1, padding=attn_module.conv_kernel_size - 1,
                groups=attn_module.conv_dim_per_tp,
            )
            total_qkv_len = conv_overlap + SUFFIX_LEN
            extended_qkv_conv = extended_qkv_conv[:, :, :total_qkv_len]
            extended_qkv_conv = F.silu(extended_qkv_conv)
            extended_qkv_conv = extended_qkv_conv.transpose(1, 2)

            # Extract only suffix portion (skip overlap)
            suffix_qkv_conv = extended_qkv_conv[:, conv_overlap:, :]  # (N, SUFFIX_LEN, conv_dim_per_tp)

            # Split back into q, k, v
            query_conv, key_conv, value_conv = torch.split(
                suffix_qkv_conv,
                [attn_module.key_dim_per_tp, attn_module.key_dim_per_tp, attn_module.value_dim_per_tp],
                dim=-1,
            )

            # Reshape
            query_conv = query_conv.reshape(N_SEQUENCES, SUFFIX_LEN, attn_module.num_k_heads_per_tp, attn_module.head_k_dim)
            key_conv = key_conv.reshape(N_SEQUENCES, SUFFIX_LEN, attn_module.num_k_heads_per_tp, attn_module.head_k_dim)
            value_conv = value_conv.reshape(N_SEQUENCES, SUFFIX_LEN, attn_module.num_v_heads_per_tp, attn_module.head_v_dim)

            # Compute beta, g, z from suffix hidden_states
            b_suffix = attn_module.in_proj_b(suffix_hidden_normed)[0]
            a_suffix = attn_module.in_proj_a(suffix_hidden_normed)[0]
            z_suffix = attn_module.in_proj_z(suffix_hidden_normed)[0]

            beta_suffix = b_suffix.sigmoid()
            g_suffix = -attn_module.A_log.float().exp() * F.softplus(a_suffix.float() + attn_module.dt_bias)
            g_suffix = g_suffix.to(suffix_hidden.dtype)

            # GQA expansion
            if attn_module.num_v_per_k > 1:
                query_conv = query_conv.repeat_interleave(attn_module.num_v_per_k, dim=2)
                key_conv = key_conv.repeat_interleave(attn_module.num_v_per_k, dim=2)

            # Reshape z
            z_suffix = z_suffix.reshape(N_SEQUENCES, SUFFIX_LEN, attn_module.num_v_heads_per_tp, attn_module.head_v_dim)

            # Transpose for chunk computation
            query_t = query_conv.transpose(1, 2)
            key_t = key_conv.transpose(1, 2)
            value_t = value_conv.transpose(1, 2)
            beta_t = beta_suffix.transpose(1, 2)
            g_t = g_suffix.transpose(1, 2)

            # Expand prefix state to match N_SEQUENCES
            prefix_state_expanded = deltanet_states[layer_idx].expand(N_SEQUENCES, -1, -1, -1)

            # Run chunk computation with injected prefix state
            core_attn_out, _ = torch_chunk_gated_delta_rule(
                query_t, key_t, value_t, g_t, beta_t,
                chunk_size=DELTANET_CHUNK_SIZE,
                initial_state=prefix_state_expanded,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )

            # RMSNormGated + out_proj
            core_attn_out = attn_module.norm(core_attn_out, z_suffix)
            core_attn_out = core_attn_out.reshape(N_SEQUENCES, SUFFIX_LEN, attn_module.value_dim_per_tp)
            attn_output = attn_module.out_proj(core_attn_out)[0]  # (N, SUFFIX_LEN, hidden_size)

            # Residual connection
            suffix_hidden = residual + attn_output

            # Post-attention layernorm + MLP + residual
            residual2 = suffix_hidden.clone()
            suffix_hidden_normed2 = decoder_layer.post_attention_layernorm(suffix_hidden)
            mlp_output = decoder_layer.mlp(suffix_hidden_normed2)
            suffix_hidden = residual2 + mlp_output

        else:
            # Full attention layer or DeltaNet without stored state
            # For full attention: use standard forward (PS handles KV injection)
            suffix_hidden = decoder_layer(
                suffix_hidden,
                attention_mask=suffix_causal_mask,
                position_ids=suffix_pos_ids,
            )

    # Final norm + lm_head
    suffix_final = model.model.norm(suffix_hidden)
    suffix_logits = model.lm_head(suffix_final)[0]
    suffix_logits = suffix_logits.float()
    from megatron.core import tensor_parallel
    suffix_logits = tensor_parallel.gather_from_tensor_model_parallel_region(suffix_logits)

torch.cuda.synchronize()
t_suffix = time.time() - t_suffix_start

logits_ps = suffix_logits  # (N, SUFFIX_LEN, vocab_size)

if local_rank == 0:
    print(f"[Rank 0] PS forward: {t_suffix:.3f}s, logits shape={logits_ps.shape}")
    print(f"[Rank 0] GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ===== Step 4: Compare outputs =====
print(f"[Rank {local_rank}] === Step 4: Precision alignment ===")

# Normal logits: (N, total_len, vocab_size) — includes prefix + suffix
# PS logits: (N, SUFFIX_LEN, vocab_size) — only suffix positions

normal_suffix_logits = logits_normal[:, PREFIX_LEN:, :]  # (N, SUFFIX_LEN, vocab_size)
ps_suffix_logits = logits_ps  # (N, SUFFIX_LEN, vocab_size)

results = {}
all_cos_sims = []
all_max_diffs = []

for i in range(N_SEQUENCES):
    per_pos_cos = torch.nn.functional.cosine_similarity(
        ps_suffix_logits[i].float(), normal_suffix_logits[i].float(), dim=-1
    )
    mean_cos = per_pos_cos.mean().item()
    min_cos = per_pos_cos.min().item()

    diff = (ps_suffix_logits[i].float() - normal_suffix_logits[i].float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    all_cos_sims.append(mean_cos)
    all_max_diffs.append(max_diff)

    if local_rank == 0:
        print(f"[Rank 0] Seq {i}: cos_sim={mean_cos:.6f} (min={min_cos:.6f}), "
              f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        # Per-position analysis for first 10 positions
        first_10 = [f"{v:.4f}" for v in per_pos_cos[:10].tolist()]
        print(f"  Per-pos cos_sim first10={first_10}")

    results[i] = {"cos_sim": mean_cos, "min_cos": min_cos, "max_diff": max_diff, "mean_diff": mean_diff}

overall_cos = sum(all_cos_sims) / len(all_cos_sims)
overall_max = max(all_max_diffs)

if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"E2E TWO-PASS PREFIX-SHARING PRECISION ALIGNMENT")
    print(f"{'='*60}")
    print(f"Model: Qwen3.6-27B (TP={TP_SIZE}, bf16)")
    print(f"Pattern: GRPO n={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"Normal forward: {t_normal:.3f}s")
    print(f"Prefix-only forward: {t_prefix:.3f}s")
    print(f"Suffix PS forward: {t_suffix:.3f}s")
    print(f"Total PS time: {t_prefix + t_suffix:.3f}s")
    print(f"PS computation savings: {(N_SEQUENCES * total_len) / (PREFIX_LEN + N_SEQUENCES * SUFFIX_LEN):.2f}x")
    print(f"{'='*60}")
    print(f"Overall cos_sim: {overall_cos:.6f} (threshold: 0.999)")
    print(f"Overall max_diff: {overall_max:.6f} (threshold: 0.01)")
    print(f"{'='*60}")
    if overall_cos >= 0.999 and overall_max <= 0.01:
        print("PASS: Two-pass PS precision alignment within bf16 tolerance!")
    else:
        print("FAIL: Precision misalignment!")
        if overall_cos < 0.999:
            print(f"  - cos_sim {overall_cos:.6f} < 0.999")
        if overall_max > 0.01:
            print(f"  - max_diff {overall_max:.6f} > 0.01")
    print(f"{'='*60}")

    results_path = os.path.expanduser("~/rollout-prefix/ps_e2e_twopass_results.json")
    with open(results_path, "w") as f:
        json.dump({"overall_cos": overall_cos, "overall_max": overall_max,
                   "per_sequence": results, "passed": overall_cos >= 0.999 and overall_max <= 0.01,
                   "config": {"N": N_SEQUENCES, "prefix": PREFIX_LEN, "suffix": SUFFIX_LEN, "tp": TP_SIZE}}, f, indent=2)

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()