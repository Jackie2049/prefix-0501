#!/usr/bin/env python3
"""Two-pass E2E prefix-sharing precision alignment test for Qwen3.6-27B.

Implements the two-pass approach validated at per-layer level:
1. Prefix pass: Process prefix tokens through all layers, capturing
   DeltaNet recurrent states + conv1d overlap context + full attn prefix KV
2. Suffix pass: Process suffix tokens through all layers, injecting
   DeltaNet states + conv1d overlap + full attn prefix KV expansion
3. Compare suffix logits vs normal (full sequence) forward

Key insight: The residual stream at suffix positions is preserved because:
- Embedding at suffix positions is identical (same tokens)
- Each layer's output at suffix positions is correct (DeltaNet with state
  injection, full attention with KV expansion)
- Therefore the residual accumulates correctly through all layers

DeltaNet PS requires THREE injections:
1. Recurrent state injection (initial_state for torch_chunk_gated_delta_rule)
2. Conv1d context injection (last 3 prefix tokens' hidden_states for overlap)
3. Chunk boundary alignment (PREFIX_LEN must be >= chunk_size=64)

Usage: torchrun --nproc_per_node=4 scripts/run_ps_e2e_twopass_v2.py
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
import torch.distributed as dist
from transformers import AutoConfig
from safetensors.torch import load_file

# ===== Configuration =====
HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
TP_SIZE = 4
PREFIX_LEN = 64      # Must be >= chunk_size (64) for proper chunk boundary alignment
SUFFIX_LEN = 64      # Suffix length
N_SEQUENCES = 4      # n=4 for GRPO
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
layer_types = config.layer_types
num_layers = config.num_hidden_layers
hidden_size = config.hidden_size

if local_rank == 0:
    print(f"Config: {type(config).__name__}, hidden={hidden_size}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"head_dim={config.head_dim}, layers={num_layers}")
    print(f"Layer types: {sum(1 for t in layer_types if t=='full_attention')} full attn, "
          f"{sum(1 for t in layer_types if t=='linear_attention')} linear attn")

# ===== Instantiate model (padded format for layer-by-layer processing) =====
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
# (Same weight loading logic as run_ps_e2e.py)
print(f"[Rank {local_rank}] Loading pretrained weights...")
hf_state_dict = {}
for i in range(1, 12):
    shard_path = os.path.join(HF_MODEL_PATH, f"model.safetensors-{i:05d}-of-00011.safetensors")
    shard_dict = load_file(shard_path)
    hf_state_dict.update(shard_dict)

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
    return (shard_tensor(q_portion, 0, tp_size, tp_rank),
            shard_tensor(k_portion, 0, tp_size, tp_rank),
            shard_tensor(v_portion, 0, tp_size, tp_rank))

def shard_conv1d_weight(conv1d_weight, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    q_portion = conv1d_weight[:key_dim]
    k_portion = conv1d_weight[key_dim:key_dim*2]
    v_portion = conv1d_weight[key_dim*2:]
    return torch.cat([
        shard_tensor(q_portion, 0, tp_size, tp_rank),
        shard_tensor(k_portion, 0, tp_size, tp_rank),
        shard_tensor(v_portion, 0, tp_size, tp_rank),
    ], dim=0).contiguous()

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

loaded_count = 0
for layer_idx in range(num_layers):
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
total_len = PREFIX_LEN + SUFFIX_LEN
torch.manual_seed(SEED)
prefix_ids = torch.randint(0, vocab_size, (PREFIX_LEN,), device="cpu").tolist()

sequences = []
for i in range(N_SEQUENCES):
    suffix_ids = torch.randint(0, vocab_size, (SUFFIX_LEN,), device="cpu").tolist()
    sequences.append(prefix_ids + suffix_ids)

print(f"[Rank {local_rank}] Test: N={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}, total={total_len}")

# ===== Helper: determine layer type =====
def is_deltanet_layer(layer_idx):
    return layer_types[layer_idx] == "linear_attention"

conv_overlap_size = 3  # conv_kernel_size - 1

# ====================================================================
# Step 1: Normal forward (no PS) — reference logits
# ====================================================================
print(f"[Rank {local_rank}] === Step 1: Normal forward (reference) ===")

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

# ====================================================================
# Step 2: Prefix-only forward — capture DeltaNet states + overlap context + prefix KV
# ====================================================================
print(f"[Rank {local_rank}] === Step 2: Prefix-only forward ===")

# Embed prefix tokens (1 sequence)
prefix_input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)  # (1, PREFIX_LEN)
prefix_embeds = model.model.embed_tokens(prefix_input_ids)  # (1, PREFIX_LEN, hidden_size)

# Process through layers manually
hidden_prefix = prefix_embeds
deltanet_states = {}       # layer_idx → recurrent_state (1, v_heads_per_tp, k_dim, v_dim)
deltanet_overlaps = {}     # layer_idx → overlap hidden_states (1, overlap, hidden_size)
prefix_kv_store = {}       # layer_idx → (key, value) for full attention layers

torch.cuda.synchronize()
t1 = time.time()

with torch.no_grad():
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn_module = layer.self_attn

        # Layernorm
        residual = hidden_prefix
        hidden_normed = layer.input_layernorm(hidden_prefix)

        if is_deltanet_layer(layer_idx):
            # Store conv1d overlap context (last 3 tokens of layernorm output)
            deltanet_overlaps[layer_idx] = hidden_normed[:, -conv_overlap_size:, :].clone()

            # Call DeltaNet forward with output_final_state=True
            attn_output, recurrent_state = attn_module.forward(
                hidden_normed,
                attention_mask=None,
                position_ids=torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0),
                initial_state=None,
                output_final_state=True,
            )
            deltanet_states[layer_idx] = recurrent_state
        else:
            # Full attention: compute and store prefix KV for later expansion
            # Compute Q, K, V projections
            q_full = attn_module.q_proj(hidden_normed)[0]
            k_states = attn_module.k_proj(hidden_normed)[0]
            v_states = attn_module.v_proj(hidden_normed)[0]

            # Chunk q_proj into query + gate
            q_shape = (PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim * 2)
            query_states, gate = torch.chunk(q_full.view(*q_shape), 2, dim=-1)
            gate = gate.reshape(PREFIX_LEN, attn_module.num_heads_per_tp * attn_module.head_dim)

            # Reshape: flash_attn apply_rotary_emb expects (batch, seq, heads, head_dim)
            query_states = query_states.view(1, PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
            k_states = k_states.view(1, PREFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)
            v_states = v_states.view(1, PREFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)

            # QK normalization (per-head RMSNorm BEFORE RoPE)
            query_states = attn_module.q_norm(query_states)
            key_states = attn_module.k_norm(k_states)

            # Partial RoPE AFTER q_norm/k_norm
            cos, sin = attn_module.rotary_emb(v_states, seq_len=PREFIX_LEN)
            cos, sin = cos[:, :cos.shape[1] // 2], sin[:, :sin.shape[1] // 2]

            from flash_attn.layers.rotary import apply_rotary_emb
            if attn_module.rope_dim == attn_module.head_dim:
                query_states = apply_rotary_emb(query_states, cos, sin, interleaved=False, inplace=False)
                key_states = apply_rotary_emb(key_states, cos, sin, interleaved=False, inplace=False)
            else:
                # Partial RoPE: only rotate first rope_dim dims
                q_rot = query_states[:, :, :, :attn_module.rope_dim]
                q_pass = query_states[:, :, :, attn_module.rope_dim:]
                k_rot = key_states[:, :, :, :attn_module.rope_dim]
                k_pass = key_states[:, :, :, attn_module.rope_dim:]
                q_rot = apply_rotary_emb(q_rot, cos, sin, interleaved=False, inplace=False)
                k_rot = apply_rotary_emb(k_rot, cos, sin, interleaved=False, inplace=False)
                query_states = torch.cat([q_rot, q_pass], dim=-1)
                key_states = torch.cat([k_rot, k_pass], dim=-1)

            # GQA: repeat KV heads to match query heads
            num_key_value_groups = attn_module.num_key_value_groups
            if num_key_value_groups > 1:
                key_states = key_states.unsqueeze(3).expand(
                    -1, -1, -1, num_key_value_groups, -1
                ).reshape(1, PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
                v_states = v_states.unsqueeze(3).expand(
                    -1, -1, -1, num_key_value_groups, -1
                ).reshape(1, PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)

            # Store prefix KV (in per-head format for flash_attn)
            prefix_kv_store[layer_idx] = (key_states, v_states)
            # key: (1, PREFIX_LEN, num_heads_per_tp, head_dim)
            # value: (1, PREFIX_LEN, num_heads_per_tp, head_dim)

            # Run full attention on prefix tokens normally
            # Build causal mask for 1 sequence of PREFIX_LEN
            causal_mask = torch.zeros((1, 1, PREFIX_LEN, PREFIX_LEN), dtype=hidden_normed.dtype, device=device)
            causal_mask = causal_mask.masked_fill(
                torch.triu(torch.ones(PREFIX_LEN, PREFIX_LEN, dtype=torch.bool, device=device), diagonal=1),
                torch.finfo(hidden_normed.dtype).min,
            )
            attn_output = attn_module.forward(
                hidden_normed,
                attention_mask=causal_mask,
                position_ids=torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0),
            )

        hidden_prefix = residual + attn_output

        # MLP
        residual = hidden_prefix
        hidden_normed = layer.post_attention_layernorm(hidden_prefix)
        mlp_output = layer.mlp(hidden_normed)
        hidden_prefix = residual + mlp_output

    # Final layernorm (not needed for state capture, but for completeness)
    hidden_prefix_final = model.model.norm(hidden_prefix)

torch.cuda.synchronize()
t_prefix = time.time() - t1

if local_rank == 0:
    print(f"[Rank 0] Prefix pass: {t_prefix:.3f}s")
    print(f"[Rank 0] Captured {len(deltanet_states)} DeltaNet states, "
          f"{len(deltanet_overlaps)} overlap contexts, {len(prefix_kv_store)} prefix KV")
    for lid in sorted(deltanet_states.keys())[:3]:
        print(f"  Layer {lid}: state shape={deltanet_states[lid].shape}, "
              f"overlap shape={deltanet_overlaps[lid].shape}")

# ====================================================================
# Step 3: Suffix-only forward with state injection (PS forward)
# ====================================================================
print(f"[Rank {local_rank}] === Step 3: Suffix-only forward with state injection ===")

# Build suffix sequences: N sequences, each SUFFIX_LEN tokens
suffix_ids_list = [seq[PREFIX_LEN:] for seq in sequences]
suffix_input_ids = torch.tensor(suffix_ids_list, dtype=torch.long, device=device)  # (N, SUFFIX_LEN)

# Embed suffix tokens
suffix_embeds = model.model.embed_tokens(suffix_input_ids)  # (N, SUFFIX_LEN, hidden_size)

# Process through layers with state injection
hidden_suffix = suffix_embeds

torch.cuda.synchronize()
t2 = time.time()

with torch.no_grad():
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn_module = layer.self_attn

        # Layernorm
        residual = hidden_suffix
        hidden_normed = layer.input_layernorm(hidden_suffix)

        if is_deltanet_layer(layer_idx):
            # Inject DeltaNet prefix state + conv1d overlap
            prefix_state = deltanet_states[layer_idx]  # (1, v_heads_per_tp, k_dim, v_dim)
            prefix_state_expanded = prefix_state.expand(N_SEQUENCES, -1, -1, -1).contiguous()

            overlap_hidden = deltanet_overlaps[layer_idx]  # (1, overlap, hidden_size)
            overlap_expanded = overlap_hidden.expand(N_SEQUENCES, -1, -1).contiguous()

            attn_output = attn_module.forward(
                hidden_normed,
                attention_mask=None,
                position_ids=torch.arange(SUFFIX_LEN, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1),
                initial_state=prefix_state_expanded,
                output_final_state=False,
                conv_overlap_hidden=overlap_expanded,
            )
        else:
            # Full attention with prefix KV expansion
            prefix_key, prefix_value = prefix_kv_store[layer_idx]
            # prefix_key: (1, PREFIX_LEN, num_heads_per_tp, head_dim)
            # prefix_value: (1, PREFIX_LEN, num_heads_per_tp, head_dim)

            # Compute Q, K, V for suffix tokens
            q_full = attn_module.q_proj(hidden_normed)[0]
            k_states = attn_module.k_proj(hidden_normed)[0]
            v_states = attn_module.v_proj(hidden_normed)[0]

            # Chunk q_proj into query + gate
            q_shape = (N_SEQUENCES, SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim * 2)
            query_states, gate = torch.chunk(q_full.view(*q_shape), 2, dim=-1)
            gate = gate.reshape(N_SEQUENCES, SUFFIX_LEN, attn_module.num_heads_per_tp * attn_module.head_dim)

            # Reshape
            query_states = query_states.view(N_SEQUENCES, SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
            k_states = k_states.view(N_SEQUENCES, SUFFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)
            v_states = v_states.view(N_SEQUENCES, SUFFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)

            # QK normalization
            query_states = attn_module.q_norm(query_states)
            key_states = attn_module.k_norm(k_states)

            # Partial RoPE AFTER q_norm/k_norm
            # Position IDs for suffix start at PREFIX_LEN (offset from prefix)
            cos_suffix, sin_suffix = attn_module.rotary_emb(v_states, seq_len=PREFIX_LEN + SUFFIX_LEN)
            # Slice cos/sin for suffix positions only (positions PREFIX_LEN to total_len)
            # rotary_emb returns (seq_len, 1, rope_dim) format
            # We need positions PREFIX_LEN..PREFIX_LEN+SUFFIX_LEN-1
            cos_suffix = cos_suffix[PREFIX_LEN:PREFIX_LEN + SUFFIX_LEN]
            sin_suffix = sin_suffix[PREFIX_LEN:PREFIX_LEN + SUFFIX_LEN]
            cos_suffix = cos_suffix[:, :cos_suffix.shape[1] // 2]
            sin_suffix = sin_suffix[:, :sin_suffix.shape[1] // 2]

            # For prefix KV, we need the cos/sin at prefix positions (already applied in Step 2)
            # The prefix KV was computed with RoPE at positions 0..PREFIX_LEN-1
            # We don't need to re-apply RoPE to prefix KV

            if attn_module.rope_dim == attn_module.head_dim:
                from flash_attn.layers.rotary import apply_rotary_emb
                query_states = apply_rotary_emb(query_states, cos_suffix, sin_suffix, interleaved=False, inplace=False)
                key_states = apply_rotary_emb(key_states, cos_suffix, sin_suffix, interleaved=False, inplace=False)
            else:
                q_rot = query_states[:, :, :, :attn_module.rope_dim]
                q_pass = query_states[:, :, :, attn_module.rope_dim:]
                k_rot = key_states[:, :, :, :attn_module.rope_dim]
                k_pass = key_states[:, :, :, attn_module.rope_dim:]
                from flash_attn.layers.rotary import apply_rotary_emb
                q_rot = apply_rotary_emb(q_rot, cos_suffix, sin_suffix, interleaved=False, inplace=False)
                k_rot = apply_rotary_emb(k_rot, cos_suffix, sin_suffix, interleaved=False, inplace=False)
                query_states = torch.cat([q_rot, q_pass], dim=-1)
                key_states = torch.cat([k_rot, k_pass], dim=-1)

            # GQA: repeat KV heads to match query heads
            num_key_value_groups = attn_module.num_key_value_groups
            if num_key_value_groups > 1:
                # key_states: (N, SUFFIX_LEN, kv_heads_per_tp, head_dim)
                # Expand kv_heads to query heads via repeat_interleave along dim=2
                key_states = key_states.repeat_interleave(num_key_value_groups, dim=2)
                v_states = v_states.repeat_interleave(num_key_value_groups, dim=2)

            # Expand prefix KV to all N sequences
            expanded_prefix_key = prefix_key.expand(N_SEQUENCES, -1, -1, -1).contiguous()
            expanded_prefix_value = prefix_value.expand(N_SEQUENCES, -1, -1, -1).contiguous()

            # Concatenate prefix KV + suffix KV → expanded KV
            expanded_key = torch.cat([expanded_prefix_key, key_states], dim=1)   # (N, PREFIX_LEN+SUFFIX_LEN, heads, head_dim)
            expanded_value = torch.cat([expanded_prefix_value, v_states], dim=1)

            # Build cu_seqlens for flash_attn
            # Q: suffix only (SUFFIX_LEN per sequence)
            # K/V: prefix + suffix (PREFIX_LEN + SUFFIX_LEN per sequence)
            cu_seqlens_q = torch.tensor(
                [0] + [SUFFIX_LEN] * N_SEQUENCES, device=device, dtype=torch.int32
            ).cumsum(0).to(torch.int32)
            cu_seqlens_kv = torch.tensor(
                [0] + [PREFIX_LEN + SUFFIX_LEN] * N_SEQUENCES, device=device, dtype=torch.int32
            ).cumsum(0).to(torch.int32)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                query_states = query_states.to(torch.float16)
                expanded_key = expanded_key.to(torch.float16)
                expanded_value = expanded_value.to(torch.float16)

            from flash_attn import flash_attn_varlen_func
            attn_output = flash_attn_varlen_func(
                query_states,
                expanded_key,
                expanded_value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=SUFFIX_LEN,
                max_seqlen_k=PREFIX_LEN + SUFFIX_LEN,
                dropout_p=0.0,
                softmax_scale=attn_module.scaling,
                causal=True,  # Causal within suffix, but prefix is bidirectional
            )

            attn_output = attn_output.to(input_dtype)
            attn_output = attn_output.reshape(N_SEQUENCES, SUFFIX_LEN, attn_module.q_output_size_per_tp)

            # Apply output gate BEFORE o_proj
            if attn_module.attn_output_gate:
                attn_output = attn_output * torch.sigmoid(gate)

            # o_proj
            attn_output = attn_output.reshape(N_SEQUENCES, SUFFIX_LEN, 1, attn_module.q_output_size_per_tp).contiguous()
            attn_output = attn_module.o_proj(attn_output)[0]

        hidden_suffix = residual + attn_output

        # MLP
        residual = hidden_suffix
        hidden_normed = layer.post_attention_layernorm(hidden_suffix)
        mlp_output = layer.mlp(hidden_normed)
        hidden_suffix = residual + mlp_output

    # Final layernorm
    hidden_suffix = model.model.norm(hidden_suffix)

    # Compute logits for suffix portion
    from megatron.core import tensor_parallel as tp
    logits_suffix = model.lm_head(hidden_suffix)[0]
    logits_suffix = tp.gather_from_tensor_model_parallel_region(logits_suffix)
    logits_suffix = logits_suffix.float()

torch.cuda.synchronize()
t_suffix = time.time() - t2

if local_rank == 0:
    print(f"[Rank 0] Suffix pass: {t_suffix:.3f}s")
    print(f"[Rank 0] Suffix logits shape: {logits_suffix.shape}")

# ====================================================================
# Step 4: Precision alignment check
# ====================================================================
print(f"[Rank {local_rank}] === Step 4: Precision alignment ===")

# Reference: logits_normal at suffix positions
# PS: logits_suffix (N, SUFFIX_LEN, vocab_size)
normal_suffix_logits = logits_normal[:, PREFIX_LEN:, :]  # (N, SUFFIX_LEN, vocab_size)
ps_suffix_logits = logits_suffix  # (N, SUFFIX_LEN, vocab_size)

all_cos_sims = []
all_max_diffs = []
all_mean_diffs = []

for seq_idx in range(N_SEQUENCES):
    per_pos_cos = F.cosine_similarity(
        ps_suffix_logits[seq_idx].float(), normal_suffix_logits[seq_idx].float(), dim=-1
    )
    mean_cos = per_pos_cos.mean().item()
    min_cos = per_pos_cos.min().item()
    diff = (ps_suffix_logits[seq_idx].float() - normal_suffix_logits[seq_idx].float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    all_cos_sims.append(mean_cos)
    all_max_diffs.append(max_diff)
    all_mean_diffs.append(mean_diff)

    if local_rank == 0:
        print(f"[Rank 0] Seq {seq_idx}: cos_sim={mean_cos:.6f} (min={min_cos:.6f}), "
              f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        # Print per-position cos_sim for first 10 and last 5
        first_10 = [f"{v:.4f}" for v in per_pos_cos[:10].tolist()]
        last_5 = [f"{v:.4f}" for v in per_pos_cos[-5:].tolist()]
        print(f"  Per-pos: first10={first_10}, last5={last_5}")

overall_cos = sum(all_cos_sims) / len(all_cos_sims)
overall_max = max(all_max_diffs)
overall_mean = sum(all_mean_diffs) / len(all_mean_diffs)

COS_SIM_THRESHOLD = 0.999
MAX_DIFF_THRESHOLD = 0.01

passed = overall_cos >= COS_SIM_THRESHOLD and overall_max <= MAX_DIFF_THRESHOLD

if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"TWO-PASS E2E PREFIX-SHARING PRECISION ALIGNMENT")
    print(f"{'='*60}")
    print(f"Model: Qwen3.6-27B (TP={TP_SIZE}, bf16)")
    print(f"Pattern: GRPO n={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"DeltaNet layers: {len(deltanet_states)} (state injection + conv1d overlap)")
    print(f"Full attn layers: {len(prefix_kv_store)} (KV expansion)")
    print(f"Normal forward: {t_normal:.3f}s")
    print(f"Prefix pass: {t_prefix:.3f}s")
    print(f"Suffix pass: {t_suffix:.3f}s")
    print(f"Total PS time: {t_prefix + t_suffix:.3f}s")
    print(f"PS computation savings: {(N_SEQUENCES * total_len) / (PREFIX_LEN + N_SEQUENCES * SUFFIX_LEN):.2f}x")
    print(f"{'='*60}")
    print(f"Overall cos_sim: {overall_cos:.6f} (threshold: {COS_SIM_THRESHOLD})")
    print(f"Overall max_diff: {overall_max:.6f} (threshold: {MAX_DIFF_THRESHOLD})")
    print(f"Overall mean_diff: {overall_mean:.6f}")
    print(f"{'='*60}")
    if passed:
        print("PASS: Two-pass prefix-sharing precision alignment within bf16 tolerance!")
    else:
        print("FAIL: Precision misalignment detected!")
        if overall_cos < COS_SIM_THRESHOLD:
            print(f"  - cos_sim {overall_cos:.6f} < {COS_SIM_THRESHOLD}")
        if overall_max > MAX_DIFF_THRESHOLD:
            print(f"  - max_diff {overall_max:.6f} > {MAX_DIFF_THRESHOLD}")
    print(f"{'='*60}")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()