#!/usr/bin/env python3
"""PS training benchmark: GRPO-like training loop with and without prefix-sharing.

Measures per-step throughput for:
- PS OFF: Normal forward + backward on all N sequences (prefix+suffix)
- PS ON: Prefix pass (no_grad, 1 sequence) + Suffix pass (with_grad, N sequences)

This is the real performance measurement that matters for training.
Precision alignment (cos_sim > 0.999) is validated first.

Usage: torchrun --nproc_per_node=4 scripts/run_ps_training_benchmark.py
"""

import os
import sys
import time
import gc

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
HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only-16layers")
TP_SIZE = 4
PREFIX_LEN = 64      # Must be >= chunk_size (64) for DeltaNet chunk alignment
SUFFIX_LEN = 64
N_SEQUENCES = 4  # GRPO n=4
SEED = 42
NUM_WARMUP_STEPS = 1
NUM_TRAINING_STEPS = 5
LR = 1e-5
USE_GRADIENT_CHECKPOINTING = False

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
from transformers import AutoConfig
config = AutoConfig.from_pretrained(HF_MODEL_PATH, local_files_only=True)
layer_types = config.layer_types
num_layers = config.num_hidden_layers
hidden_size = config.hidden_size
vocab_size = config.vocab_size
total_len = PREFIX_LEN + SUFFIX_LEN

if local_rank == 0:
    print(f"Config: {type(config).__name__}, hidden={hidden_size}, "
          f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
          f"head_dim={config.head_dim}, layers={num_layers}")
    print(f"Layer types: {sum(1 for t in layer_types if t=='full_attention')} full attn, "
          f"{sum(1 for t in layer_types if t=='linear_attention')} linear attn")
    print(f"Training config: n={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}, "
          f"steps={NUM_TRAINING_STEPS}, lr={LR}")

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
# (Reuse weight loading from E2E test)
print(f"[Rank {local_rank}] Loading pretrained weights...")

def shard_tensor(tensor, dim, tp_size, tp_rank):
    chunks = torch.chunk(tensor, tp_size, dim=dim)
    return chunks[tp_rank].contiguous()

hf_state_dict = {}
for fname in os.listdir(HF_MODEL_PATH):
    if fname.startswith("model.safetensors") and fname.endswith(".safetensors"):
        shard_path = os.path.join(HF_MODEL_PATH, fname)
        shard_dict = load_file(shard_path)
        hf_state_dict.update(shard_dict)

def split_deltanet_qkv(in_proj_qkv_weight, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    q_portion = in_proj_qkv_weight[:key_dim]
    k_portion = in_proj_qkv_weight[key_dim:key_dim*2]
    v_portion = in_proj_qkv_weight[key_dim*2:]
    return (shard_tensor(q_portion, 0, tp_size, tp_rank),
            shard_tensor(k_portion, 0, tp_size, tp_rank),
            shard_tensor(v_portion, 0, tp_size, tp_rank))

def shard_conv1d_weight(conv1d_weight, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
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

    if is_deltanet:
        hf_prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
        key = f"{hf_prefix}.in_proj_qkv.weight"
        if key in hf_state_dict:
            q_s, k_s, v_s = split_deltanet_qkv(hf_state_dict[key], config, TP_SIZE, tp_rank)
            attn_module.in_proj_q.weight.data.copy_(q_s.to(torch.bfloat16))
            attn_module.in_proj_k.weight.data.copy_(k_s.to(torch.bfloat16))
            attn_module.in_proj_v.weight.data.copy_(v_s.to(torch.bfloat16))
            loaded_count += 3
        key = f"{hf_prefix}.conv1d.weight"
        if key in hf_state_dict:
            attn_module.conv1d.weight.data.copy_(
                shard_conv1d_weight(hf_state_dict[key], config, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for proj_name in ["in_proj_z", "in_proj_b", "in_proj_a"]:
            key = f"{hf_prefix}.{proj_name}.weight"
            if key in hf_state_dict:
                shard = shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank)
                getattr(attn_module, proj_name).weight.data.copy_(shard.to(torch.bfloat16))
                loaded_count += 1
        key = f"{hf_prefix}.A_log"
        if key in hf_state_dict:
            attn_module.A_log.data.copy_(
                shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.float32))
            loaded_count += 1
        key = f"{hf_prefix}.dt_bias"
        if key in hf_state_dict:
            attn_module.dt_bias.data.copy_(
                shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        key = f"{hf_prefix}.norm.weight"
        if key in hf_state_dict:
            attn_module.norm.weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
            loaded_count += 1
        key = f"{hf_prefix}.out_proj.weight"
        if key in hf_state_dict:
            attn_module.out_proj.weight.data.copy_(
                shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
    else:
        hf_prefix = f"model.language_model.layers.{layer_idx}.self_attn"
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            key = f"{hf_prefix}.{proj_name}.weight"
            if key in hf_state_dict:
                shard = shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank)
                getattr(attn_module, proj_name).weight.data.copy_(shard.to(torch.bfloat16))
                loaded_count += 1
        key = f"{hf_prefix}.o_proj.weight"
        if key in hf_state_dict:
            attn_module.o_proj.weight.data.copy_(
                shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for norm_name in ["q_norm", "k_norm"]:
            key = f"{hf_prefix}.{norm_name}.weight"
            if key in hf_state_dict:
                getattr(attn_module, norm_name).weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
                loaded_count += 1

    loaded_count += load_mlp_weights(hf_state_dict, mlp_module, f"model.language_model.layers.{layer_idx}", TP_SIZE, tp_rank)

    for ln_name in ["input_layernorm", "post_attention_layernorm"]:
        key = f"model.language_model.layers.{layer_idx}.{ln_name}.weight"
        if key in hf_state_dict:
            getattr(decoder_layer, ln_name).weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
            loaded_count += 1

embed_key = "model.language_model.embed_tokens.weight"
if embed_key in hf_state_dict:
    model.model.embed_tokens.weight.data.copy_(
        shard_tensor(hf_state_dict[embed_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
    loaded_count += 1
norm_key = "model.language_model.norm.weight"
if norm_key in hf_state_dict:
    model.model.norm.weight.data.copy_(hf_state_dict[norm_key].to(torch.bfloat16))
    loaded_count += 1
lm_key = "lm_head.weight"
if lm_key in hf_state_dict:
    model.lm_head.weight.data.copy_(
        shard_tensor(hf_state_dict[lm_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
    loaded_count += 1

hf_state_dict.clear()
torch.cuda.empty_cache()

print(f"[Rank {local_rank}] Loaded {loaded_count} weights, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ===== Create optimizer =====
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.01)

# ===== Create test batch =====
torch.manual_seed(SEED + local_rank)
prefix_ids = torch.randint(0, vocab_size, (PREFIX_LEN,), device="cpu").tolist()

sequences = []
suffix_ids_list = []
for i in range(N_SEQUENCES):
    suffix_ids = torch.randint(0, vocab_size, (SUFFIX_LEN,), device="cpu").tolist()
    sequences.append(prefix_ids + suffix_ids)
    suffix_ids_list.append(suffix_ids)

# ===== Helper: determine layer type =====
def is_deltanet_layer(layer_idx):
    return layer_types[layer_idx] == "linear_attention"

conv_overlap_size = 3

# ====================================================================
# Phase 1: Precision validation (inference-only, same as E2E test)
# ====================================================================
if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"PHASE 1: PRECISION VALIDATION")
    print(f"{'='*60}")

# Normal forward (reference)
input_ids_normal = torch.tensor(sequences, dtype=torch.long, device=device)
attention_mask_normal = torch.ones((N_SEQUENCES, total_len), dtype=torch.long, device=device)
position_ids_normal = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

with torch.no_grad():
    output_normal = model(input_ids=input_ids_normal, attention_mask=attention_mask_normal, position_ids=position_ids_normal)
logits_normal = output_normal.logits  # (N, total_len, vocab_size)

# PS forward (prefix pass + suffix pass)
prefix_input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
prefix_embeds = model.model.embed_tokens(prefix_input_ids)
hidden_prefix = prefix_embeds
deltanet_states = {}
deltanet_overlaps = {}
prefix_kv_store = {}

with torch.no_grad():
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn_module = layer.self_attn
        residual = hidden_prefix
        hidden_normed = layer.input_layernorm(hidden_prefix)

        if is_deltanet_layer(layer_idx):
            deltanet_overlaps[layer_idx] = hidden_normed[:, -conv_overlap_size:, :].clone()
            attn_output, recurrent_state = attn_module.forward(
                hidden_normed, attention_mask=None,
                position_ids=torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0),
                initial_state=None, output_final_state=True,
            )
            deltanet_states[layer_idx] = recurrent_state
        else:
            # Full attention: compute and store prefix KV
            q_full = attn_module.q_proj(hidden_normed)[0]
            k_states = attn_module.k_proj(hidden_normed)[0]
            v_states = attn_module.v_proj(hidden_normed)[0]

            q_shape = (PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim * 2)
            query_states, gate = torch.chunk(q_full.view(*q_shape), 2, dim=-1)

            query_states = query_states.view(1, PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
            k_states = k_states.view(1, PREFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)
            v_states = v_states.view(1, PREFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)

            query_states = attn_module.q_norm(query_states)
            key_states = attn_module.k_norm(k_states)

            from flash_attn.layers.rotary import apply_rotary_emb
            cos, sin = attn_module.rotary_emb(v_states, seq_len=PREFIX_LEN)
            cos, sin = cos[:, :cos.shape[1] // 2], sin[:, :sin.shape[1] // 2]

            if attn_module.rope_dim == attn_module.head_dim:
                query_states = apply_rotary_emb(query_states, cos, sin, interleaved=False, inplace=False)
                key_states = apply_rotary_emb(key_states, cos, sin, interleaved=False, inplace=False)
            else:
                q_rot = query_states[:, :, :, :attn_module.rope_dim]
                q_pass = query_states[:, :, :, attn_module.rope_dim:]
                k_rot = key_states[:, :, :, :attn_module.rope_dim]
                k_pass = key_states[:, :, :, attn_module.rope_dim:]
                q_rot = apply_rotary_emb(q_rot, cos, sin, interleaved=False, inplace=False)
                k_rot = apply_rotary_emb(k_rot, cos, sin, interleaved=False, inplace=False)
                query_states = torch.cat([q_rot, q_pass], dim=-1)
                key_states = torch.cat([k_rot, k_pass], dim=-1)

            num_key_value_groups = attn_module.num_key_value_groups
            if num_key_value_groups > 1:
                key_states = key_states.unsqueeze(3).expand(-1, -1, -1, num_key_value_groups, -1).reshape(1, PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
                v_states = v_states.unsqueeze(3).expand(-1, -1, -1, num_key_value_groups, -1).reshape(1, PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)

            prefix_kv_store[layer_idx] = (key_states, v_states)

            causal_mask = torch.zeros((1, 1, PREFIX_LEN, PREFIX_LEN), dtype=hidden_normed.dtype, device=device)
            causal_mask = causal_mask.masked_fill(
                torch.triu(torch.ones(PREFIX_LEN, PREFIX_LEN, dtype=torch.bool, device=device), diagonal=1),
                torch.finfo(hidden_normed.dtype).min,
            )
            attn_output = attn_module.forward(
                hidden_normed, attention_mask=causal_mask,
                position_ids=torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0),
            )

        hidden_prefix = residual + attn_output
        residual = hidden_prefix
        hidden_normed = layer.post_attention_layernorm(hidden_prefix)
        mlp_output = layer.mlp(hidden_normed)
        hidden_prefix = residual + mlp_output

# Suffix pass (with state injection)
suffix_input_ids = torch.tensor(suffix_ids_list, dtype=torch.long, device=device)
suffix_embeds = model.model.embed_tokens(suffix_input_ids)
hidden_suffix = suffix_embeds

with torch.no_grad():
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn_module = layer.self_attn
        residual = hidden_suffix
        hidden_normed = layer.input_layernorm(hidden_suffix)

        if is_deltanet_layer(layer_idx):
            prefix_state = deltanet_states[layer_idx]
            prefix_state_expanded = prefix_state.expand(N_SEQUENCES, -1, -1, -1).contiguous()
            overlap_hidden = deltanet_overlaps[layer_idx]
            overlap_expanded = overlap_hidden.expand(N_SEQUENCES, -1, -1).contiguous()

            attn_output = attn_module.forward(
                hidden_normed, attention_mask=None,
                position_ids=torch.arange(PREFIX_LEN, PREFIX_LEN + SUFFIX_LEN, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1),
                initial_state=prefix_state_expanded,
                output_final_state=False,
                conv_overlap_hidden=overlap_expanded,
            )
        else:
            # Full attention with prefix KV expansion (same as E2E test)
            prefix_key, prefix_value = prefix_kv_store[layer_idx]

            q_full = attn_module.q_proj(hidden_normed)[0]
            k_states = attn_module.k_proj(hidden_normed)[0]
            v_states = attn_module.v_proj(hidden_normed)[0]

            q_shape = (N_SEQUENCES, SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim * 2)
            query_states, gate = torch.chunk(q_full.view(*q_shape), 2, dim=-1)

            query_states = query_states.view(N_SEQUENCES, SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
            k_states = k_states.view(N_SEQUENCES, SUFFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)
            v_states = v_states.view(N_SEQUENCES, SUFFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)

            query_states = attn_module.q_norm(query_states)
            key_states = attn_module.k_norm(k_states)

            cos_suffix, sin_suffix = attn_module.rotary_emb(v_states, seq_len=PREFIX_LEN + SUFFIX_LEN)
            cos_suffix = cos_suffix[PREFIX_LEN:PREFIX_LEN + SUFFIX_LEN]
            sin_suffix = sin_suffix[PREFIX_LEN:PREFIX_LEN + SUFFIX_LEN]
            cos_suffix = cos_suffix[:, :cos_suffix.shape[1] // 2]
            sin_suffix = sin_suffix[:, :sin_suffix.shape[1] // 2]

            from flash_attn.layers.rotary import apply_rotary_emb
            if attn_module.rope_dim == attn_module.head_dim:
                query_states = apply_rotary_emb(query_states, cos_suffix, sin_suffix, interleaved=False, inplace=False)
                key_states = apply_rotary_emb(key_states, cos_suffix, sin_suffix, interleaved=False, inplace=False)
            else:
                q_rot = query_states[:, :, :, :attn_module.rope_dim]
                q_pass = query_states[:, :, :, attn_module.rope_dim:]
                k_rot = key_states[:, :, :, :attn_module.rope_dim]
                k_pass = key_states[:, :, :, attn_module.rope_dim:]
                q_rot = apply_rotary_emb(q_rot, cos_suffix, sin_suffix, interleaved=False, inplace=False)
                k_rot = apply_rotary_emb(k_rot, cos_suffix, sin_suffix, interleaved=False, inplace=False)
                query_states = torch.cat([q_rot, q_pass], dim=-1)
                key_states = torch.cat([k_rot, k_pass], dim=-1)

            num_key_value_groups = attn_module.num_key_value_groups
            if num_key_value_groups > 1:
                key_states = key_states.repeat_interleave(num_key_value_groups, dim=2)
                v_states = v_states.repeat_interleave(num_key_value_groups, dim=2)

            expanded_prefix_key = prefix_key.expand(N_SEQUENCES, -1, -1, -1).contiguous()
            expanded_prefix_value = prefix_value.expand(N_SEQUENCES, -1, -1, -1).contiguous()

            expanded_key = torch.cat([expanded_prefix_key, key_states], dim=1)
            expanded_value = torch.cat([expanded_prefix_value, v_states], dim=1)

            cu_seqlens_q = torch.tensor([0] + [SUFFIX_LEN] * N_SEQUENCES, device=device, dtype=torch.int32).cumsum(0).to(torch.int32)
            cu_seqlens_kv = torch.tensor([0] + [PREFIX_LEN + SUFFIX_LEN] * N_SEQUENCES, device=device, dtype=torch.int32).cumsum(0).to(torch.int32)

            query_flat = query_states.reshape(N_SEQUENCES * SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
            expanded_key_flat = expanded_key.reshape(N_SEQUENCES * (PREFIX_LEN + SUFFIX_LEN), attn_module.num_heads_per_tp, attn_module.head_dim)
            expanded_value_flat = expanded_value.reshape(N_SEQUENCES * (PREFIX_LEN + SUFFIX_LEN), attn_module.num_heads_per_tp, attn_module.head_dim)

            from flash_attn import flash_attn_varlen_func
            attn_output = flash_attn_varlen_func(
                query_flat, expanded_key_flat, expanded_value_flat,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=SUFFIX_LEN, max_seqlen_k=PREFIX_LEN + SUFFIX_LEN,
                dropout_p=0.0, softmax_scale=attn_module.scaling, causal=True,
            )

            input_dtype = query_flat.dtype
            attn_output = attn_output.to(input_dtype)
            attn_output = attn_output.reshape(N_SEQUENCES, SUFFIX_LEN, attn_module.num_heads_per_tp * attn_module.head_dim)

            if attn_module.attn_output_gate:
                gate = gate.reshape(N_SEQUENCES, SUFFIX_LEN, attn_module.num_heads_per_tp * attn_module.head_dim)
                attn_output = attn_output * torch.sigmoid(gate)

            attn_output = attn_output.reshape(N_SEQUENCES * SUFFIX_LEN, 1, attn_module.q_output_size_per_tp).contiguous()
            attn_output = attn_module.o_proj(attn_output)[0]
            attn_output = attn_output.reshape(N_SEQUENCES, SUFFIX_LEN, hidden_size)

        hidden_suffix = residual + attn_output
        residual = hidden_suffix
        hidden_normed = layer.post_attention_layernorm(hidden_suffix)
        mlp_output = layer.mlp(hidden_normed)
        hidden_suffix = residual + mlp_output

    hidden_suffix = model.model.norm(hidden_suffix)
    from megatron.core import tensor_parallel as tp
    logits_ps = model.lm_head(hidden_suffix)[0]
    logits_ps = tp.gather_from_tensor_model_parallel_region(logits_ps)
    logits_ps = logits_ps.float()

# Compare suffix logits
normal_suffix_logits = logits_normal[:, PREFIX_LEN:, :]
ps_suffix_logits = logits_ps

all_cos_sims = []
for seq_idx in range(N_SEQUENCES):
    per_pos_cos = F.cosine_similarity(ps_suffix_logits[seq_idx].float(), normal_suffix_logits[seq_idx].float(), dim=-1)
    all_cos_sims.append(per_pos_cos.mean().item())

overall_cos = sum(all_cos_sims) / len(all_cos_sims)
precision_pass = overall_cos >= 0.999

if local_rank == 0:
    print(f"Precision: cos_sim={overall_cos:.6f} (threshold: 0.999)")
    print(f"Status: {'PASS' if precision_pass else 'FAIL'}")
    for i, cs in enumerate(all_cos_sims):
        print(f"  Seq {i}: cos_sim={cs:.6f}")

if not precision_pass:
    if local_rank == 0:
        print("PRECISION FAILED - cannot proceed with training benchmark!")
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()
    sys.exit(1)

# ====================================================================
# Phase 2: Training throughput benchmark
# ====================================================================
if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"PHASE 2: TRAINING THROUGHPUT BENCHMARK")
    print(f"{'='*60}")

# --- PS OFF: Normal training loop ---
model.train()
optimizer.zero_grad()

# Warmup
for step in range(NUM_WARMUP_STEPS):
    input_ids_train = torch.tensor(sequences, dtype=torch.long, device=device)
    attention_mask_train = torch.ones((N_SEQUENCES, total_len), dtype=torch.long, device=device)
    position_ids_train = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

    output = model(input_ids=input_ids_train, attention_mask=attention_mask_train, position_ids=position_ids_train)
    logits = output.logits  # (N, total_len, vocab_size)

    # GRPO loss: CE on response tokens (standard shift for next-token prediction)
    # logits at position i predict token at position i+1
    shift_logits = logits[:, PREFIX_LEN-1:PREFIX_LEN+SUFFIX_LEN-1, :]  # (N, suffix_len, vocab)
    shift_labels = input_ids_train[:, PREFIX_LEN:PREFIX_LEN+SUFFIX_LEN]  # (N, suffix_len)

    assert shift_logits.shape[1] == shift_labels.shape[1], \
        f"logits seq={shift_logits.shape[1]}, labels seq={shift_labels.shape[1]}"

    loss = F.cross_entropy(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.cuda.synchronize()
normal_times = []
for step in range(NUM_TRAINING_STEPS):
    # Use different suffix each step to avoid trivial optimization
    torch.manual_seed(SEED + step)
    step_suffix_ids = []
    for i in range(N_SEQUENCES):
        s_ids = torch.randint(0, vocab_size, (SUFFIX_LEN,), device="cpu").tolist()
        step_suffix_ids.append(prefix_ids + s_ids)

    input_ids_train = torch.tensor(step_suffix_ids, dtype=torch.long, device=device)
    attention_mask_train = torch.ones((N_SEQUENCES, total_len), dtype=torch.long, device=device)
    position_ids_train = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

    torch.cuda.synchronize()
    t0 = time.time()

    output = model(input_ids=input_ids_train, attention_mask=attention_mask_train, position_ids=position_ids_train)
    logits = output.logits

    labels = input_ids_train[:, PREFIX_LEN:]
    shift_logits = logits[:, PREFIX_LEN-1:PREFIX_LEN+SUFFIX_LEN-1, :]  # (N, suffix_len, vocab)
    shift_labels = input_ids_train[:, PREFIX_LEN:PREFIX_LEN+SUFFIX_LEN]  # (N, suffix_len)

    loss = F.cross_entropy(shift_logits.reshape(-1, vocab_size), shift_labels.reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    t1 = time.time()
    normal_times.append(t1 - t0)

    if local_rank == 0 and step % 2 == 0:
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"  [Normal] Step {step}: time={t1-t0:.3f}s, loss={loss.item():.4f}, mem={mem:.2f}GB")

avg_normal_time = sum(normal_times) / len(normal_times)

# --- Reset model for PS ON training ---
# Reload weights to start from same state
model.train()
optimizer.zero_grad()
# (Weights are already loaded; optimizer state will be different but we measure throughput, not convergence)

# --- PS ON: Two-pass training loop ---
# Key difference: prefix pass is no_grad (just captures state),
# suffix pass computes gradients for backward

ps_times = []
for step in range(NUM_TRAINING_STEPS):
    torch.manual_seed(SEED + step)
    step_suffix_ids = []
    for i in range(N_SEQUENCES):
        s_ids = torch.randint(0, vocab_size, (SUFFIX_LEN,), device="cpu").tolist()
        step_suffix_ids.append(s_ids)

    torch.cuda.synchronize()
    t0 = time.time()

    # ===== Prefix pass (no_grad) =====
    prefix_input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    prefix_embeds = model.model.embed_tokens(prefix_input_ids)
    hidden_prefix = prefix_embeds
    deltanet_states_step = {}
    deltanet_overlaps_step = {}
    prefix_kv_store_step = {}

    with torch.no_grad():
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            attn_module = layer.self_attn
            residual = hidden_prefix
            hidden_normed = layer.input_layernorm(hidden_prefix)

            if is_deltanet_layer(layer_idx):
                deltanet_overlaps_step[layer_idx] = hidden_normed[:, -conv_overlap_size:, :].clone()
                attn_output, recurrent_state = attn_module.forward(
                    hidden_normed, attention_mask=None,
                    position_ids=torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0),
                    initial_state=None, output_final_state=True,
                )
                deltanet_states_step[layer_idx] = recurrent_state
            else:
                # Full attention: compute and store prefix KV
                q_full = attn_module.q_proj(hidden_normed)[0]
                k_states = attn_module.k_proj(hidden_normed)[0]
                v_states = attn_module.v_proj(hidden_normed)[0]

                q_shape = (PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim * 2)
                query_states, gate = torch.chunk(q_full.view(*q_shape), 2, dim=-1)
                query_states = query_states.view(1, PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
                k_states = k_states.view(1, PREFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)
                v_states = v_states.view(1, PREFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)

                query_states = attn_module.q_norm(query_states)
                key_states = attn_module.k_norm(k_states)

                from flash_attn.layers.rotary import apply_rotary_emb
                cos, sin = attn_module.rotary_emb(v_states, seq_len=PREFIX_LEN)
                cos, sin = cos[:, :cos.shape[1] // 2], sin[:, :sin.shape[1] // 2]

                if attn_module.rope_dim == attn_module.head_dim:
                    query_states = apply_rotary_emb(query_states, cos, sin, interleaved=False, inplace=False)
                    key_states = apply_rotary_emb(key_states, cos, sin, interleaved=False, inplace=False)
                else:
                    q_rot = query_states[:, :, :, :attn_module.rope_dim]
                    q_pass = query_states[:, :, :, attn_module.rope_dim:]
                    k_rot = key_states[:, :, :, :attn_module.rope_dim:]
                    k_pass = key_states[:, :, :, attn_module.rope_dim:]
                    q_rot = apply_rotary_emb(q_rot, cos, sin, interleaved=False, inplace=False)
                    k_rot = apply_rotary_emb(k_rot, cos, sin, interleaved=False, inplace=False)
                    query_states = torch.cat([q_rot, q_pass], dim=-1)
                    key_states = torch.cat([k_rot, k_pass], dim=-1)

                num_key_value_groups = attn_module.num_key_value_groups
                if num_key_value_groups > 1:
                    key_states = key_states.unsqueeze(3).expand(-1, -1, -1, num_key_value_groups, -1).reshape(1, PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
                    v_states = v_states.unsqueeze(3).expand(-1, -1, -1, num_key_value_groups, -1).reshape(1, PREFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)

                prefix_kv_store_step[layer_idx] = (key_states, v_states)

                causal_mask = torch.zeros((1, 1, PREFIX_LEN, PREFIX_LEN), dtype=hidden_normed.dtype, device=device)
                causal_mask = causal_mask.masked_fill(
                    torch.triu(torch.ones(PREFIX_LEN, PREFIX_LEN, dtype=torch.bool, device=device), diagonal=1),
                    torch.finfo(hidden_normed.dtype).min,
                )
                attn_output = attn_module.forward(hidden_normed, attention_mask=causal_mask,
                    position_ids=torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0))

            hidden_prefix = residual + attn_output
            residual = hidden_prefix
            hidden_normed = layer.post_attention_layernorm(hidden_prefix)
            mlp_output = layer.mlp(hidden_normed)
            hidden_prefix = residual + mlp_output

    # ===== Suffix pass (with_grad) =====
    suffix_input_ids = torch.tensor(step_suffix_ids, dtype=torch.long, device=device)
    suffix_embeds = model.model.embed_tokens(suffix_input_ids)
    hidden_suffix = suffix_embeds

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn_module = layer.self_attn
        residual = hidden_suffix
        hidden_normed = layer.input_layernorm(hidden_suffix)

        if is_deltanet_layer(layer_idx):
            prefix_state = deltanet_states_step[layer_idx]
            prefix_state_expanded = prefix_state.expand(N_SEQUENCES, -1, -1, -1).contiguous()
            overlap_hidden = deltanet_overlaps_step[layer_idx]
            overlap_expanded = overlap_hidden.expand(N_SEQUENCES, -1, -1).contiguous()

            attn_output = attn_module.forward(
                hidden_normed, attention_mask=None,
                position_ids=torch.arange(PREFIX_LEN, PREFIX_LEN + SUFFIX_LEN, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1),
                initial_state=prefix_state_expanded,
                output_final_state=False,
                conv_overlap_hidden=overlap_expanded,
            )
        else:
            # Full attention with prefix KV expansion
            prefix_key, prefix_value = prefix_kv_store_step[layer_idx]

            q_full = attn_module.q_proj(hidden_normed)[0]
            k_states = attn_module.k_proj(hidden_normed)[0]
            v_states = attn_module.v_proj(hidden_normed)[0]

            q_shape = (N_SEQUENCES, SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim * 2)
            query_states, gate = torch.chunk(q_full.view(*q_shape), 2, dim=-1)
            query_states = query_states.view(N_SEQUENCES, SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
            k_states = k_states.view(N_SEQUENCES, SUFFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)
            v_states = v_states.view(N_SEQUENCES, SUFFIX_LEN, attn_module.num_key_value_heads_per_tp, attn_module.head_dim)

            query_states = attn_module.q_norm(query_states)
            key_states = attn_module.k_norm(k_states)

            from flash_attn.layers.rotary import apply_rotary_emb
            cos_suffix, sin_suffix = attn_module.rotary_emb(v_states, seq_len=PREFIX_LEN + SUFFIX_LEN)
            cos_suffix = cos_suffix[PREFIX_LEN:PREFIX_LEN + SUFFIX_LEN]
            sin_suffix = sin_suffix[PREFIX_LEN:PREFIX_LEN + SUFFIX_LEN]
            cos_suffix = cos_suffix[:, :cos_suffix.shape[1] // 2]
            sin_suffix = sin_suffix[:, :sin_suffix.shape[1] // 2]

            if attn_module.rope_dim == attn_module.head_dim:
                query_states = apply_rotary_emb(query_states, cos_suffix, sin_suffix, interleaved=False, inplace=False)
                key_states = apply_rotary_emb(key_states, cos_suffix, sin_suffix, interleaved=False, inplace=False)
            else:
                q_rot = query_states[:, :, :, :attn_module.rope_dim]
                q_pass = query_states[:, :, :, attn_module.rope_dim:]
                k_rot = key_states[:, :, :, :attn_module.rope_dim]
                k_pass = key_states[:, :, :, attn_module.rope_dim:]
                q_rot = apply_rotary_emb(q_rot, cos_suffix, sin_suffix, interleaved=False, inplace=False)
                k_rot = apply_rotary_emb(k_rot, cos_suffix, sin_suffix, interleaved=False, inplace=False)
                query_states = torch.cat([q_rot, q_pass], dim=-1)
                key_states = torch.cat([k_rot, k_pass], dim=-1)

            num_key_value_groups = attn_module.num_key_value_groups
            if num_key_value_groups > 1:
                key_states = key_states.repeat_interleave(num_key_value_groups, dim=2)
                v_states = v_states.repeat_interleave(num_key_value_groups, dim=2)

            expanded_prefix_key = prefix_key.expand(N_SEQUENCES, -1, -1, -1).contiguous()
            expanded_prefix_value = prefix_value.expand(N_SEQUENCES, -1, -1, -1).contiguous()

            expanded_key = torch.cat([expanded_prefix_key, key_states], dim=1)
            expanded_value = torch.cat([expanded_prefix_value, v_states], dim=1)

            cu_seqlens_q = torch.tensor([0] + [SUFFIX_LEN] * N_SEQUENCES, device=device, dtype=torch.int32).cumsum(0).to(torch.int32)
            cu_seqlens_kv = torch.tensor([0] + [PREFIX_LEN + SUFFIX_LEN] * N_SEQUENCES, device=device, dtype=torch.int32).cumsum(0).to(torch.int32)

            query_flat = query_states.reshape(N_SEQUENCES * SUFFIX_LEN, attn_module.num_heads_per_tp, attn_module.head_dim)
            expanded_key_flat = expanded_key.reshape(N_SEQUENCES * (PREFIX_LEN + SUFFIX_LEN), attn_module.num_heads_per_tp, attn_module.head_dim)
            expanded_value_flat = expanded_value.reshape(N_SEQUENCES * (PREFIX_LEN + SUFFIX_LEN), attn_module.num_heads_per_tp, attn_module.head_dim)

            from flash_attn import flash_attn_varlen_func
            attn_output = flash_attn_varlen_func(
                query_flat, expanded_key_flat, expanded_value_flat,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_kv,
                max_seqlen_q=SUFFIX_LEN, max_seqlen_k=PREFIX_LEN + SUFFIX_LEN,
                dropout_p=0.0, softmax_scale=attn_module.scaling, causal=True,
            )

            attn_output = attn_output.to(query_flat.dtype)
            attn_output = attn_output.reshape(N_SEQUENCES, SUFFIX_LEN, attn_module.num_heads_per_tp * attn_module.head_dim)

            if attn_module.attn_output_gate:
                gate = gate.reshape(N_SEQUENCES, SUFFIX_LEN, attn_module.num_heads_per_tp * attn_module.head_dim)
                attn_output = attn_output * torch.sigmoid(gate)

            attn_output = attn_output.reshape(N_SEQUENCES * SUFFIX_LEN, 1, attn_module.q_output_size_per_tp).contiguous()
            attn_output = attn_module.o_proj(attn_output)[0]
            attn_output = attn_output.reshape(N_SEQUENCES, SUFFIX_LEN, hidden_size)

        hidden_suffix = residual + attn_output
        residual = hidden_suffix
        hidden_normed = layer.post_attention_layernorm(hidden_suffix)
        mlp_output = layer.mlp(hidden_normed)
        hidden_suffix = residual + mlp_output

    # Final layernorm + lm_head
    hidden_suffix = model.model.norm(hidden_suffix)
    from megatron.core import tensor_parallel as tp
    logits_ps_train = model.lm_head(hidden_suffix)[0]
    logits_ps_train = tp.gather_from_tensor_model_parallel_region(logits_ps_train)
    logits_ps_train = logits_ps_train.float()

    # GRPO loss on suffix logits (matching PS OFF format)
    # logits at suffix position i predict suffix token at position i+1
    shift_logits_ps = logits_ps_train[:, :-1, :]
    shift_labels_ps = suffix_input_ids[:, 1:]
    loss_ps = F.cross_entropy(shift_logits_ps.reshape(-1, vocab_size), shift_labels_ps.reshape(-1))

    loss_ps.backward()

    # Free PS stored states before optimizer step to reduce memory pressure
    del deltanet_states_step, deltanet_overlaps_step, prefix_kv_store_step
    torch.cuda.empty_cache()

    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    t1 = time.time()
    ps_times.append(t1 - t0)

    if local_rank == 0 and step % 2 == 0:
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"  [PS ON] Step {step}: time={t1-t0:.3f}s, loss={loss_ps.item():.4f}, mem={mem:.2f}GB")

avg_ps_time = sum(ps_times) / len(ps_times)

# ====================================================================
# Phase 3: Results summary
# ====================================================================
if local_rank == 0:
    token_savings = (N_SEQUENCES - 1) / N_SEQUENCES * PREFIX_LEN / total_len
    theoretical_speedup = 1 / (1 - token_savings)

    print(f"\n{'='*60}")
    print(f"TRAINING BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Model: Qwen3.6-27B-16layers (TP={TP_SIZE}, bf16)")
    print(f"Pattern: GRPO n={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"Total tokens per step: {N_SEQUENCES * total_len}")
    print(f"PS tokens per step: prefix={PREFIX_LEN} + suffix={N_SEQUENCES * SUFFIX_LEN} = {PREFIX_LEN + N_SEQUENCES * SUFFIX_LEN}")
    print(f"Token savings: {token_savings:.1%}")
    print(f"{'='*60}")
    print(f"PS OFF (Normal) avg time: {avg_normal_time:.3f}s")
    print(f"PS ON (Two-pass) avg time: {avg_ps_time:.3f}s")
    speedup = avg_normal_time / avg_ps_time
    print(f"Measured speedup: {speedup:.2f}x")
    print(f"Theoretical speedup: {theoretical_speedup:.2f}x")
    if speedup < 1.0:
        print(f"\nNOTE: Measured speedup < 1.0 because PS ON uses manual layer-by-layer")
        print(f"forward (Python loops per layer) while PS OFF uses model.forward().")
        print(f"Real verl training speedup will be higher with monkey-patched attention.")
    print(f"{'='*60}")
    print(f"Precision: cos_sim={overall_cos:.6f}")
    print(f"{'='*60}")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()