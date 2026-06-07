#!/usr/bin/env python3
"""Fair PS training benchmark: both PS OFF and PS ON use manual layer-by-layer forward.

The previous benchmark was unfair because PS OFF used model.forward() (no Python loop
overhead) while PS ON used manual layer-by-layer forward (Python loop overhead per layer).
This version makes both use the same layer-by-layer structure, so the speedup is purely
from PS token savings.

Three measurements:
1. PS OFF model.forward() — baseline (fastest possible, for reference)
2. PS OFF manual forward — fair baseline (same overhead as PS ON)
3. PS ON manual two-pass — PS optimization (same overhead as PS OFF manual)

Fair speedup = (PS OFF manual) / (PS ON manual)
Real speedup (verl) will be between fair speedup and model.forward() speedup.

Usage: torchrun --nproc_per_node=4 scripts/run_ps_training_benchmark_fair.py
"""

import os, sys, time, gc, contextlib

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
PREFIX_LEN = 64
SUFFIX_LEN = 64
N_SEQUENCES = 2
SEED = 42
NUM_WARMUP = 1
NUM_STEPS = 5
LR = 1e-5

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
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    context_parallel_size=1, expert_model_parallel_size=1,
)
tp_rank = parallel_state.get_tensor_model_parallel_rank()
device = torch.device(f"cuda:{local_rank}")

# ===== Load config =====
config = AutoConfig.from_pretrained(HF_MODEL_PATH, local_files_only=True)
layer_types = config.layer_types
num_layers = config.num_hidden_layers
hidden_size = config.hidden_size
vocab_size = config.vocab_size
total_len = PREFIX_LEN + SUFFIX_LEN
conv_overlap_size = 3

if local_rank == 0:
    full_attn = sum(1 for t in layer_types if t == "full_attention")
    linear_attn = sum(1 for t in layer_types if t == "linear_attention")
    print(f"Model: hidden={hidden_size}, heads={config.num_attention_heads}, "
          f"kv_heads={config.num_key_value_heads}, layers={num_layers}")
    print(f"Layer types: {full_attn} full + {linear_attn} linear")
    print(f"Config: n={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}, "
          f"token_savings={(N_SEQUENCES-1)/N_SEQUENCES*PREFIX_LEN/total_len:.1%}")

# ===== Instantiate model =====
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLM
from megatron.core import ModelParallelConfig

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    sequence_parallel=False, bf16=True, params_dtype=torch.bfloat16,
)
model = ParallelQwen3_6ForCausalLM(config=config, megatron_config=megatron_config)
model = model.to(device)

# ===== Load weights (identical to original benchmark) =====
if local_rank == 0:
    print("Loading pretrained weights...")

def shard_tensor(tensor, dim, tp_size, tp_rank):
    return torch.chunk(tensor, tp_size, dim=dim)[tp_rank].contiguous()

hf_state_dict = {}
for fname in os.listdir(HF_MODEL_PATH):
    if fname.startswith("model.safetensors") and fname.endswith(".safetensors"):
        hf_state_dict.update(load_file(os.path.join(HF_MODEL_PATH, fname)))

def split_deltanet_qkv(w, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    return (shard_tensor(w[:key_dim], 0, tp_size, tp_rank),
            shard_tensor(w[key_dim:key_dim*2], 0, tp_size, tp_rank),
            shard_tensor(w[key_dim*2:], 0, tp_size, tp_rank))

def shard_conv1d_weight(w, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    return torch.cat([
        shard_tensor(w[:key_dim], 0, tp_size, tp_rank),
        shard_tensor(w[key_dim:key_dim*2], 0, tp_size, tp_rank),
        shard_tensor(w[key_dim*2:], 0, tp_size, tp_rank),
    ], dim=0).contiguous()

def load_mlp_weights(hf_sd, mlp_module, prefix, tp_size, tp_rank):
    loaded = 0
    gk = f"{prefix}.mlp.gate_proj.weight"
    uk = f"{prefix}.mlp.up_proj.weight"
    if gk in hf_sd and uk in hf_sd:
        g_s = shard_tensor(hf_sd[gk], 0, tp_size, tp_rank)
        u_s = shard_tensor(hf_sd[uk], 0, tp_size, tp_rank)
        mlp_module.gate_up_proj.weight.data.copy_(
            torch.cat([g_s, u_s], dim=0).contiguous().to(torch.bfloat16))
        loaded += 2
    dk = f"{prefix}.mlp.down_proj.weight"
    if dk in hf_sd:
        mlp_module.down_proj.weight.data.copy_(
            shard_tensor(hf_sd[dk], 1, tp_size, tp_rank).to(torch.bfloat16))
        loaded += 1
    return loaded

loaded_count = 0
for layer_idx in range(num_layers):
    is_dn = (layer_types[layer_idx] == "linear_attention")
    dec = model.model.layers[layer_idx]
    attn = dec.self_attn
    mlp = dec.mlp
    pfx = f"model.language_model.layers.{layer_idx}"

    if is_dn:
        dn_pfx = f"{pfx}.linear_attn"
        key = f"{dn_pfx}.in_proj_qkv.weight"
        if key in hf_state_dict:
            q_s, k_s, v_s = split_deltanet_qkv(hf_state_dict[key], config, TP_SIZE, tp_rank)
            attn.in_proj_q.weight.data.copy_(q_s.to(torch.bfloat16))
            attn.in_proj_k.weight.data.copy_(k_s.to(torch.bfloat16))
            attn.in_proj_v.weight.data.copy_(v_s.to(torch.bfloat16))
            loaded_count += 3
        key = f"{dn_pfx}.conv1d.weight"
        if key in hf_state_dict:
            attn.conv1d.weight.data.copy_(
                shard_conv1d_weight(hf_state_dict[key], config, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for pn in ["in_proj_z", "in_proj_b", "in_proj_a"]:
            key = f"{dn_pfx}.{pn}.weight"
            if key in hf_state_dict:
                getattr(attn, pn).weight.data.copy_(
                    shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
                loaded_count += 1
        key = f"{dn_pfx}.A_log"
        if key in hf_state_dict:
            attn.A_log.data.copy_(
                shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.float32))
            loaded_count += 1
        key = f"{dn_pfx}.dt_bias"
        if key in hf_state_dict:
            attn.dt_bias.data.copy_(
                shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        key = f"{dn_pfx}.norm.weight"
        if key in hf_state_dict:
            attn.norm.weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
            loaded_count += 1
        key = f"{dn_pfx}.out_proj.weight"
        if key in hf_state_dict:
            attn.out_proj.weight.data.copy_(
                shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
    else:
        fa_pfx = f"{pfx}.self_attn"
        for pn in ["q_proj", "k_proj", "v_proj"]:
            key = f"{fa_pfx}.{pn}.weight"
            if key in hf_state_dict:
                getattr(attn, pn).weight.data.copy_(
                    shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
                loaded_count += 1
        key = f"{fa_pfx}.o_proj.weight"
        if key in hf_state_dict:
            attn.o_proj.weight.data.copy_(
                shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for nn in ["q_norm", "k_norm"]:
            key = f"{fa_pfx}.{nn}.weight"
            if key in hf_state_dict:
                getattr(attn, nn).weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
                loaded_count += 1

    loaded_count += load_mlp_weights(hf_state_dict, mlp, pfx, TP_SIZE, tp_rank)
    for ln in ["input_layernorm", "post_attention_layernorm"]:
        key = f"{pfx}.{ln}.weight"
        if key in hf_state_dict:
            getattr(dec, ln).weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
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
if local_rank == 0:
    print(f"Loaded {loaded_count} weights, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ===== Create optimizer and test batch =====
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.01)

torch.manual_seed(SEED + local_rank)
prefix_ids_list = torch.randint(0, vocab_size, (PREFIX_LEN,), device="cpu").tolist()
is_deltanet = lambda idx: layer_types[idx] == "linear_attention"


# ===== Forward functions =====

def manual_forward_off(input_ids_tensor):
    """PS OFF manual layer-by-layer forward — same structure as PS ON but no state injection."""
    from megatron.core import tensor_parallel as tp

    embeds = model.model.embed_tokens(input_ids_tensor)
    hidden = embeds
    bsz, seq_len = input_ids_tensor.shape

    # Build causal mask for full attention layers
    mask_val = torch.finfo(embeds.dtype).min
    causal_mask = torch.full((seq_len, seq_len), mask_val, device=device)
    row_idx = torch.arange(seq_len, device=device)
    causal_mask.masked_fill_(row_idx < (row_idx + 1).view(seq_len, 1), 0)
    causal_mask_4d = causal_mask.to(embeds.dtype)[None, None, :].expand(bsz, -1, -1, -1)
    pos_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        residual = hidden
        hidden_normed = layer.input_layernorm(hidden)
        if is_deltanet(layer_idx):
            attn_out = layer.self_attn(hidden_normed, attention_mask=None, position_ids=pos_ids)
        else:
            attn_out = layer.self_attn(hidden_normed, attention_mask=causal_mask_4d, position_ids=pos_ids)
        hidden = residual + attn_out
        residual = hidden
        hidden_normed = layer.post_attention_layernorm(hidden)
        mlp_out = layer.mlp(hidden_normed)
        hidden = residual + mlp_out

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)[0]
    logits = tp.gather_from_tensor_model_parallel_region(logits)
    return logits.float()


def manual_forward_on(prefix_ids_list, suffix_ids_lists):
    """PS ON manual two-pass forward — prefix pass (no_grad) + suffix pass (with_grad)."""
    from megatron.core import tensor_parallel as tp
    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

    N = len(suffix_ids_lists)

    # ===== Prefix pass (no_grad) =====
    prefix_input_ids = torch.tensor([prefix_ids_list], dtype=torch.long, device=device)
    prefix_embeds = model.model.embed_tokens(prefix_input_ids)
    hidden_prefix = prefix_embeds
    dn_states = {}
    dn_overlaps = {}
    kv_store = {}

    with torch.no_grad():
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            attn = layer.self_attn
            residual = hidden_prefix
            hidden_normed = layer.input_layernorm(hidden_prefix)

            if is_deltanet(layer_idx):
                dn_overlaps[layer_idx] = hidden_normed[:, -conv_overlap_size:, :].clone()
                attn_out, state = attn.forward(
                    hidden_normed, attention_mask=None,
                    position_ids=torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0),
                    initial_state=None, output_final_state=True,
                )
                dn_states[layer_idx] = state
            else:
                # Full attention: compute and store prefix KV
                q_full = attn.q_proj(hidden_normed)[0]
                k_raw = attn.k_proj(hidden_normed)[0]
                v_raw = attn.v_proj(hidden_normed)[0]

                q_shape = (PREFIX_LEN, attn.num_heads_per_tp, attn.head_dim * 2)
                query, gate = torch.chunk(q_full.view(*q_shape), 2, dim=-1)
                query = query.view(1, PREFIX_LEN, attn.num_heads_per_tp, attn.head_dim)
                k = k_raw.view(1, PREFIX_LEN, attn.num_key_value_heads_per_tp, attn.head_dim)
                v = v_raw.view(1, PREFIX_LEN, attn.num_key_value_heads_per_tp, attn.head_dim)

                query = attn.q_norm(query)
                key = attn.k_norm(k)

                cos, sin = attn.rotary_emb(v, seq_len=PREFIX_LEN)
                cos, sin = cos[:, :cos.shape[1] // 2], sin[:, :sin.shape[1] // 2]

                if attn.rope_dim == attn.head_dim:
                    query = apply_rotary_emb(query, cos, sin, interleaved=False, inplace=False)
                    key = apply_rotary_emb(key, cos, sin, interleaved=False, inplace=False)
                else:
                    q_rot = query[:, :, :, :attn.rope_dim]
                    q_pass = query[:, :, :, attn.rope_dim:]
                    k_rot = key[:, :, :, :attn.rope_dim]
                    k_pass = key[:, :, :, attn.rope_dim:]
                    q_rot = apply_rotary_emb(q_rot, cos, sin, interleaved=False, inplace=False)
                    k_rot = apply_rotary_emb(k_rot, cos, sin, interleaved=False, inplace=False)
                    query = torch.cat([q_rot, q_pass], dim=-1)
                    key = torch.cat([k_rot, k_pass], dim=-1)

                nkv_groups = attn.num_key_value_groups
                if nkv_groups > 1:
                    key = key.unsqueeze(3).expand(-1, -1, -1, nkv_groups, -1).reshape(
                        1, PREFIX_LEN, attn.num_heads_per_tp, attn.head_dim)
                    v_exp = v.unsqueeze(3).expand(-1, -1, -1, nkv_groups, -1).reshape(
                        1, PREFIX_LEN, attn.num_heads_per_tp, attn.head_dim)
                else:
                    v_exp = v

                kv_store[layer_idx] = (key, v_exp)

                mask_val = torch.finfo(hidden_normed.dtype).min
                cm = torch.full((PREFIX_LEN, PREFIX_LEN), mask_val, device=device)
                cm.masked_fill_(torch.arange(PREFIX_LEN, device=device) <
                    (torch.arange(PREFIX_LEN, device=device) + 1).view(PREFIX_LEN, 1), 0)
                causal_mask = cm.to(hidden_normed.dtype)[None, None, :].expand(1, -1, -1, -1)
                attn_out = attn.forward(hidden_normed, attention_mask=causal_mask,
                    position_ids=torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0))

            hidden_prefix = residual + attn_out
            residual = hidden_prefix
            hidden_normed = layer.post_attention_layernorm(hidden_prefix)
            mlp_out = layer.mlp(hidden_normed)
            hidden_prefix = residual + mlp_out

    # ===== Suffix pass (with_grad) =====
    suffix_input_ids = torch.tensor(suffix_ids_lists, dtype=torch.long, device=device)
    suffix_embeds = model.model.embed_tokens(suffix_input_ids)
    hidden_suffix = suffix_embeds
    suffix_pos_ids = torch.arange(PREFIX_LEN, PREFIX_LEN + SUFFIX_LEN, dtype=torch.long, device=device).unsqueeze(0).expand(N, -1)

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        attn = layer.self_attn
        residual = hidden_suffix
        hidden_normed = layer.input_layernorm(hidden_suffix)

        if is_deltanet(layer_idx):
            state_exp = dn_states[layer_idx].expand(N, -1, -1, -1).contiguous()
            overlap_exp = dn_overlaps[layer_idx].expand(N, -1, -1).contiguous()
            attn_out = attn.forward(
                hidden_normed, attention_mask=None, position_ids=suffix_pos_ids,
                initial_state=state_exp, output_final_state=False,
                conv_overlap_hidden=overlap_exp,
            )
        else:
            # Full attention with prefix KV expansion
            pk, pv = kv_store[layer_idx]
            q_full = attn.q_proj(hidden_normed)[0]
            k_raw = attn.k_proj(hidden_normed)[0]
            v_raw = attn.v_proj(hidden_normed)[0]

            q_shape = (N, SUFFIX_LEN, attn.num_heads_per_tp, attn.head_dim * 2)
            query, gate = torch.chunk(q_full.view(*q_shape), 2, dim=-1)
            query = query.view(N, SUFFIX_LEN, attn.num_heads_per_tp, attn.head_dim)
            k = k_raw.view(N, SUFFIX_LEN, attn.num_key_value_heads_per_tp, attn.head_dim)
            v = v_raw.view(N, SUFFIX_LEN, attn.num_key_value_heads_per_tp, attn.head_dim)

            query = attn.q_norm(query)
            key = attn.k_norm(k)

            cos, sin = attn.rotary_emb(v, seq_len=PREFIX_LEN + SUFFIX_LEN)
            cos = cos[PREFIX_LEN:PREFIX_LEN + SUFFIX_LEN, :cos.shape[1] // 2]
            sin = sin[PREFIX_LEN:PREFIX_LEN + SUFFIX_LEN, :sin.shape[1] // 2]

            if attn.rope_dim == attn.head_dim:
                query = apply_rotary_emb(query, cos, sin, interleaved=False, inplace=False)
                key = apply_rotary_emb(key, cos, sin, interleaved=False, inplace=False)
            else:
                q_rot = query[:, :, :, :attn.rope_dim]
                q_pass = query[:, :, :, attn.rope_dim:]
                k_rot = key[:, :, :, :attn.rope_dim]
                k_pass = key[:, :, :, attn.rope_dim:]
                q_rot = apply_rotary_emb(q_rot, cos, sin, interleaved=False, inplace=False)
                k_rot = apply_rotary_emb(k_rot, cos, sin, interleaved=False, inplace=False)
                query = torch.cat([q_rot, q_pass], dim=-1)
                key = torch.cat([k_rot, k_pass], dim=-1)

            nkv_groups = attn.num_key_value_groups
            if nkv_groups > 1:
                key = key.repeat_interleave(nkv_groups, dim=2)
                v = v.repeat_interleave(nkv_groups, dim=2)

            pk_exp = pk.expand(N, -1, -1, -1).contiguous()
            pv_exp = pv.expand(N, -1, -1, -1).contiguous()
            exp_key = torch.cat([pk_exp, key], dim=1)
            exp_value = torch.cat([pv_exp, v], dim=1)

            cu_q = torch.tensor([0] + [SUFFIX_LEN] * N, device=device).cumsum(0).to(torch.int32)
            cu_k = torch.tensor([0] + [PREFIX_LEN + SUFFIX_LEN] * N, device=device).cumsum(0).to(torch.int32)

            q_flat = query.reshape(N * SUFFIX_LEN, attn.num_heads_per_tp, attn.head_dim)
            k_flat = exp_key.reshape(N * (PREFIX_LEN + SUFFIX_LEN), attn.num_heads_per_tp, attn.head_dim)
            v_flat = exp_value.reshape(N * (PREFIX_LEN + SUFFIX_LEN), attn.num_heads_per_tp, attn.head_dim)

            attn_out = flash_attn_varlen_func(
                q_flat, k_flat, v_flat,
                cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                max_seqlen_q=SUFFIX_LEN, max_seqlen_k=PREFIX_LEN + SUFFIX_LEN,
                dropout_p=0.0, softmax_scale=attn.scaling, causal=True,
            )
            attn_out = attn_out.to(q_flat.dtype)
            attn_out = attn_out.reshape(N, SUFFIX_LEN, attn.num_heads_per_tp * attn.head_dim)

            if attn.attn_output_gate:
                gate = gate.reshape(N, SUFFIX_LEN, attn.num_heads_per_tp * attn.head_dim)
                attn_out = attn_out * torch.sigmoid(gate)

            attn_out = attn_out.reshape(N * SUFFIX_LEN, 1, attn.q_output_size_per_tp).contiguous()
            attn_out = attn.o_proj(attn_out)[0]
            attn_out = attn_out.reshape(N, SUFFIX_LEN, hidden_size)

        hidden_suffix = residual + attn_out
        residual = hidden_suffix
        hidden_normed = layer.post_attention_layernorm(hidden_suffix)
        mlp_out = layer.mlp(hidden_normed)
        hidden_suffix = residual + mlp_out

    hidden_suffix = model.model.norm(hidden_suffix)
    logits = model.lm_head(hidden_suffix)[0]
    logits = tp.gather_from_tensor_model_parallel_region(logits)

    # Free stored states before backward to reduce memory pressure
    del dn_states, dn_overlaps, kv_store
    torch.cuda.empty_cache()

    return logits.float()


# ====================================================================
# Phase 1: Precision validation
# ====================================================================
if local_rank == 0:
    print(f"\n{'='*60}")
    print("PHASE 1: PRECISION VALIDATION")
    print(f"{'='*60}")

torch.manual_seed(SEED)
suffix_ids_for_precision = [torch.randint(0, vocab_size, (SUFFIX_LEN,), device="cpu").tolist()
                             for _ in range(N_SEQUENCES)]
input_ids_normal = torch.tensor(
    [prefix_ids_list + s for s in suffix_ids_for_precision],
    dtype=torch.long, device=device)

with torch.no_grad():
    output_normal = model(
        input_ids=input_ids_normal,
        attention_mask=torch.ones((N_SEQUENCES, total_len), dtype=torch.long, device=device),
        position_ids=torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1))
    logits_normal = output_normal.logits

with torch.no_grad():
    logits_ps = manual_forward_on(prefix_ids_list, suffix_ids_for_precision)

normal_suffix = logits_normal[:, PREFIX_LEN:, :]
cos_sims = [F.cosine_similarity(logits_ps[i].float(), normal_suffix[i].float(), dim=-1).mean().item()
            for i in range(N_SEQUENCES)]
overall_cos = sum(cos_sims) / len(cos_sims)
precision_pass = overall_cos >= 0.999

if local_rank == 0:
    print(f"Precision: cos_sim={overall_cos:.6f} (threshold: 0.999)")
    print(f"Status: {'PASS' if precision_pass else 'FAIL'}")
    for i, cs in enumerate(cos_sims):
        print(f"  Seq {i}: cos_sim={cs:.6f}")

if not precision_pass:
    if local_rank == 0:
        print("PRECISION FAILED — aborting!")
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()
    sys.exit(1)

# ====================================================================
# Phase 2: Training throughput benchmark
# ====================================================================
if local_rank == 0:
    print(f"\n{'='*60}")
    print("PHASE 2: TRAINING THROUGHPUT BENCHMARK")
    print(f"{'='*60}")

model.train()
optimizer.zero_grad()

# --- A: PS OFF model.forward() (baseline) ---
if local_rank == 0:
    print("\n[A] PS OFF model.forward() — baseline (no Python loop overhead)")
times_off_model = []
for step in range(NUM_WARMUP + NUM_STEPS):
    torch.manual_seed(SEED + step)
    seqs = [prefix_ids_list + torch.randint(0, vocab_size, (SUFFIX_LEN,), device="cpu").tolist()
            for _ in range(N_SEQUENCES)]
    ids = torch.tensor(seqs, dtype=torch.long, device=device)

    torch.cuda.synchronize()
    t0 = time.time()
    output = model(ids,
        attention_mask=torch.ones(N_SEQUENCES, total_len, dtype=torch.long, device=device),
        position_ids=torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1))
    logits = output.logits
    shift_l = logits[:, PREFIX_LEN-1:PREFIX_LEN+SUFFIX_LEN-1, :]
    shift_lb = ids[:, PREFIX_LEN:PREFIX_LEN+SUFFIX_LEN]
    loss = F.cross_entropy(shift_l.reshape(-1, vocab_size), shift_lb.reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t1 = time.time()

    if step >= NUM_WARMUP:
        times_off_model.append(t1 - t0)
    if local_rank == 0 and step % 2 == 0:
        print(f"  step {step}: {t1-t0:.3f}s loss={loss.item():.4f} mem={torch.cuda.memory_allocated()/1024**3:.2f}GB")

avg_off_model = sum(times_off_model) / len(times_off_model)

# Free memory before next section
del optimizer
torch.cuda.empty_cache()
gc.collect()

# --- B: PS OFF manual forward (fair baseline) ---
if local_rank == 0:
    print("\n[B] PS OFF manual forward — fair baseline (same overhead as PS ON)")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.01)

times_off_manual = []
for step in range(NUM_WARMUP + NUM_STEPS):
    torch.manual_seed(SEED + step)
    seqs = [prefix_ids_list + torch.randint(0, vocab_size, (SUFFIX_LEN,), device="cpu").tolist()
            for _ in range(N_SEQUENCES)]
    ids = torch.tensor(seqs, dtype=torch.long, device=device)

    torch.cuda.synchronize()
    t0 = time.time()
    logits = manual_forward_off(ids)
    shift_l = logits[:, PREFIX_LEN-1:PREFIX_LEN+SUFFIX_LEN-1, :]
    shift_lb = ids[:, PREFIX_LEN:PREFIX_LEN+SUFFIX_LEN]
    loss = F.cross_entropy(shift_l.reshape(-1, vocab_size), shift_lb.reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t1 = time.time()

    if step >= NUM_WARMUP:
        times_off_manual.append(t1 - t0)
    if local_rank == 0 and step % 2 == 0:
        print(f"  step {step}: {t1-t0:.3f}s loss={loss.item():.4f} mem={torch.cuda.memory_allocated()/1024**3:.2f}GB")

avg_off_manual = sum(times_off_manual) / len(times_off_manual)

# Free memory before next section
del optimizer
torch.cuda.empty_cache()
gc.collect()

# --- C: PS ON manual two-pass forward ---
if local_rank == 0:
    print("\n[C] PS ON manual two-pass — PS optimization (same overhead as PS OFF manual)")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.01)
optimizer.zero_grad()

times_on_manual = []
for step in range(NUM_WARMUP + NUM_STEPS):
    torch.manual_seed(SEED + step)
    suffix_lists = [torch.randint(0, vocab_size, (SUFFIX_LEN,), device="cpu").tolist()
                    for _ in range(N_SEQUENCES)]

    torch.cuda.synchronize()
    t0 = time.time()
    logits = manual_forward_on(prefix_ids_list, suffix_lists)
    # PS ON logits: suffix-only (N, SUFFIX_LEN, vocab)
    shift_l = logits[:, :-1, :]
    shift_lb = torch.tensor(suffix_lists, dtype=torch.long, device=device)[:, 1:]
    loss = F.cross_entropy(shift_l.reshape(-1, vocab_size), shift_lb.reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    t1 = time.time()

    if step >= NUM_WARMUP:
        times_on_manual.append(t1 - t0)
    if local_rank == 0 and step % 2 == 0:
        print(f"  step {step}: {t1-t0:.3f}s loss={loss.item():.4f} mem={torch.cuda.memory_allocated()/1024**3:.2f}GB")

avg_on_manual = sum(times_on_manual) / len(times_on_manual)

# ====================================================================
# Phase 3: Results
# ====================================================================
token_savings = (N_SEQUENCES - 1) / N_SEQUENCES * PREFIX_LEN / total_len
theoretical = 1 / (1 - token_savings)

if local_rank == 0:
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Model: Qwen3.6-27B-16layers (TP={TP_SIZE}, bf16)")
    print(f"Pattern: GRPO n={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"Token savings: {token_savings:.1%}  |  Theoretical speedup: {theoretical:.2f}x")
    print(f"{'='*60}")
    print(f"  [A] PS OFF model.forward():  {avg_off_model:.3f}s/step")
    print(f"  [B] PS OFF manual forward:   {avg_off_manual:.3f}s/step")
    print(f"  [C] PS ON manual two-pass:   {avg_on_manual:.3f}s/step")
    print(f"{'='*60}")
    overhead = avg_off_manual / avg_off_model
    unfair = avg_off_model / avg_on_manual
    fair = avg_off_manual / avg_on_manual
    print(f"  Manual overhead: {overhead:.2f}x (B vs A)")
    print(f"  Unfair speedup:  {unfair:.2f}x (A vs C) — not apples-to-apples")
    print(f"  FAIR speedup:    {fair:.2f}x (B vs C) — same overhead method!")
    print(f"{'='*60}")
    print(f"Precision: cos_sim={overall_cos:.6f}")
    print(f"{'='*60}")
    if fair > 1.0:
        print(f"\nPS provides {fair:.2f}x training speedup with same overhead method.")
    else:
        print(f"\nPS does not show positive speedup in this manual benchmark.")
        print(f"This may be because inline attention computation (for full attn layers)")
        print(f"adds overhead that outweighs token savings. The verl monkey-patch approach")
        print(f"(model.forward() for both OFF and ON) should show better speedup.")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()