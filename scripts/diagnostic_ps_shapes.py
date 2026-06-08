#!/usr/bin/env python3
"""Diagnostic: show logits shapes and token counts for PS ON vs PS OFF.

Quick test to understand the batch format and logits shape differences.
"""
import os, sys
verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_sharing_path = os.path.expanduser("~/rollout-prefix/prefix-0501/prefix-sharing")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path); sys.path.insert(0, prefix_sharing_path); sys.path.insert(0, prefix_path)

import torch
from transformers import AutoConfig
from safetensors.torch import load_file

HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only-16layers")
TP_SIZE = 4; SEED = 42

torch.distributed.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(local_rank)

from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker, get_cuda_rng_tracker, _MODEL_PARALLEL_RNG_TRACKER_NAME,
)
initialize_rng_tracker(use_te_rng_tracker=False, inference_rng_tracker=False)
rng_tracker = get_cuda_rng_tracker(); rng_tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed=SEED)

from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    context_parallel_size=1, expert_model_parallel_size=1,
)
tp_rank = parallel_state.get_tensor_model_parallel_rank()
device = torch.device(f"cuda:{local_rank}")

config = AutoConfig.from_pretrained(HF_MODEL_PATH)
vocab_size = config.vocab_size

from megatron.core import ModelParallelConfig
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLMRmPad

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    sequence_parallel=False, bf16=True, params_dtype=torch.bfloat16,
)
model = ParallelQwen3_6ForCausalLMRmPad(config, megatron_config).to(device)

# Load weights (minimal)
hf_state_dict = {}
for i in range(1, 12):
    shard_path = os.path.join(HF_MODEL_PATH, f"model.safetensors-{i:05d}-of-00011.safetensors")
    shard_dict = load_file(shard_path); hf_state_dict.update(shard_dict)

hf_keys_filtered = {k: v for k, v in hf_state_dict.items()
                    if k.startswith("model.language_model.") or k.startswith("lm_head.")}

def shard_tensor(t, dim, tp_size, tp_rank):
    return torch.chunk(t, tp_size, dim=dim)[tp_rank].contiguous()

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

loaded_count = 0
for layer_idx in range(config.num_hidden_layers):
    is_deltanet = (config.layer_types[layer_idx] == "linear_attention")
    decoder_layer = model.model.layers[layer_idx]
    attn_module = decoder_layer.self_attn; mlp_module = decoder_layer.mlp
    hf_layer_prefix = f"model.language_model.layers.{layer_idx}"
    gate_key = f"{hf_layer_prefix}.mlp.gate_proj.weight"
    up_key = f"{hf_layer_prefix}.mlp.up_proj.weight"
    if gate_key in hf_keys_filtered and up_key in hf_keys_filtered:
        gate_up_shard = torch.cat([
            shard_tensor(hf_keys_filtered[gate_key], 0, TP_SIZE, tp_rank),
            shard_tensor(hf_keys_filtered[up_key], 0, TP_SIZE, tp_rank),
        ], dim=0).contiguous()
        mlp_module.gate_up_proj.weight.data.copy_(gate_up_shard.to(torch.bfloat16))
        loaded_count += 2
    down_key = f"{hf_layer_prefix}.mlp.down_proj.weight"
    if down_key in hf_keys_filtered:
        mlp_module.down_proj.weight.data.copy_(
            shard_tensor(hf_keys_filtered[down_key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
        loaded_count += 1
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
                getattr(attn_module, proj_name).weight.data.copy_(
                    shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
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
                getattr(attn_module, proj_name).weight.data.copy_(
                    shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
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
    for ln_name in ["input_layernorm", "post_attention_layernorm"]:
        key = f"{hf_layer_prefix}.{ln_name}.weight"
        if key in hf_keys_filtered:
            getattr(decoder_layer, ln_name).weight.data.copy_(hf_keys_filtered[key].to(torch.bfloat16))
            loaded_count += 1

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

hf_state_dict.clear(); hf_keys_filtered.clear(); torch.cuda.empty_cache()

# Install PS patches
from prefix_sharing.integrations.verl_attention import VerlQwen3_6Integration
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.model_spec import ModelSpec
from prefix_sharing.integrations.verl_mcore import build_prefix_sharing_micro_batch
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from contextlib import nullcontext

ps_config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=1, min_group_size=2)
patch_handle = VerlQwen3_6Integration(config=ps_config).install(model_config=config)

actor_config = {
    "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 1, "min_group_size": 2},
    "megatron": {"use_remove_padding": True},
}

# Test different P values
for P_LEN, S_LEN in [(64, 128), (128, 128), (256, 128)]:
    N_SEQ = 4
    total_len = P_LEN + S_LEN

    torch.manual_seed(SEED)
    prefix_tokens = torch.randint(0, vocab_size, (P_LEN,), dtype=torch.long, device=device)
    suffix_tokens_list = [torch.randint(0, vocab_size, (S_LEN,), dtype=torch.long, device=device)
                          for _ in range(N_SEQ)]
    full_input_ids = torch.stack([torch.cat([prefix_tokens, s]) for s in suffix_tokens_list])
    attention_mask = torch.ones(N_SEQ, total_len, dtype=torch.bool, device=device)
    position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N_SEQ, -1)

    with torch.no_grad():
        # PS OFF
        output_off = model(input_ids=full_input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits_off_shape = output_off.logits.shape

        # PS ON
        batch = {
            "input_ids": full_input_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "position_ids": position_ids.clone(),
            "responses": torch.stack([s for s in suffix_tokens_list]),
        }
        ps_config_test = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=P_LEN, min_group_size=N_SEQ)
        actor_config_test = {
            "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": P_LEN, "min_group_size": N_SEQ},
            "megatron": {"use_remove_padding": True},
        }
        batch, ps_runtime_state = build_prefix_sharing_micro_batch(
            batch, actor_config_test, config, model_spec=ModelSpec.from_hf_config(config),
        )

        if ps_runtime_state is not None:
            # Show batch format
            batch_ids_shape = batch["input_ids"].shape
            batch_mask = batch["attention_mask"].to(bool)
            valid_tokens_per_row = [batch_mask[row].sum().item() for row in range(N_SEQ)]
            total_valid = sum(valid_tokens_per_row)
            prefix_ids_shape = ps_runtime_state.prefix_input_ids.shape if ps_runtime_state.prefix_input_ids is not None else None

            prefix_context = prefix_sharing_runtime_context or nullcontext
            with prefix_context(ps_runtime_state) as ps_ctx:
                if ps_ctx is not None and ps_runtime_state.prefix_input_ids is not None:
                    _prefix_output = model(
                        input_ids=ps_runtime_state.prefix_input_ids,
                        attention_mask=ps_runtime_state.prefix_attention_mask,
                        position_ids=ps_runtime_state.prefix_position_ids,
                    )
                    prefix_logits_shape = _prefix_output.logits.shape
                    del _prefix_output

                suffix_output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch_mask,
                    position_ids=batch["position_ids"],
                )
                logits_on_shape = suffix_output.logits.shape

            if local_rank == 0:
                print(f"\n=== P={P_LEN}, S={S_LEN}, N={N_SEQ} ===")
                print(f"  PS OFF logits: {logits_off_shape}")
                print(f"  PS ON batch input_ids: {batch_ids_shape}")
                print(f"  PS ON valid tokens per row: {valid_tokens_per_row} (total={total_valid})")
                print(f"  PS ON prefix_input_ids: {prefix_ids_shape}")
                if prefix_logits_shape:
                    print(f"  PS ON prefix logits: {prefix_logits_shape}")
                print(f"  PS ON suffix logits: {logits_on_shape}")
                print(f"  PS OFF: {N_SEQ * total_len} tokens forward")
                print(f"  PS ON: {P_LEN} prefix + {total_valid} suffix tokens forward")
                print(f"  Token savings: {(1 - (P_LEN + total_valid)/(N_SEQ * total_len))*100:.1f}%")
        else:
            if local_rank == 0:
                print(f"P={P_LEN}, S={S_LEN}: PS not activated")

    del output_off, full_input_ids, attention_mask, position_ids
    torch.cuda.empty_cache()

patch_handle.disable() if patch_handle else None
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()