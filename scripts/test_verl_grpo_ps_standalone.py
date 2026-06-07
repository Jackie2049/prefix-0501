#!/usr/bin/env python3
"""Standalone verl GRPO training test: uses verl's build_prefix_sharing_micro_batch + MegatronPPOActor forward flow.

This validates the complete PS integration pipeline:
1. build_prefix_sharing_micro_batch → plan + trim batch + extract prefix tokens
2. Prefix pass (no_grad) → populate KV + DeltaNet stores
3. Suffix pass (with grad) → inject states → compute logits
4. PPO clipped loss → backward → optimizer step

Usage: torchrun --nproc_per_node=4 scripts/test_verl_grpo_ps_standalone.py
"""
import os, sys, time
verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_sharing_path = os.path.expanduser("~/rollout-prefix/prefix-0501/prefix-sharing")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path); sys.path.insert(0, prefix_sharing_path); sys.path.insert(0, prefix_path)

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from safetensors.torch import load_file

HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only-16layers")
TP_SIZE = 4; PREFIX_LEN = 64; SUFFIX_LEN = 128; N_SEQUENCES = 4; SEED = 42; N_STEPS = 3

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

from megatron.core import ModelParallelConfig
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLMRmPad

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    sequence_parallel=False, bf16=True, params_dtype=torch.bfloat16,
)
model = ParallelQwen3_6ForCausalLMRmPad(config, megatron_config).to(device)

# Load pretrained weights
hf_state_dict = {}
for i in range(1, 12):
    shard_path = os.path.join(HF_MODEL_PATH, f"model.safetensors-{i:05d}-of-00011.safetensors")
    shard_dict = load_file(shard_path); hf_state_dict.update(shard_dict)

hf_keys_filtered = {k: v for k, v in hf_state_dict.items()
                    if k.startswith("model.language_model.") or k.startswith("lm_head.")}

layer_types = config.layer_types; num_layers = config.num_hidden_layers; vocab_size = config.vocab_size

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
for layer_idx in range(num_layers):
    is_deltanet = (layer_types[layer_idx] == "linear_attention")
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

ps_config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=PREFIX_LEN, min_group_size=N_SEQUENCES)
integration = VerlQwen3_6Integration(config=ps_config)
patch_handle = integration.install(model_config=config)

from prefix_sharing.integrations.context import prefix_sharing_runtime_context

clip_ratio = 0.2
total_len = PREFIX_LEN + SUFFIX_LEN
response_len = SUFFIX_LEN - 1

# Simulate actor config (dict-based, as verl uses)
actor_config = {
    "prefix_sharing_config": {
        "enable_prefix_sharing": True,
        "min_prefix_len": PREFIX_LEN,
        "min_group_size": N_SEQUENCES,
    },
    "megatron": {
        "use_remove_padding": True,
    },
    "clip_ratio": clip_ratio,
}

if local_rank == 0:
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Loaded {loaded_count} weights, GPU {mem:.2f} GB")
    print(f"\n{'='*70}")
    print(f"STANDALONE VERL GRPO PS TRAINING TEST")
    print(f"Model: 16-layer ~7B, TP={TP_SIZE}, P={PREFIX_LEN}, S={SUFFIX_LEN}, N={N_SEQUENCES}")
    print(f"Using build_prefix_sharing_micro_batch + verl forward_fn")
    print(f"{'='*70}")
    print(f"{'Step':>4} {'Loss':>8} {'Cos_sim':>10} {'Time':>8} {'Peak_mem':>10}")
    print(f"{'-'*4} {'-'*8} {'-'*10} {'-'*8} {'-'*10}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
all_results = []

for step in range(N_STEPS):
    torch.manual_seed(SEED + step * 100)
    torch.cuda.synchronize(); t_start = time.time()
    torch.cuda.reset_peak_memory_stats()

    # Create verl-style micro-batch
    prefix_tokens = torch.randint(0, vocab_size, (PREFIX_LEN,), dtype=torch.long, device=device)
    suffix_tokens_list = [torch.randint(0, vocab_size, (SUFFIX_LEN,), dtype=torch.long, device=device)
                          for _ in range(N_SEQUENCES)]
    full_input_ids = torch.stack([torch.cat([prefix_tokens, s]) for s in suffix_tokens_list])

    # Create attention_mask (all True for full sequences)
    attention_mask = torch.ones(N_SEQUENCES, total_len, dtype=torch.bool, device=device)
    position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

    # Create responses field (suffix tokens, shifted for labels)
    responses = torch.stack([s for s in suffix_tokens_list])  # (N, SUFFIX_LEN)

    # Construct batch dict (verl format)
    batch = {
        "input_ids": full_input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
    }

    # Step 1: build_prefix_sharing_micro_batch (verl's PS entry point)
    batch, ps_runtime_state = build_prefix_sharing_micro_batch(
        batch, actor_config, config, model_spec=ModelSpec.from_hf_config(config),
    )

    if ps_runtime_state is None:
        if local_rank == 0:
            print(f"Step {step}: PS not activated (no sharing found)")
        # Fall through to normal forward
        pass

    # Step 2: Reference forward (no grad, compute old_log_probs from full sequence)
    # This is equivalent to the "reference model forward" in verl
    with torch.no_grad():
        ref_full_input_ids = torch.stack([torch.cat([prefix_tokens, s]) for s in suffix_tokens_list])
        ref_attention_mask = torch.ones(N_SEQUENCES, total_len, dtype=torch.bool, device=device)
        ref_position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)
        ref_output = model(input_ids=ref_full_input_ids, attention_mask=ref_attention_mask,
                          position_ids=ref_position_ids)
        suffix_logits_ref = ref_output.logits[:, PREFIX_LEN:, :]
        suffix_labels = torch.stack([s[1:] for s in suffix_tokens_list])
        suffix_log_probs_ref = F.log_softmax(suffix_logits_ref.float(), dim=-1)
        old_log_prob = suffix_log_probs_ref[:, :-1, :].gather(
            dim=-1, index=suffix_labels.unsqueeze(-1)).squeeze(-1).to(device='cpu', dtype=torch.float32)
    del ref_output, suffix_logits_ref, suffix_log_probs_ref
    del ref_full_input_ids, ref_attention_mask, ref_position_ids
    torch.cuda.empty_cache()

    # Step 3: PS two-pass forward (with grad) — mimics megatron_actor.py
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].to(bool)
    position_ids = batch["position_ids"]

    from contextlib import nullcontext
    prefix_context = prefix_sharing_runtime_context or nullcontext

    with prefix_context(ps_runtime_state) as ps_ctx:
        # Two-pass: prefix forward first if prefix tokens available
        if ps_ctx is not None and ps_runtime_state is not None \
                and hasattr(ps_runtime_state, 'prefix_input_ids') \
                and ps_runtime_state.prefix_input_ids is not None:
            _prefix_ids = ps_runtime_state.prefix_input_ids
            _prefix_mask = ps_runtime_state.prefix_attention_mask
            _prefix_pos = ps_runtime_state.prefix_position_ids

            with torch.no_grad():
                prefix_output = model(input_ids=_prefix_ids,
                                     attention_mask=_prefix_mask,
                                     position_ids=_prefix_pos)
            del prefix_output

        # Suffix pass (with grad)
        suffix_output = model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids)
        suffix_logits_ps = suffix_output.logits

    # Determine how many suffix positions we have
    suffix_log_probs = F.log_softmax(suffix_logits_ps.float(), dim=-1)
    new_log_prob = suffix_log_probs[:, :-1, :].gather(
        dim=-1, index=suffix_labels.unsqueeze(-1)).squeeze(-1)
    del suffix_log_probs

    cos_sim = F.cosine_similarity(
        new_log_prob.float().flatten(), old_log_prob.to(device=device).float().flatten(), dim=0).item()

    # GRPO advantage + PPO clipped loss
    torch.manual_seed(SEED + step * 200)
    rewards = torch.randn(N_SEQUENCES, dtype=torch.float32, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    advantages_per_pos = advantages.unsqueeze(1).expand(N_SEQUENCES, response_len)

    old_log_prob_gpu = old_log_prob.to(device=device)
    ratio = torch.exp(new_log_prob - old_log_prob_gpu)
    clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
    policy_loss = -torch.min(ratio * advantages_per_pos, clipped_ratio * advantages_per_pos).mean()

    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step(); optimizer.zero_grad()

    mem_peak = torch.cuda.max_memory_allocated() / 1024**3
    torch.cuda.synchronize(); t_total = time.time() - t_start

    if local_rank == 0:
        print(f"{step:>4} {policy_loss.item():>8.4f} {cos_sim:>10.6f} {t_total:>8.3f}s {mem_peak:>8.2f}GB")

    all_results.append((step, policy_loss.item(), cos_sim, t_total, mem_peak))

    del output, suffix_logits_ps, new_log_prob, old_log_prob, old_log_prob_gpu
    del batch, input_ids, attention_mask, position_ids
    torch.cuda.empty_cache()

if local_rank == 0:
    print(f"\n{'='*70}")
    print(f"STANDALONE VERL GRPO PS TRAINING RESULTS")
    print(f"{'='*70}")
    print(f"{'Step':>4} {'Loss':>8} {'Cos_sim':>10} {'Time':>8} {'Peak_mem':>10}")
    for step, loss, cos, tt, mm in all_results:
        print(f"{step:>4} {loss:>8.4f} {cos:>10.6f} {tt:>8.3f}s {mm:>8.2f}GB")
    print(f"{'='*70}")

    min_cos = min(cos for _, _, cos, _, _ in all_results)
    avg_time = sum(t for _, _, _, t, _ in all_results) / N_STEPS
    max_mem = max(m for _, _, _, _, m in all_results)

    if min_cos >= 0.99:
        print(f"PASS: cos_sim >= 0.99 (min={min_cos:.6f})")
    else:
        print(f"FAIL: cos_sim < 0.99 (min={min_cos:.6f})")
    print(f"Peak memory: {max_mem:.2f}GB, Avg time: {avg_time:.3f}s")
    print(f"PS pipeline: build_prefix_sharing_micro_batch → prefix pass → suffix pass → backward → optimizer")

parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()