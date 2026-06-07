#!/usr/bin/env python3
"""Full standalone GRPO training loop with PS (no Ray required).

Simulates the verl GRPO pipeline:
1. Multiple prompt groups (each with N=8 responses)
2. Full training loop: forward → log_probs → advantages → PPO loss → backward → optimizer
3. Compares PS-enabled vs normal training
4. Measures throughput and precision alignment

Usage: torchrun --nproc_per_node=4 scripts/test_grpo_ps_full_training.py
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
TP_SIZE = 4; PREFIX_LEN = 64; SUFFIX_LEN = 128; N_SEQUENCES = 4; SEED = 42; N_GROUPS = 2; N_STEPS = 5

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

# Load pretrained weights (compact inline — same as other scripts)
hf_state_dict = {}
for i in range(1, 12):
    shard_path = os.path.join(HF_MODEL_PATH, f"model.safetensors-{i:05d}-of-00011.safetensors")
    shard_dict = load_file(shard_path); hf_state_dict.update(shard_dict)

hf_keys_filtered = {k: v for k, v in hf_state_dict.items()
                    if k.startswith("model.language_model.") or k.startswith("lm_head.")}

layer_types = config.layer_types; num_layers = config.num_hidden_layers

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
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.model_spec import ModelSpec

ps_config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=PREFIX_LEN, min_group_size=N_SEQUENCES)
integration = VerlQwen3_6Integration(config=ps_config)
patch_handle = integration.install(model_config=config)

from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
from prefix_sharing.backends.packed_layout import PackedBatchLayout

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

vocab_size = config.vocab_size
total_len = PREFIX_LEN + SUFFIX_LEN
response_len = SUFFIX_LEN - 1
clip_ratio = 0.2

if local_rank == 0:
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"Loaded {loaded_count} weights, GPU {mem:.2f} GB")
    print(f"\n{'='*70}")
    print(f"FULL GRPO PS TRAINING LOOP")
    print(f"Model: 16-layer ~7B, TP={TP_SIZE}, P={PREFIX_LEN}, S={SUFFIX_LEN}")
    print(f"N_sequences={N_SEQUENCES}, N_groups={N_GROUPS}, N_steps={N_STEPS}")
    print(f"Optimizer: AdamW lr=1e-5 wd=0.01, Clip={clip_ratio}")
    print(f"{'='*70}")

all_results = []
for step in range(N_STEPS):
    torch.cuda.synchronize(); step_start = time.time()

    step_loss = 0.0
    step_cos_sims = []

    for group in range(N_GROUPS):
        # Each group represents a different prompt with N=8 responses
        torch.manual_seed(SEED + step * 100 + group * 10)

        prefix_tokens = torch.randint(0, vocab_size, (PREFIX_LEN,), dtype=torch.long, device=device)
        suffix_tokens_list = [torch.randint(0, vocab_size, (SUFFIX_LEN,), dtype=torch.long, device=device)
                              for _ in range(N_SEQUENCES)]
        full_sequences = [torch.cat([prefix_tokens, suffix]) for suffix in suffix_tokens_list]
        full_input_ids = torch.stack(full_sequences)
        suffix_input_ids = torch.stack(suffix_tokens_list)
        prefix_input_ids = prefix_tokens.unsqueeze(0)
        suffix_labels = suffix_input_ids[:, 1:]  # (N, SUFFIX_LEN-1)

        # Normal forward (reference, no grad) — compute only selected log_probs
        full_attention_mask = torch.ones(N_SEQUENCES, total_len, dtype=torch.bool, device=device)
        full_position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

        with torch.no_grad():
            output_normal = model(input_ids=full_input_ids, attention_mask=full_attention_mask,
                                 position_ids=full_position_ids)
            # Compute log_probs only for suffix positions (not the full vocab tensor)
            suffix_logits_normal = output_normal.logits[:, PREFIX_LEN:, :].clone()
            # Gather selected log_probs for suffix labels
            suffix_log_probs_normal = F.log_softmax(suffix_logits_normal.float(), dim=-1)
            old_log_prob_selected = suffix_log_probs_normal[:, :-1, :].gather(
                dim=-1, index=suffix_labels.unsqueeze(-1)
            ).squeeze(-1).to(device='cpu', dtype=torch.float32)
        del output_normal, suffix_logits_normal, suffix_log_probs_normal; torch.cuda.empty_cache()

        # PS two-pass forward (with gradients)
        sequences_list = [seq.tolist() for seq in full_input_ids]
        ps_plan = PrefixSharingPlanner(ps_config).plan(sequences_list)

        prefix_attention_mask = torch.ones(1, PREFIX_LEN, dtype=torch.bool, device=device)
        prefix_position_ids = torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0)
        suffix_attention_mask = torch.ones(N_SEQUENCES, SUFFIX_LEN, dtype=torch.bool, device=device)
        suffix_position_ids = torch.arange(PREFIX_LEN, total_len, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

        suffix_lengths = [SUFFIX_LEN] * N_SEQUENCES
        suffix_packed_layout = PackedBatchLayout.from_valid_lengths(suffix_lengths)

        ps_runtime_state = PrefixSharingRuntimeState(
            prefix_sharing_plan=ps_plan, backend=None,
            packed_batch_layout=suffix_packed_layout,
            model_spec=ModelSpec.from_hf_config(config),
            prefix_input_ids=prefix_input_ids,
            prefix_attention_mask=prefix_attention_mask,
            prefix_position_ids=prefix_position_ids,
        )

        with prefix_sharing_runtime_context(ps_runtime_state) as ctx_state:
            with torch.no_grad():
                prefix_output = model(input_ids=ps_runtime_state.prefix_input_ids,
                                     attention_mask=ps_runtime_state.prefix_attention_mask,
                                     position_ids=ps_runtime_state.prefix_position_ids)
            del prefix_output

            suffix_output = model(input_ids=suffix_input_ids,
                                 attention_mask=suffix_attention_mask,
                                 position_ids=suffix_position_ids)
            logits_ps = suffix_output.logits

        # Compute log_probs and precision check
        log_probs_ps_all = F.log_softmax(logits_ps.float(), dim=-1)
        new_log_prob_selected = log_probs_ps_all[:, :-1, :].gather(
            dim=-1, index=suffix_labels.unsqueeze(-1)
        ).squeeze(-1)
        del log_probs_ps_all

        old_log_prob_gpu = old_log_prob_selected.to(device=device)
        cos_sim = F.cosine_similarity(
            new_log_prob_selected.float().flatten(),
            old_log_prob_gpu.float().flatten(),
            dim=0,
        ).item()
        step_cos_sims.append(cos_sim)

        # GRPO advantage
        torch.manual_seed(SEED + step * 200 + group * 20)
        rewards = torch.randn(N_SEQUENCES, dtype=torch.float32, device=device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        advantages_per_pos = advantages.unsqueeze(1).expand(N_SEQUENCES, response_len)

        # PPO clipped policy loss
        ratio = torch.exp(new_log_prob_selected - old_log_prob_gpu)
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -torch.min(ratio * advantages_per_pos, clipped_ratio * advantages_per_pos).mean()

        step_loss += policy_loss.item()

        # Backward
        policy_loss.backward()

        del old_log_prob_selected, old_log_prob_gpu, new_log_prob_selected, logits_ps, suffix_output
        del full_input_ids, suffix_input_ids, prefix_input_ids
        torch.cuda.empty_cache()

    # Average loss across groups
    avg_loss = step_loss / N_GROUPS
    avg_cos_sim = sum(step_cos_sims) / len(step_cos_sims)

    # Gradient clip + optimizer step
    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize(); step_time = time.time() - step_start

    if local_rank == 0:
        print(f"Step {step}: avg_loss={avg_loss:.4f}, avg_cos_sim={avg_cos_sim:.6f}, "
              f"grad_norm={total_grad_norm:.4f}, time={step_time:.2f}s "
              f"({N_GROUPS} groups × {N_SEQUENCES} seqs)")

    all_results.append((step, avg_loss, avg_cos_sim, total_grad_norm, step_time))

patch_handle.disable()

if local_rank == 0:
    print(f"\n{'='*70}")
    print(f"FULL GRPO PS TRAINING RESULTS")
    print(f"{'='*70}")
    print(f"{'Step':>4} {'Loss':>8} {'Cos_sim':>10} {'Grad_norm':>10} {'Time':>8}")
    for step, loss, cos, gn, tt in all_results:
        print(f"{step:>4} {loss:>8.4f} {cos:>10.6f} {gn:>10.4f} {tt:>8.2f}s")
    print(f"{'='*70}")

    min_cos = min(cos for _, _, cos, _, _ in all_results)
    if min_cos >= 0.99:
        print(f"PASS: All steps cos_sim >= 0.99 (min={min_cos:.6f})")
    else:
        print(f"FAIL: cos_sim < 0.99 (min={min_cos:.6f})")

    total_time = sum(tt for _, _, _, _, tt in all_results)
    total_tokens = N_STEPS * N_GROUPS * N_SEQUENCES * total_len
    throughput = total_tokens / total_time
    print(f"Throughput: {throughput:.0f} tokens/s (total {total_tokens} tok in {total_time:.1f}s)")

parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()