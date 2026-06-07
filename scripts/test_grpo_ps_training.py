#!/usr/bin/env python3
"""Minimal GRPO training simulation with prefix-sharing.

Validates the end-to-end two-pass PS integration in a simplified GRPO scenario:
1. N=16 sequences sharing a common prefix (GRPO n=16 where PS crossover achieved)
2. Two-pass PS forward: prefix pass → suffix pass
3. Compute log_probs from logits
4. Compute GRPO group-normalized advantage
5. Compute PPO clipped policy loss
6. Backward pass to verify gradient flow through PS patches

This tests the full training pipeline without the verl/Ray infrastructure.

Usage: torchrun --nproc_per_node=4 scripts/test_grpo_ps_training.py
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

HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
TP_SIZE = 4; PREFIX_LEN = 64; SUFFIX_LEN = 32; N_SEQUENCES = 8; SEED = 42

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

# Load pretrained weights (compact inline)
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
if local_rank == 0:
    print(f"Loaded {loaded_count} weights, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GB")

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

vocab_size = config.vocab_size
total_len = PREFIX_LEN + SUFFIX_LEN

# Create GRPO-like sequences: N=16 sequences sharing same prefix, different suffixes
torch.manual_seed(SEED + 500)
prefix_tokens = torch.randint(0, vocab_size, (PREFIX_LEN,), dtype=torch.long, device=device)
suffix_tokens_list = [torch.randint(0, vocab_size, (SUFFIX_LEN,), dtype=torch.long, device=device)
                      for _ in range(N_SEQUENCES)]

full_sequences = [torch.cat([prefix_tokens, suffix]) for suffix in suffix_tokens_list]
full_input_ids = torch.stack(full_sequences)
suffix_input_ids = torch.stack(suffix_tokens_list)
prefix_input_ids = prefix_tokens.unsqueeze(0)

if local_rank == 0:
    print(f"\n=== GRPO PS Training Simulation (N={N_SEQUENCES}, P={PREFIX_LEN}, S={SUFFIX_LEN}) ===")
    print(f"Total tokens: Normal={N_SEQUENCES*total_len}, PS={PREFIX_LEN+N_SEQUENCES*SUFFIX_LEN}")

# ===== Step 1: Normal forward (reference logits + log_probs) =====
if local_rank == 0:
    print("Step 1: Normal forward (reference)")

full_attention_mask = torch.ones(N_SEQUENCES, total_len, dtype=torch.bool, device=device)
full_position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

# Create labels: shift right (predict next token)
labels = full_input_ids[:, 1:].contiguous()  # (N, total_len-1)
label_mask = torch.ones(N_SEQUENCES, total_len - 1, dtype=torch.bool, device=device)

with torch.no_grad():
    output_normal = model(input_ids=full_input_ids, attention_mask=full_attention_mask,
                         position_ids=full_position_ids)
    logits_normal = output_normal.logits.clone()  # (N, total_len, vocab_size_sharded)
    # Immediately compute log_probs and free logits
    log_probs_normal_all = F.log_softmax(logits_normal.float(), dim=-1).clone()

# Free normal forward memory
del output_normal, logits_normal
torch.cuda.empty_cache()

if local_rank == 0:
    mem_after_normal = torch.cuda.memory_allocated() / 1024**3
    print(f"  After normal forward + cleanup: GPU {mem_after_normal:.2f} GB")

# Select suffix portion log_probs (positions PREFIX_LEN to total_len-1, which is suffix_len positions)
# suffix_len positions in logits, but labels are suffix_len-1 (shift by 1)
suffix_log_probs_normal = log_probs_normal_all[:, PREFIX_LEN:, :]  # (N, suffix_len, vocab)

# For labels in suffix portion: tokens at positions PREFIX_LEN+1 to total_len-1
# i.e., suffix_input_ids shifted by 1 (predict next suffix token)
suffix_labels = suffix_input_ids[:, 1:]  # (N, suffix_len-1) = (N, 31)

# Old log_probs for PPO ratio computation — move to CPU to free GPU memory
# suffix_log_probs_normal has suffix_len positions, suffix_labels has suffix_len-1 positions
old_log_prob_selected = suffix_log_probs_normal[:, :-1, :].gather(
    dim=-1, index=suffix_labels.unsqueeze(-1)
).squeeze(-1).to(device='cpu', dtype=torch.float32)  # (N, suffix_len-1) on CPU
del log_probs_normal_all, suffix_log_probs_normal
torch.cuda.empty_cache()

if local_rank == 0:
    mem_after_normal = torch.cuda.memory_allocated() / 1024**3
    print(f"  After normal forward + cleanup: GPU {mem_after_normal:.2f} GB")

# ===== Step 2: Two-pass PS forward =====
if local_rank == 0:
    print("Step 2: Two-pass PS forward")

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

torch.cuda.synchronize()
t_ps_start = time.time()

with prefix_sharing_runtime_context(ps_runtime_state) as ctx_state:
    # Prefix pass
    with torch.no_grad():
        prefix_output = model(
            input_ids=ps_runtime_state.prefix_input_ids,
            attention_mask=ps_runtime_state.prefix_attention_mask,
            position_ids=ps_runtime_state.prefix_position_ids,
        )
    del prefix_output

    if local_rank == 0:
        print(f"  Prefix pass done: {len(ctx_state.store._entries)} attn KV, "
              f"{len(ctx_state.deltanet_store._entries)} DeltaNet states")

    # Suffix pass — need gradients for training
    suffix_output = model(
        input_ids=suffix_input_ids,
        attention_mask=suffix_attention_mask,
        position_ids=suffix_position_ids,
    )
    logits_ps = suffix_output.logits  # (N, SUFFIX_LEN, vocab_size_sharded)

torch.cuda.synchronize()
t_ps = time.time() - t_ps_start

# Compute log_probs for suffix portion (suffix_len positions, take first suffix_len-1)
log_probs_ps_all = F.log_softmax(logits_ps.float(), dim=-1)  # (N, suffix_len, vocab)
new_log_prob_selected = log_probs_ps_all[:, :-1, :].gather(
    dim=-1, index=suffix_labels.unsqueeze(-1)
).squeeze(-1)  # (N, suffix_len-1)
del log_probs_ps_all  # Free memory

# ===== Step 3: Precision check (cos_sim between PS and normal log_probs) =====
if local_rank == 0:
    print("Step 3: Precision alignment (log_probs comparison)")

# Move old_log_prob_selected back to GPU for comparison
old_log_prob_gpu = old_log_prob_selected.to(device=device)

all_cos_sims = []
for i in range(N_SEQUENCES):
    cos_sim = F.cosine_similarity(
        new_log_prob_selected[i].float().flatten(),
        old_log_prob_gpu[i].float().flatten(),
        dim=0,
    ).item()
    all_cos_sims.append(cos_sim)
overall_cos = sum(all_cos_sims) / len(all_cos_sims)

if local_rank == 0:
    print(f"  Overall log_prob cos_sim: {overall_cos:.6f} (threshold: 0.99)")
    print(f"  Min log_prob cos_sim: {min(all_cos_sims):.6f}")
    # Note: log_prob cos_sim is lower than logits cos_sim because log_probs are sparse
    # Only a few vocab entries have significant probability
    status = "PASS" if overall_cos >= 0.99 else "FAIL"
    print(f"  Precision: {status}")

# ===== Step 4: GRPO group-normalized advantage =====
if local_rank == 0:
    print("Step 4: GRPO advantage computation")

# Simulate rewards: random per-sequence outcome rewards
torch.manual_seed(SEED + 1000)
rewards = torch.randn(N_SEQUENCES, dtype=torch.float32, device=device)

# GRPO advantage: group normalization (no critic needed)
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # (N,)
# Expand to per-position: same advantage for all positions in a sequence
response_len = SUFFIX_LEN - 1  # number of prediction positions in suffix
advantages_per_pos = advantages.unsqueeze(1).expand(N_SEQUENCES, response_len)  # (N, response_len)

# ===== Step 5: PPO clipped policy loss =====
if local_rank == 0:
    print("Step 5: PPO clipped policy loss")

# Old log_probs already computed and stored in step 1

# New log_probs (from PS forward, already in new_log_prob_selected)

# PPO clipped ratio
clip_ratio = 0.2
# Move old_log_prob_selected back to GPU for ratio computation
old_log_prob_gpu = old_log_prob_selected.to(device=device)
ratio = torch.exp(new_log_prob_selected - old_log_prob_gpu)  # (N, SUFFIX_LEN)
clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)

# Policy loss: -min(ratio * advantage, clipped_ratio * advantage)
policy_loss_per_pos = -torch.min(ratio * advantages_per_pos, clipped_ratio * advantages_per_pos)
policy_loss = policy_loss_per_pos.mean()  # scalar

if local_rank == 0:
    print(f"  Policy loss: {policy_loss.item():.4f}")
    print(f"  Ratio mean: {ratio.mean().item():.4f}, std: {ratio.std().item():.4f}")
    print(f"  Clipped fraction: {(ratio.abs() > 1 + clip_ratio).float().mean().item():.4f}")

# ===== Step 6: Backward pass =====
if local_rank == 0:
    print("Step 6: Backward pass (gradient flow test)")

# Zero gradients first
model.zero_grad()

# Try backward — may OOM on 24GB GPU with 27B model
try:
    policy_loss.backward()

    # Check gradient flow
    total_grad_norm = 0.0
    params_with_grad = 0
    layers_with_grad = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
            params_with_grad += 1
    total_grad_norm = total_grad_norm ** 0.5

    # Check which layers got gradients
    has_layer_grad = set()
    for name, _ in model.named_parameters():
        if name.split('.')[2].isdigit() if 'layers.' in name else False:
            layer_idx = int(name.split('layers.')[1].split('.')[0])
            has_layer_grad.add(layer_idx)
    # Actually check which layers have at least one parameter with grad
    for name, param in model.named_parameters():
        if param.grad is not None:
            for i in range(num_layers):
                if f"layers.{i}" in name:
                    has_layer_grad.add(i)

    layers_with_grad = len(has_layer_grad)
    grad_status = "PASS" if layers_with_grad == num_layers else "PARTIAL"

    if local_rank == 0:
        print(f"  Total grad norm: {total_grad_norm:.4f}")
        print(f"  Parameters with grad: {params_with_grad}/{sum(1 for p in model.parameters() if p.requires_grad)}")
        print(f"  Layers with gradient: {layers_with_grad}/{num_layers}")
        print(f"  Gradient flow: {grad_status}")

except torch.cuda.OutOfMemoryError:
    if local_rank == 0:
        print("  Backward OOM — expected for 27B model on 24GB GPU")
        print("  Gradient flow: SKIPPED (OOM)")
        print("  Note: In real GRPO training, gradient checkpointing is used")
    # OOM during backward — this is expected on 24GB GPU without gradient checkpointing
    # Clear gradients and continue
    model.zero_grad()
    torch.cuda.empty_cache()
    grad_status = "OOM"
    total_grad_norm = -1.0
    layers_with_grad = 0

# ===== Step 7: Summary =====
patch_handle.disable()

if local_rank == 0:
    print(f"\n{'='*70}")
    print("GRPO PS TRAINING SIMULATION SUMMARY")
    print(f"{'='*70}")
    print(f"Model: Qwen3.6-27B (TP=4, bf16, RmPad, pretrained)")
    print(f"N={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"Precision: cos_sim={overall_cos:.6f} [{status}]")
    print(f"PS time: {t_ps:.3f}s")
    print(f"Policy loss: {policy_loss.item():.4f}")
    print(f"Gradient norm: {total_grad_norm:.4f}")
    print(f"Gradient flow: {layers_with_grad}/{num_layers} layers")
    compute_savings = (N_SEQUENCES * total_len - PREFIX_LEN - N_SEQUENCES * SUFFIX_LEN) / (N_SEQUENCES * total_len) * 100
    print(f"Compute savings: {compute_savings:.1f}%")
    print(f"{'='*70}")

    if overall_cos >= 0.999 and layers_with_grad == num_layers:
        print("PASS: GRPO PS training simulation complete!")
    else:
        print("PARTIAL: Some issues detected")

# Cleanup
del logits_ps, suffix_output
torch.cuda.empty_cache()

parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()