#!/usr/bin/env python3
"""BSHD-mode PS E2E with real Qwen3.6 converted weights (TP=4, 4 GPUs).

Uses verl's load_state_dict_to_megatron_gptmodel to load converted weights
into a TP-sharded Megatron model, then runs the full PS E2E test:
1. Normal forward on full sequence (prefix + suffix) → reference logits
2. Two-pass PS forward: prefix pass (store KV/carry) → suffix pass (inject states)
3. Compare suffix log_probs: cos_sim should be >= 0.99

Usage: torchrun --nproc_per_node=4 scripts/test_bshd_ps_e2e_real_weights.py
"""
import os, sys, time
from dataclasses import dataclass

WORK_DIR = os.path.expanduser("~/rollout-prefix/prefix-0501")
VERL_DIR = os.path.join(WORK_DIR, "dependency/verl_v070")
PS_DIR = os.path.join(WORK_DIR, "prefix-sharing")
MEGATRON_DIR = os.path.join(WORK_DIR, "dependency/Megatron-LM-core_v0.12.1")

sys.path.insert(0, VERL_DIR)
sys.path.insert(0, PS_DIR)
sys.path.insert(0, MEGATRON_DIR)

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoConfig

HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only-16layers")
CONVERTED_PT = os.path.expanduser("~/rollout-prefix/qwen36_megatron_converted.pt")
TP_SIZE = 4
PREFIX_LEN = 64
SUFFIX_LEN = 32
N_SEQUENCES = 8
SEED = 42

# Initialize distributed
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
rank = torch.distributed.get_rank()
device = torch.device(f"cuda:{local_rank}")

def _gather_logits_tp(logits_per_rank):
    """AllGather logits across TP ranks to get full vocab_size logits."""
    if TP_SIZE <= 1:
        return logits_per_rank
    tp_group = parallel_state.get_tensor_model_parallel_group()
    logits_list = [torch.empty_like(logits_per_rank) for _ in range(TP_SIZE)]
    torch.distributed.all_gather(logits_list, logits_per_rank, group=tp_group)
    return torch.cat(logits_list, dim=-1)

# Load HF config
config = AutoConfig.from_pretrained(HF_MODEL_PATH)
if local_rank == 0:
    print(f"Model config: num_hidden_layers={config.num_hidden_layers}, "
          f"full_attention_interval={config.full_attention_interval}, "
          f"partial_rotary_factor={config.partial_rotary_factor}, "
          f"attn_output_gate={config.attn_output_gate}")

# Initialize model
from verl.models.mcore.registry import init_mcore_model, hf_to_mcore_config

tfconfig = hf_to_mcore_config(config, torch.bfloat16, sequence_parallel=False)
if local_rank == 0:
    print(f"TransformerConfig: sequence_parallel={tfconfig.sequence_parallel}")
model = init_mcore_model(tfconfig, config, pre_process=True, post_process=True)
model = model.to(device)

# ===== Load real converted weights (direct TP sharding, no distributed loader) =====
if local_rank == 0:
    print(f"\n[1] Loading converted weights from {CONVERTED_PT}...")

# All ranks load from shared filesystem (simpler than distributed broadcast)
converted_sd = torch.load(CONVERTED_PT, map_location="cpu", weights_only=True)
if local_rank == 0:
    print(f"  Loaded {len(converted_sd)} keys from converted .pt")

FULL_ATTENTION_INTERVAL = config.full_attention_interval  # 4
num_layers = config.num_hidden_layers  # 16

# Sharding dimensions
sa_num_heads = config.num_attention_heads  # 24 (SelfAttention)
sa_num_kv_heads = config.num_key_value_heads  # 4 (SelfAttention)
sa_head_dim = config.head_dim  # 256 (SelfAttention)
dn_num_heads = getattr(config, 'deltanet_num_heads', 48)  # DeltaNet
dn_num_kv_heads = getattr(config, 'deltanet_kv_heads', 16)  # DeltaNet
dn_head_dim = sa_head_dim // 2 if sa_head_dim == 256 else 128  # 128 (DeltaNet)
hidden_size = config.hidden_size  # 5120
intermediate_size = config.intermediate_size  # 17408

# Embedding + final layernorm + lm_head (vocab-sharded)
model.embedding.word_embeddings.weight.data.copy_(
    converted_sd['model.embed_tokens.weight'][tp_rank * (config.vocab_size // TP_SIZE):
                                               (tp_rank + 1) * (config.vocab_size // TP_SIZE)]
)
model.decoder.final_layernorm.weight.data.copy_(converted_sd['model.norm.weight'])
model.output_layer.weight.data.copy_(
    converted_sd['lm_head.weight'][tp_rank * (config.vocab_size // TP_SIZE):
                                    (tp_rank + 1) * (config.vocab_size // TP_SIZE)]
)

for layer_idx in range(num_layers):
    is_dn = layer_idx % FULL_ATTENTION_INTERVAL != FULL_ATTENTION_INTERVAL - 1
    layer_name = f"model.layers.{layer_idx}"
    layer = model.decoder.layers[layer_idx]
    attn = layer.self_attention

    # Input layernorm
    ln_key = f'{layer_name}.input_layernorm.weight'
    if hasattr(layer, 'input_layernorm') and hasattr(layer.input_layernorm, 'weight'):
        layer.input_layernorm.weight.data.copy_(converted_sd[ln_key])
    elif hasattr(attn.linear_qkv, 'layer_norm_weight'):
        attn.linear_qkv.layer_norm_weight.data.copy_(converted_sd[ln_key])

    # Post attention layernorm
    post_ln_key = f'{layer_name}.post_attention_layernorm.weight'
    if hasattr(layer, 'pre_mlp_layernorm') and hasattr(layer.pre_mlp_layernorm, 'weight'):
        layer.pre_mlp_layernorm.weight.data.copy_(converted_sd[post_ln_key])
    elif hasattr(layer.mlp.linear_fc1, 'layer_norm_weight'):
        layer.mlp.linear_fc1.layer_norm_weight.data.copy_(converted_sd[post_ln_key])

    # MLP (gate+up interleaved, down column-sharded)
    gate_w = converted_sd[f'{layer_name}.mlp.gate_proj.weight']
    up_w = converted_sd[f'{layer_name}.mlp.up_proj.weight']
    int_size_tp = intermediate_size // TP_SIZE
    gate_w_tp = gate_w[tp_rank * int_size_tp:(tp_rank + 1) * int_size_tp]
    up_w_tp = up_w[tp_rank * int_size_tp:(tp_rank + 1) * int_size_tp]
    layer.mlp.linear_fc1.weight.data.copy_(torch.cat([gate_w_tp, up_w_tp], dim=0))
    layer.mlp.linear_fc2.weight.data.copy_(
        converted_sd[f'{layer_name}.mlp.down_proj.weight'][:, tp_rank * int_size_tp:(tp_rank + 1) * int_size_tp]
    )

    # Q/K norm (shared across both attention types)
    if hasattr(attn, 'q_layernorm') and attn.q_layernorm is not None:
        qnorm_key = f'{layer_name}.self_attn.q_norm.weight'
        if qnorm_key in converted_sd:
            attn.q_layernorm.weight.data.copy_(converted_sd[qnorm_key])
    if hasattr(attn, 'k_layernorm') and attn.k_layernorm is not None:
        knorm_key = f'{layer_name}.self_attn.k_norm.weight'
        if knorm_key in converted_sd:
            attn.k_layernorm.weight.data.copy_(converted_sd[knorm_key])

    # o_proj (column-sharded for both types)
    o_proj_key = f'{layer_name}.self_attn.o_proj.weight'
    if is_dn:
        # DeltaNet: o_proj maps from (num_heads*head_dim) → hidden_size
        # ColumnParallelLinear shards output dim, so we shard along dim=1
        dn_hidden_per_head = dn_num_heads * dn_head_dim  # 48*128=6144
        layer.self_attention.linear_proj.weight.data.copy_(
            converted_sd[o_proj_key][:, tp_rank * (dn_hidden_per_head // TP_SIZE):
                                      (tp_rank + 1) * (dn_hidden_per_head // TP_SIZE)]
        )
    else:
        # SelfAttention: o_proj maps from (num_heads*head_dim) → hidden_size
        # ColumnParallelLinear shards output dim (dim=1 for RowParallel)
        sa_hidden_per_head = sa_num_heads * sa_head_dim  # 24*256=6144
        layer.self_attention.linear_proj.weight.data.copy_(
            converted_sd[o_proj_key][:, tp_rank * (sa_hidden_per_head // TP_SIZE):
                                      (tp_rank + 1) * (sa_hidden_per_head // TP_SIZE)]
        )

    if is_dn:
        # DeltaNet QKV: simple [Q, K, V] concat with TP sharding
        q_w = converted_sd[f'{layer_name}.self_attn.q_proj.weight']
        k_w = converted_sd[f'{layer_name}.self_attn.k_proj.weight']
        v_w = converted_sd[f'{layer_name}.self_attn.v_proj.weight']
        q_size_tp = dn_num_heads * dn_head_dim // TP_SIZE  # 48*128//4=1536
        k_size_tp = dn_num_kv_heads * dn_head_dim // TP_SIZE  # 16*128//4=512
        v_size_tp = dn_num_kv_heads * dn_head_dim // TP_SIZE  # 512
        q_tp = q_w[tp_rank * q_size_tp:(tp_rank + 1) * q_size_tp]
        k_tp = k_w[tp_rank * k_size_tp:(tp_rank + 1) * k_size_tp]
        v_tp = v_w[tp_rank * v_size_tp:(tp_rank + 1) * v_size_tp]
        attn.linear_qkv.weight.data.copy_(torch.cat([q_tp, k_tp, v_tp], dim=0))

        # DeltaNet-specific projections (beta, decay sharded along output dim)
        attn.beta_proj.weight.data.copy_(
            converted_sd[f'{layer_name}.self_attn.beta_proj.weight']
            [tp_rank * (dn_num_heads // TP_SIZE):(tp_rank + 1) * (dn_num_heads // TP_SIZE)]
        )
        if f'{layer_name}.self_attn.beta_proj.bias' in converted_sd:
            attn.beta_proj.bias.data.copy_(
                converted_sd[f'{layer_name}.self_attn.beta_proj.bias']
                [tp_rank * (dn_num_heads // TP_SIZE):(tp_rank + 1) * (dn_num_heads // TP_SIZE)]
            )
        attn.decay_proj.weight.data.copy_(
            converted_sd[f'{layer_name}.self_attn.decay_proj.weight']
            [tp_rank * (dn_num_heads // TP_SIZE):(tp_rank + 1) * (dn_num_heads // TP_SIZE)]
        )
        if f'{layer_name}.self_attn.decay_proj.bias' in converted_sd:
            attn.decay_proj.bias.data.copy_(
                converted_sd[f'{layer_name}.self_attn.decay_proj.bias']
                [tp_rank * (dn_num_heads // TP_SIZE):(tp_rank + 1) * (dn_num_heads // TP_SIZE)]
            )

        # DeltaNet gate_proj (sharded along output dim: num_heads*head_dim)
        if hasattr(attn, 'gate_proj'):
            gate_key = f'{layer_name}.self_attn.gate_proj.weight'
            if gate_key in converted_sd:
                dn_gate_dim = dn_num_heads * dn_head_dim  # 6144
                attn.gate_proj.weight.data.copy_(
                    converted_sd[gate_key]
                    [tp_rank * (dn_gate_dim // TP_SIZE):(tp_rank + 1) * (dn_gate_dim // TP_SIZE)]
                )

        # DeltaNet buffers (not TP-sharded, all ranks get full copy)
        for buf_name, sd_name in [
            ('conv1d_weight', f'{layer_name}.self_attn.conv1d.weight'),
            ('A_log', f'{layer_name}.self_attn.A_log'),
            ('dt_bias', f'{layer_name}.self_attn.dt_bias'),
            ('norm_weight', f'{layer_name}.self_attn.norm.weight'),
        ]:
            if sd_name in converted_sd:
                setattr(attn, buf_name,
                        converted_sd[sd_name].clone().to(attn.linear_qkv.weight.device))

    else:
        # SelfAttention QKV: GQA-interleaved format with TP sharding
        q_w = converted_sd[f'{layer_name}.self_attn.q_proj.weight']
        k_w = converted_sd[f'{layer_name}.self_attn.k_proj.weight']
        v_w = converted_sd[f'{layer_name}.self_attn.v_proj.weight']
        # GQA interleaving: per KV-group, [Q_group, K_group, V_group]
        num_q_heads = sa_num_heads  # 24
        num_kv_heads = sa_num_kv_heads  # 4
        head_dim = sa_head_dim  # 256
        num_qg = num_q_heads // num_kv_heads  # 6
        q_size_tp = num_q_heads * head_dim // TP_SIZE  # 1536
        kv_size_tp = num_kv_heads * head_dim // TP_SIZE  # 256
        qkv_tp_size = q_size_tp + 2 * kv_size_tp  # 2048
        # Per TP rank: 1 KV group (since 4 KV groups / 4 TP = 1)
        num_kv_groups_tp = num_kv_heads // TP_SIZE  # 1
        qkv_interleaved = torch.zeros(qkv_tp_size, hidden_size, dtype=q_w.dtype)
        for g in range(num_kv_groups_tp):
            # Global group index for this TP rank
            global_g = tp_rank * num_kv_groups_tp + g
            q_per_group = num_qg * head_dim  # 6*256=1536
            k_per_group = head_dim  # 256
            v_per_group = head_dim  # 256
            total_per_group = q_per_group + k_per_group + v_per_group  # 2048
            q_start = global_g * num_qg * head_dim
            q_end = q_start + q_per_group
            k_start = global_g * head_dim
            k_end = k_start + k_per_group
            v_start = global_g * head_dim
            v_end = v_start + v_per_group
            offset = g * total_per_group
            qkv_interleaved[offset:offset + q_per_group] = q_w[q_start:q_end]
            qkv_interleaved[offset + q_per_group:offset + q_per_group + k_per_group] = k_w[k_start:k_end]
            qkv_interleaved[offset + q_per_group + k_per_group:offset + total_per_group] = v_w[v_start:v_end]
        attn.linear_qkv.weight.data.copy_(qkv_interleaved)

        # SelfAttention gate_proj (sharded along output dim: hidden_size)
        if hasattr(attn, 'gate_proj'):
            gate_key = f'{layer_name}.self_attn.gate_proj.weight'
            if gate_key in converted_sd:
                attn.gate_proj.weight.data.copy_(
                    converted_sd[gate_key]
                    [tp_rank * (hidden_size // TP_SIZE):(tp_rank + 1) * (hidden_size // TP_SIZE)]
                )

if local_rank == 0:
    print("  Weights loaded via direct TP sharding!")

del converted_sd
torch.cuda.empty_cache()

# Verify model architecture
num_layers = len(model.decoder.layers)
full_attn_count = sum(1 for i in range(num_layers) if i % config.full_attention_interval == config.full_attention_interval - 1)
deltanet_count = num_layers - full_attn_count
if local_rank == 0:
    attn_types = []
    for i in range(num_layers):
        attn = model.decoder.layers[i].self_attention
        type_name = attn.__class__.__name__
        attn_types.append(f"L{i}:{type_name}")
    print(f"Model layers: {attn_types}")
    print(f"Full attention: {full_attn_count}, DeltaNet: {deltanet_count}")

mem = torch.cuda.memory_allocated() / 1024**3
if local_rank == 0:
    print(f"Model loaded with real weights, GPU {mem:.2f} GiB")

# ===== Step 1: Normal forward (reference) =====
vocab_size = config.vocab_size
total_len = PREFIX_LEN + SUFFIX_LEN

torch.manual_seed(SEED + 500)
prefix_tokens = torch.randint(0, vocab_size, (PREFIX_LEN,), dtype=torch.long, device=device)
suffix_tokens_list = [torch.randint(0, vocab_size, (SUFFIX_LEN,), dtype=torch.long, device=device)
                      for _ in range(N_SEQUENCES)]
full_sequences = [torch.cat([prefix_tokens, suffix]) for suffix in suffix_tokens_list]
full_input_ids = torch.stack(full_sequences)  # (N, total_len)
full_position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

torch.cuda.synchronize()
t_normal_start = time.time()

layer_outputs_normal = {}
def capture_normal_output(module, input, output, layer_idx):
    if isinstance(output, tuple):
        layer_outputs_normal[layer_idx] = output[0].detach().cpu().float()

hooks_normal = []
for i in range(num_layers):
    h = model.decoder.layers[i].register_forward_hook(
        lambda m, inp, out, idx=i: capture_normal_output(m, inp, out, idx)
    )
    hooks_normal.append(h)

with torch.no_grad():
    logits_normal = model(
        input_ids=full_input_ids,
        position_ids=full_position_ids,
        attention_mask=None,
    )

torch.cuda.synchronize()
t_normal = time.time() - t_normal_start

for h in hooks_normal:
    h.remove()

full_logits_normal = _gather_logits_tp(logits_normal)
del logits_normal

log_probs_normal = F.log_softmax(full_logits_normal.float(), dim=-1)
suffix_labels = full_input_ids[:, PREFIX_LEN:]
selected_normal = log_probs_normal[:, PREFIX_LEN-1:-1, :].gather(
    dim=-1, index=suffix_labels.unsqueeze(-1)
).squeeze(-1).to(device='cpu', dtype=torch.float32)

del log_probs_normal
torch.cuda.empty_cache()

if local_rank == 0:
    print(f"Step 1 (normal forward): {t_normal:.3f}s, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

# ===== Step 2: Two-pass PS forward =====
suffix_input_ids = torch.stack(suffix_tokens_list)
prefix_input_ids = prefix_tokens.unsqueeze(0)
suffix_position_ids = torch.arange(PREFIX_LEN, total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)
prefix_position_ids = torch.arange(PREFIX_LEN, device=device).unsqueeze(0)

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.model_spec import ModelSpec
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.backends.packed_layout import PackedBatchLayout

ps_config = PrefixSharingConfig(
    enable_prefix_sharing=True,
    min_prefix_len=PREFIX_LEN,
    min_group_size=N_SEQUENCES,
)

sequences_list = [seq.tolist() for seq in full_input_ids]
ps_plan = PrefixSharingPlanner(ps_config).plan(sequences_list)

suffix_lengths = [SUFFIX_LEN] * N_SEQUENCES
suffix_layout = PackedBatchLayout.from_valid_lengths(suffix_lengths)

model_spec = ModelSpec.from_hf_config(config)

# Install SelfAttention KV injection monkey-patch
# This makes SelfAttention layers (L3,7,11,15) store/inject KV during PS forward.
# DeltaNet layers have their own forward (carry injection) — unaffected by this patch.
from prefix_sharing.integrations.megatron_attention import MegatronAttentionIntegration
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
attn_integration = MegatronAttentionIntegration(
    config=ps_config,
    backend=TorchReferenceBackend(),
)
patch_handle = attn_integration.install()
if local_rank == 0:
    print("  SelfAttention KV injection patch installed")
    # Verify patch is active
    from megatron.core.transformer.attention import SelfAttention
    sa_fwd_name = SelfAttention.forward.__name__
    print(f"  SelfAttention.forward name: {sa_fwd_name}")
    # Check L3 SelfAttention instance
    l3_attn = model.decoder.layers[3].self_attention
    l3_fwd_name = l3_attn.forward.__func__.__name__ if hasattr(l3_attn.forward, '__func__') else 'unknown'
    l3_cls = l3_attn.__class__.__name__
    print(f"  L3 attn class: {l3_cls}, forward.__func__.__name: {l3_fwd_name}")
    # Check if forward on the instance vs class are different
    l3_instance_fwd = type(l3_attn).__dict__.get('forward', None)
    print(f"  L3 instance dict has 'forward': {l3_instance_fwd is not None}")
    sa_class_fwd = SelfAttention.__dict__.get('forward', None)
    print(f"  SelfAttention class dict 'forward' name: {sa_class_fwd.__name__ if sa_class_fwd else 'None'}")

@dataclass
class PsState:
    prefix_sharing_plan: object
    packed_batch_layout: object
    backend: object
    model_spec: object

ps_runtime_state = PsState(
    prefix_sharing_plan=ps_plan,
    packed_batch_layout=suffix_layout,
    backend=None,
    model_spec=model_spec,
)

torch.cuda.synchronize()
t_ps_start = time.time()

layer_outputs_ps = {}
def capture_ps_output(module, input, output, layer_idx):
    if isinstance(output, tuple):
        layer_outputs_ps[layer_idx] = output[0].detach().cpu().float()

hooks_ps = []
for i in range(num_layers):
    h = model.decoder.layers[i].register_forward_hook(
        lambda m, inp, out, idx=i: capture_ps_output(m, inp, out, idx)
    )
    hooks_ps.append(h)

with prefix_sharing_runtime_context(ps_runtime_state) as ctx:
    # Prefix pass
    with torch.no_grad():
        prefix_logits = model(
            input_ids=prefix_input_ids,
            position_ids=prefix_position_ids,
            attention_mask=None,
        )
    del prefix_logits
    torch.cuda.empty_cache()

    if local_rank == 0:
        print(f"  Prefix pass done, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
        kv_count = len(ctx.store._entries)
        dn_count = len(ctx.deltanet_store._entries)
        print(f"  Store: KV slots={kv_count}, DeltaNet slots={dn_count}")

    # Suffix pass
    suffix_logits = model(
        input_ids=suffix_input_ids,
        position_ids=suffix_position_ids,
        attention_mask=None,
    )

torch.cuda.synchronize()
t_ps = time.time() - t_ps_start

for h in hooks_ps:
    h.remove()

# Per-layer hidden state comparison
if local_rank == 0:
    print("\nPer-layer hidden state cos_sim (suffix positions):")
    for i in range(num_layers):
        if i in layer_outputs_normal and i in layer_outputs_ps:
            normal_suffix_hidden = layer_outputs_normal[i][PREFIX_LEN:, :, :]
            ps_suffix_hidden = layer_outputs_ps[i]
            if normal_suffix_hidden.shape == ps_suffix_hidden.shape:
                cs = F.cosine_similarity(
                    normal_suffix_hidden.flatten(),
                    ps_suffix_hidden.flatten(),
                    dim=0,
                ).item()
                print(f"  L{i} ({model.decoder.layers[i].self_attention.__class__.__name__}): "
                      f"cos_sim = {cs:.6f}")
            else:
                print(f"  L{i}: shape mismatch normal={normal_suffix_hidden.shape} "
                      f"vs ps={ps_suffix_hidden.shape}")

full_suffix_logits = _gather_logits_tp(suffix_logits)
del suffix_logits

# Per-position logits cos_sim
normal_suffix_logits_raw = full_logits_normal[:, PREFIX_LEN-1:-1, :].float()
ps_suffix_logits_raw = full_suffix_logits.float()

if local_rank == 0:
    logits_cos_per_pos = []
    for pos_idx in range(SUFFIX_LEN - 1):
        cs = F.cosine_similarity(
            normal_suffix_logits_raw[:, pos_idx+1, :],
            ps_suffix_logits_raw[:, pos_idx, :],
            dim=-1,
        ).mean().item()
        logits_cos_per_pos.append(cs)
    print(f"  Per-position logits cos_sim: min={min(logits_cos_per_pos):.6f}, "
          f"max={max(logits_cos_per_pos):.6f}, mean={sum(logits_cos_per_pos)/len(logits_cos_per_pos):.6f}")

del full_logits_normal, normal_suffix_logits_raw, ps_suffix_logits_raw
torch.cuda.empty_cache()

log_probs_ps = F.log_softmax(full_suffix_logits.float(), dim=-1)
suffix_labels_ps = suffix_input_ids[:, 1:]
selected_ps = log_probs_ps[:, :-1, :].gather(
    dim=-1, index=suffix_labels_ps.unsqueeze(-1)
).squeeze(-1)

del full_suffix_logits, log_probs_ps
torch.cuda.empty_cache()

if local_rank == 0:
    print(f"Step 2 (PS forward): {t_ps:.3f}s, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

# ===== Step 3: Precision check =====
old_aligned = selected_normal[:, 1:].to(device=device, dtype=torch.float32)
new_aligned = selected_ps.to(device=device, dtype=torch.float32)

all_cos_sims = []
for i in range(N_SEQUENCES):
    cos_sim = F.cosine_similarity(
        new_aligned[i].flatten(),
        old_aligned[i].flatten(),
        dim=0,
    ).item()
    all_cos_sims.append(cos_sim)
overall_cos = sum(all_cos_sims) / len(all_cos_sims)
max_diff = (new_aligned - old_aligned).abs().max().item()

if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"BSHD PS E2E Real Weights Test (Qwen3-27B 16-layer, TP=4)")
    print(f"{'='*60}")
    print(f"Precision: log_prob cos_sim = {overall_cos:.6f}")
    print(f"           max_diff = {max_diff:.6f}")
    print(f"           per-sequence: {all_cos_sims}")
    status = "PASS" if overall_cos >= 0.99 else "FAIL"
    print(f"Status: {status}")
    print(f"Normal forward time: {t_normal:.3f}s")
    print(f"PS forward time: {t_ps:.3f}s")
    speedup = t_normal / t_ps if t_ps > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'='*60}")

# ===== Step 4: Backward pass =====
ps_runtime_state2 = PsState(
    prefix_sharing_plan=ps_plan,
    packed_batch_layout=suffix_layout,
    backend=None,
    model_spec=model_spec,
)

grad_status = "SKIP"
with prefix_sharing_runtime_context(ps_runtime_state2) as ctx2:
    with torch.no_grad():
        prefix_logits2 = model(
            input_ids=prefix_input_ids,
            position_ids=prefix_position_ids,
            attention_mask=None,
        )
    del prefix_logits2
    torch.cuda.empty_cache()

    suffix_logits2 = model(
        input_ids=suffix_input_ids,
        position_ids=suffix_position_ids,
        attention_mask=None,
    )

loss = suffix_logits2.float().mean()

model.zero_grad()
try:
    loss.backward()
    total_grad_norm = 0.0
    params_with_grad = 0
    layers_with_grad = set()
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
            params_with_grad += 1
            for i in range(num_layers):
                if f"layers.{i}" in name or f"decoder.layers.{i}" in name:
                    layers_with_grad.add(i)
    total_grad_norm = total_grad_norm ** 0.5
    n_layers_grad = len(layers_with_grad)
    grad_status = "PASS" if n_layers_grad == num_layers else "PARTIAL"
    if local_rank == 0:
        print(f"Backward: {grad_status}, grad_norm={total_grad_norm:.4f}, "
              f"params_with_grad={params_with_grad}, layers={n_layers_grad}/{num_layers}")
except torch.cuda.OutOfMemoryError:
    if local_rank == 0:
        print("Backward OOM!")
    grad_status = "OOM"
    model.zero_grad()
    torch.cuda.empty_cache()

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
if local_rank == 0:
    print(f"Peak GPU memory: {peak_mem:.2f} GiB")
    if overall_cos >= 0.99 and grad_status == "PASS":
        print("\n*** BSHD PS E2E Real Weights: ALL PASS ***")
    elif overall_cos >= 0.99:
        print(f"\n*** BSHD PS E2E Real Weights: PARTIAL (forward PASS, backward {grad_status}) ***")
    else:
        print(f"\n*** BSHD PS E2E Real Weights: FAIL (cos_sim={overall_cos:.6f}) ***")

# Cleanup
patch_handle.disable()
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()