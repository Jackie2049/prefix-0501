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

# ===== Load real converted weights via verl loader =====
if local_rank == 0:
    print(f"\n[1] Loading converted weights from {CONVERTED_PT}...")

# Only rank 0 loads the state_dict; other ranks pass empty dict
if rank == 0:
    converted_sd = torch.load(CONVERTED_PT, map_location="cpu", weights_only=True)
    if local_rank == 0:
        print(f"  Loaded {len(converted_sd)} keys from converted .pt")
else:
    converted_sd = {}

from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel
load_state_dict_to_megatron_gptmodel(converted_sd, [model], config, torch.bfloat16)

if local_rank == 0:
    print("  Weights loaded via verl loader!")

# Free state_dict on rank 0 to save CPU memory
if rank == 0:
    del converted_sd

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
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()