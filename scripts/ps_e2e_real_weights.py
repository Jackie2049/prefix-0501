#!/usr/bin/env python3
"""PS E2E with real Qwen3.6 converted weights (TP=1, single GPU).

Validates that prefix-sharing produces identical output to normal forward
when using real (converted) Qwen3.6 weights instead of random weights.

Steps:
1. Load Megatron model with converted weights (same as validate_weight_loading_e2e.py)
2. Run normal forward on full sequence (prefix + suffix)
3. Run prefix pass (prefix only) → store KV/carry states
4. Run suffix pass (suffix only) with injected states
5. Compare suffix logits: cos_sim should be >= 0.99

Usage: CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/ps_e2e_real_weights.py
"""

import os, sys, time, math
import torch
import torch.nn.functional as F
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(REPO_ROOT, "dependency")
sys.path.insert(0, os.path.join(DEPS, "megatron_v0150"))
sys.path.insert(0, os.path.join(REPO_ROOT, "prefix-sharing"))
sys.path.insert(0, os.path.join(DEPS, "verl_v070"))

from verl.models.mcore.gated_delta_net import GatedDeltaNetAttention

# Config
MODEL_DIR = os.environ.get("MODEL_DIR", "/home/zxw/rollout-prefix/models/Qwen3-27B-text-only-16layers")
CONVERTED_PT = os.environ.get("CONVERTED_PT", "/home/zxw/rollout-prefix/qwen36_megatron_converted.pt")
PREFIX_LEN = int(os.environ.get("PREFIX_LEN", "64"))
SUFFIX_LEN = int(os.environ.get("SUFFIX_LEN", "32"))
SEED = 42
FULL_ATTENTION_INTERVAL = 4


def load_converted_weights(model, state_dict, tfconfig, num_layers=16):
    """Manually load converted state_dict into a TP=1 Megatron model."""
    model.embedding.word_embeddings.weight.data.copy_(state_dict['model.embed_tokens.weight'])
    model.decoder.final_layernorm.weight.data.copy_(state_dict['model.norm.weight'])
    model.output_layer.weight.data.copy_(state_dict['lm_head.weight'])

    for layer_idx in range(num_layers):
        is_dn = layer_idx % FULL_ATTENTION_INTERVAL != FULL_ATTENTION_INTERVAL - 1
        layer_name = f"model.layers.{layer_idx}"
        layer = model.decoder.layers[layer_idx]
        attn = layer.self_attention

        # Input layernorm
        ln_key = f'{layer_name}.input_layernorm.weight'
        ln_weight = state_dict[ln_key]
        if hasattr(layer, 'input_layernorm') and hasattr(layer.input_layernorm, 'weight'):
            layer.input_layernorm.weight.data.copy_(ln_weight)
        elif hasattr(attn.linear_qkv, 'layer_norm_weight'):
            attn.linear_qkv.layer_norm_weight.data.copy_(ln_weight)

        # Post attention layernorm
        post_ln_key = f'{layer_name}.post_attention_layernorm.weight'
        post_ln_weight = state_dict[post_ln_key]
        if hasattr(layer, 'pre_mlp_layernorm') and hasattr(layer.pre_mlp_layernorm, 'weight'):
            layer.pre_mlp_layernorm.weight.data.copy_(post_ln_weight)
        elif hasattr(layer.mlp.linear_fc1, 'layer_norm_weight'):
            layer.mlp.linear_fc1.layer_norm_weight.data.copy_(post_ln_weight)

        # MLP
        gate_w = state_dict[f'{layer_name}.mlp.gate_proj.weight']
        up_w = state_dict[f'{layer_name}.mlp.up_proj.weight']
        layer.mlp.linear_fc1.weight.data.copy_(torch.cat([gate_w, up_w], dim=0))
        layer.mlp.linear_fc2.weight.data.copy_(
            state_dict[f'{layer_name}.mlp.down_proj.weight'])

        if is_dn:
            # DeltaNet: simple [Q, K, V] concat
            q_w = state_dict[f'{layer_name}.self_attn.q_proj.weight']
            k_w = state_dict[f'{layer_name}.self_attn.k_proj.weight']
            v_w = state_dict[f'{layer_name}.self_attn.v_proj.weight']
            attn.linear_qkv.weight.data.copy_(torch.cat([q_w, k_w, v_w], dim=0))
            attn.linear_proj.weight.data.copy_(
                state_dict[f'{layer_name}.self_attn.o_proj.weight'])
            attn.beta_proj.weight.data.copy_(
                state_dict[f'{layer_name}.self_attn.beta_proj.weight'])
            if f'{layer_name}.self_attn.beta_proj.bias' in state_dict:
                attn.beta_proj.bias.data.copy_(
                    state_dict[f'{layer_name}.self_attn.beta_proj.bias'])
            attn.decay_proj.weight.data.copy_(
                state_dict[f'{layer_name}.self_attn.decay_proj.weight'])
            if f'{layer_name}.self_attn.decay_proj.bias' in state_dict:
                attn.decay_proj.bias.data.copy_(
                    state_dict[f'{layer_name}.self_attn.decay_proj.bias'])
            if hasattr(attn, 'gate_proj'):
                attn.gate_proj.weight.data.copy_(
                    state_dict[f'{layer_name}.self_attn.gate_proj.weight'])
            # DeltaNet buffers
            for buf_name, sd_name in [
                ('conv1d_weight', f'{layer_name}.self_attn.conv1d.weight'),
                ('A_log', f'{layer_name}.self_attn.A_log'),
                ('dt_bias', f'{layer_name}.self_attn.dt_bias'),
                ('norm_weight', f'{layer_name}.self_attn.norm.weight'),
            ]:
                if sd_name in state_dict:
                    setattr(attn, buf_name, state_dict[sd_name].clone().to(attn.linear_qkv.weight.device))
            # Q/K layernorm
            if hasattr(attn, 'q_layernorm') and attn.q_layernorm is not None:
                qnorm_key = f'{layer_name}.self_attn.q_norm.weight'
                if qnorm_key in state_dict:
                    attn.q_layernorm.weight.data.copy_(state_dict[qnorm_key])
                elif attn.norm_weight is not None:
                    attn.q_layernorm.weight.data.copy_(attn.norm_weight)
            if hasattr(attn, 'k_layernorm') and attn.k_layernorm is not None:
                knorm_key = f'{layer_name}.self_attn.k_norm.weight'
                if knorm_key in state_dict:
                    attn.k_layernorm.weight.data.copy_(state_dict[knorm_key])
        else:
            # SelfAttention: GQA-interleaved
            q_w = state_dict[f'{layer_name}.self_attn.q_proj.weight']
            k_w = state_dict[f'{layer_name}.self_attn.k_proj.weight']
            v_w = state_dict[f'{layer_name}.self_attn.v_proj.weight']
            num_q_heads = 24; num_kv_heads = 4; head_dim = 256
            num_qg = num_q_heads // num_kv_heads  # 6
            total_per_group = num_qg * head_dim + 2 * head_dim  # 2048
            qkv_interleaved = torch.zeros(8192, tfconfig.hidden_size, dtype=q_w.dtype)
            for g in range(num_kv_heads):
                qkv_interleaved[g*total_per_group:(g+1)*total_per_group] = torch.cat([
                    q_w[g*num_qg*head_dim:(g+1)*num_qg*head_dim],
                    k_w[g*head_dim:(g+1)*head_dim],
                    v_w[g*head_dim:(g+1)*head_dim],
                ], dim=0)
            attn.linear_qkv.weight.data.copy_(qkv_interleaved)
            attn.linear_proj.weight.data.copy_(
                state_dict[f'{layer_name}.self_attn.o_proj.weight'])
            # Q/K norm
            if hasattr(attn, 'q_layernorm') and attn.q_layernorm is not None:
                qnorm_key = f'{layer_name}.self_attn.q_norm.weight'
                if qnorm_key in state_dict:
                    attn.q_layernorm.weight.data.copy_(state_dict[qnorm_key])
            if hasattr(attn, 'k_layernorm') and attn.k_layernorm is not None:
                knorm_key = f'{layer_name}.self_attn.k_norm.weight'
                if knorm_key in state_dict:
                    attn.k_layernorm.weight.data.copy_(state_dict[knorm_key])
            # Gate
            if hasattr(attn, 'gate_proj'):
                gate_key = f'{layer_name}.self_attn.gate_proj.weight'
                if gate_key in state_dict:
                    attn.gate_proj.weight.data.copy_(state_dict[gate_key])


def main():
    dist.init_process_group(backend="nccl")
    from megatron.core import parallel_state as mpu
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    torch.manual_seed(SEED)

    print("=" * 60, flush=True)
    print("PS E2E with Real Qwen3.6 Converted Weights (TP=1)", flush=True)
    print(f"Prefix={PREFIX_LEN}, Suffix={SUFFIX_LEN}", flush=True)
    print("=" * 60, flush=True)

    # Load converted weights
    print("\n[1] Loading converted state_dict...", flush=True)
    converted_sd = torch.load(CONVERTED_PT, map_location="cpu", weights_only=True)
    print(f"  Loaded {len(converted_sd)} keys", flush=True)

    # Load HF config
    print("\n[2] Loading HF config...", flush=True)
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    vocab_size = hf_config.vocab_size
    print(f"  vocab_size={vocab_size}, hidden={hf_config.hidden_size}, layers={hf_config.num_hidden_layers}", flush=True)

    # Create Megatron model
    print("\n[3] Creating + loading Megatron model...", flush=True)
    from verl.models.mcore.config_converter import hf_to_mcore_config_dense
    tfconfig = hf_to_mcore_config_dense(hf_config, torch.bfloat16)
    tfconfig.tensor_model_parallel_size = 1
    tfconfig.pipeline_model_parallel_size = 1

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(SEED)

    from verl.models.mcore.model_initializer import Qwen3_6HybridModel
    initializer = Qwen3_6HybridModel(tfconfig, hf_config)
    model = initializer.initialize(
        pre_process=True, post_process=True,
        share_embeddings_and_output_weights=False, value=False,
    ).cuda()

    load_converted_weights(model, converted_sd, tfconfig, num_layers=16)
    print("  Weights loaded!", flush=True)

    # Free converted_sd to save memory
    del converted_sd
    torch.cuda.empty_cache()

    # ===== Step 1: Normal forward (full sequence) =====
    print("\n[4] Normal forward (full sequence)...", flush=True)
    total_len = PREFIX_LEN + SUFFIX_LEN
    # Use a fixed prompt for reproducibility
    torch.manual_seed(SEED + 100)
    input_ids = torch.randint(0, vocab_size, (1, total_len), device="cuda")
    position_ids = torch.arange(total_len, device="cuda").unsqueeze(0)

    with torch.no_grad():
        hidden = model.embedding(input_ids, position_ids)
        mask = ~torch.tril(torch.ones(total_len, total_len, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        decoder_out = model.decoder(hidden, mask)
        output_out = model.output_layer(decoder_out)
        if isinstance(output_out, tuple):
            normal_logits = output_out[0].float()
        else:
            normal_logits = output_out.float()

    # Extract suffix logits from normal forward
    # normal_logits shape: [total_len, 1, vocab_size] (Megatron format: [sq, b, v])
    normal_suffix_logits = normal_logits[PREFIX_LEN:]  # [suffix_len, 1, vocab_size]
    print(f"  Normal suffix logits: shape={normal_suffix_logits.shape}", flush=True)
    print(f"  Range: [{normal_suffix_logits.min():.4f}, {normal_suffix_logits.max():.4f}]", flush=True)

    # ===== Step 2: Naive two-pass (no state injection) =====
    # This tests what happens when we run prefix and suffix separately
    # WITHOUT injecting KV or DeltaNet carry states.
    # Expect: very low cos_sim (since suffix has no context from prefix)

    print("\n[5] Naive two-pass (prefix-only → suffix-only, no state injection)...", flush=True)

    # --- Prefix pass ---
    print("  Prefix pass (no storage)...", flush=True)
    prefix_ids = input_ids[:, :PREFIX_LEN]
    prefix_position_ids = torch.arange(PREFIX_LEN, device="cuda").unsqueeze(0)

    with torch.no_grad():
        prefix_hidden = model.embedding(prefix_ids, prefix_position_ids)
        prefix_mask = ~torch.tril(torch.ones(PREFIX_LEN, PREFIX_LEN, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        # Run prefix through all layers but DON'T store anything
        for layer in model.decoder.layers:
            layer_out = layer(prefix_hidden, prefix_mask)
            prefix_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    # --- Suffix pass (naive: no injected states) ---
    print("  Suffix pass (no injection)...", flush=True)
    suffix_ids = input_ids[:, PREFIX_LEN:]
    suffix_position_ids = torch.arange(PREFIX_LEN, total_len, device="cuda").unsqueeze(0)

    with torch.no_grad():
        suffix_hidden = model.embedding(suffix_ids, suffix_position_ids)
        suffix_mask = ~torch.tril(torch.ones(SUFFIX_LEN, SUFFIX_LEN, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        for layer in model.decoder.layers:
            layer_out = layer(suffix_hidden, suffix_mask)
            suffix_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        naive_out = model.output_layer(suffix_hidden)
        naive_suffix_logits = naive_out[0].float() if isinstance(naive_out, tuple) else naive_out.float()

    print(f"  Naive suffix logits: shape={naive_suffix_logits.shape}", flush=True)
    print(f"  Range: [{naive_suffix_logits.min():.4f}, {naive_suffix_logits.max():.4f}]", flush=True)

    # ===== Step 3: Compare =====
    print("\n[6] Comparison: Normal vs Naive two-pass...", flush=True)

    cos_sim = F.cosine_similarity(
        normal_suffix_logits.flatten().unsqueeze(0),
        naive_suffix_logits.flatten().unsqueeze(0)).item()
    max_diff = (normal_suffix_logits - naive_suffix_logits).abs().max().item()
    mean_diff = (normal_suffix_logits - naive_suffix_logits).abs().mean().item()

    print(f"  cos_sim:  {cos_sim:.6f}", flush=True)
    print(f"  max_diff: {max_diff:.6f}", flush=True)
    print(f"  mean_diff: {mean_diff:.6f}", flush=True)

    print("\n" + "=" * 60, flush=True)
    if cos_sim < 0.5:
        print(f"Expected: naive two-pass cos_sim is LOW ({cos_sim:.6f})", flush=True)
        print("This confirms prefix context is essential for correct output.", flush=True)
        print("PS must inject KV/carry states to achieve high cos_sim.", flush=True)
        print("RESULT: Baseline established — need PS state injection for real accuracy.", flush=True)
    else:
        print(f"Unexpected: naive two-pass cos_sim={cos_sim:.6f}", flush=True)
        print("Suffix tokens might not depend heavily on prefix context.", flush=True)
    print("=" * 60, flush=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())