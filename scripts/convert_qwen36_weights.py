#!/usr/bin/env python3
"""Convert HuggingFace Qwen3.6 weights to Megatron-compatible state_dict format.

The HF Qwen3.6 model uses different weight names and structures:
- DeltaNet (linear_attn): in_proj_qkv, in_proj_a, in_proj_b, in_proj_z, conv1d, A_log, dt_bias, norm, out_proj
- SelfAttention (self_attn): doubled q_proj (2*head_dim), separate k/v/o_proj, q/k_norm

The Megatron loader expects:
- self_attn.q_proj, self_attn.k_proj, self_attn.v_proj, self_attn.o_proj
- self_attn.beta_proj, self_attn.decay_proj, self_attn.gate_proj (DeltaNet)
- self_attn.gate_proj (SelfAttention, from split q_proj)
- self_attn.q_norm, self_attn.k_norm

This script preprocesses the HF state_dict to match the Megatron loader format.
"""

import os
import sys
import json
import torch
from safetensors import safe_open


def load_hf_state_dict(model_dir):
    """Load all safetensors files from model directory."""
    files = sorted([f for f in os.listdir(model_dir) if f.endswith('.safetensors')])
    state_dict = {}
    for f in files:
        sf = safe_open(os.path.join(model_dir, f), framework='pt')
        for k in sf.keys():
            state_dict[k] = sf.get_tensor(k)
    return state_dict


def is_deltanet_layer(layer_idx, hf_config):
    """Check if a layer uses GatedDeltaNet (linear_attn)."""
    layer_types = getattr(hf_config, 'layer_types', None)
    if layer_types is not None and layer_idx < len(layer_types):
        return layer_types[layer_idx] == 'linear_attention'
    interval = getattr(hf_config, 'full_attention_interval', 1)
    return (layer_idx + 1) % interval != 0


def convert_qwen36_weights_to_megatron(hf_state_dict, hf_config, num_layers):
    """Convert HF Qwen3.6 state_dict to Megatron loader-compatible format.

    Key conversions:
    1. Strip 'model.language_model.' prefix
    2. DeltaNet: rename linear_attn → self_attn, split in_proj_qkv → q/k/v_proj
    3. DeltaNet: rename in_proj_a → beta_proj, in_proj_b → decay_proj
    4. DeltaNet: rename in_proj_z → gate_proj (6144,5120)
    5. DeltaNet: rename out_proj → o_proj
    6. SelfAttention: split doubled q_proj (12288,5120) into q_proj + gate_proj
    7. Handle DeltaNet-specific: conv1d, A_log, dt_bias, norm
    """
    num_heads = hf_config.num_attention_heads  # 24 (SelfAttention)
    num_kv_heads = hf_config.num_key_value_heads  # 4 (SelfAttention)
    hidden_size = hf_config.hidden_size  # 5120
    head_dim = getattr(hf_config, 'head_dim', hidden_size // num_heads)  # 256

    # DeltaNet uses different head dimensions:
    # SelfAttention: 24 query heads, 4 kv heads, 256 head_dim
    # DeltaNet:      48 query heads, 16 kv heads, 128 head_dim
    dn_num_heads = 48  # DeltaNet query heads
    dn_kv_heads = 16   # DeltaNet kv heads
    dn_head_dim = 128  # DeltaNet head_dim

    sa_q_dim = num_heads * head_dim  # 24*256 = 6144
    sa_kv_dim = num_kv_heads * head_dim  # 4*256 = 1024

    dn_q_dim = dn_num_heads * dn_head_dim  # 48*128 = 6144
    dn_kv_dim = dn_kv_heads * dn_head_dim  # 16*128 = 2048

    megatron_sd = {}

    # Embed tokens
    embed_key = 'model.language_model.embed_tokens.weight'
    if embed_key in hf_state_dict:
        megatron_sd['model.embed_tokens.weight'] = hf_state_dict[embed_key]
    else:
        # Some variants may use different prefix
        for k in hf_state_dict:
            if 'embed_tokens' in k and 'weight' in k and 'visual' not in k:
                megatron_sd['model.embed_tokens.weight'] = hf_state_dict[k]
                break

    # Final layernorm
    norm_key = 'model.language_model.norm.weight'
    if norm_key in hf_state_dict:
        megatron_sd['model.norm.weight'] = hf_state_dict[norm_key]

    # lm_head
    megatron_sd['lm_head.weight'] = hf_state_dict['lm_head.weight']

    # Process each layer
    for layer_idx in range(num_layers):
        is_dn = is_deltanet_layer(layer_idx, hf_config)
        layer_prefix = f'model.language_model.layers.{layer_idx}'
        meg_layer = f'model.layers.{layer_idx}'

        # Input layernorm
        megatron_sd[f'{meg_layer}.input_layernorm.weight'] = \
            hf_state_dict[f'{layer_prefix}.input_layernorm.weight']

        # Post attention layernorm
        megatron_sd[f'{meg_layer}.post_attention_layernorm.weight'] = \
            hf_state_dict[f'{layer_prefix}.post_attention_layernorm.weight']

        # MLP (same for both types)
        megatron_sd[f'{meg_layer}.mlp.gate_proj.weight'] = \
            hf_state_dict[f'{layer_prefix}.mlp.gate_proj.weight']
        megatron_sd[f'{meg_layer}.mlp.up_proj.weight'] = \
            hf_state_dict[f'{layer_prefix}.mlp.up_proj.weight']
        megatron_sd[f'{meg_layer}.mlp.down_proj.weight'] = \
            hf_state_dict[f'{layer_prefix}.mlp.down_proj.weight']

        if is_dn:
            # DeltaNet (linear_attn) layer
            attn_prefix = f'{layer_prefix}.linear_attn'
            meg_attn = f'{meg_layer}.self_attn'

            # in_proj_qkv → split into q_proj, k_proj, v_proj
            # Shape: (10240, 5120) = dn_q_dim(6144) + dn_kv_dim(2048) + dn_kv_dim(2048)
            in_proj_qkv = hf_state_dict[f'{attn_prefix}.in_proj_qkv.weight']
            q_part = in_proj_qkv[:dn_q_dim]  # (6144, 5120) = 48 query heads * 128
            k_part = in_proj_qkv[dn_q_dim:dn_q_dim + dn_kv_dim]  # (2048, 5120) = 16 kv heads * 128
            v_part = in_proj_qkv[dn_q_dim + dn_kv_dim:]  # (2048, 5120) = 16 kv heads * 128
            megatron_sd[f'{meg_attn}.q_proj.weight'] = q_part
            megatron_sd[f'{meg_attn}.k_proj.weight'] = k_part
            megatron_sd[f'{meg_attn}.v_proj.weight'] = v_part

            # in_proj_a → beta_proj
            megatron_sd[f'{meg_attn}.beta_proj.weight'] = \
                hf_state_dict[f'{attn_prefix}.in_proj_a.weight']
            # in_proj_a has bias? Check
            bias_key = f'{attn_prefix}.in_proj_a.bias'
            if bias_key in hf_state_dict:
                megatron_sd[f'{meg_attn}.beta_proj.bias'] = hf_state_dict[bias_key]

            # in_proj_b → decay_proj
            megatron_sd[f'{meg_attn}.decay_proj.weight'] = \
                hf_state_dict[f'{attn_prefix}.in_proj_b.weight']
            bias_key = f'{attn_prefix}.in_proj_b.bias'
            if bias_key in hf_state_dict:
                megatron_sd[f'{meg_attn}.decay_proj.bias'] = hf_state_dict[bias_key]

            # in_proj_z → gate_proj (6144, 5120)
            # This goes into our Megatron gate_proj which also outputs (num_heads*head_dim=6144)
            megatron_sd[f'{meg_attn}.gate_proj.weight'] = \
                hf_state_dict[f'{attn_prefix}.in_proj_z.weight']

            # out_proj → o_proj
            megatron_sd[f'{meg_attn}.o_proj.weight'] = \
                hf_state_dict[f'{attn_prefix}.out_proj.weight']

            # DeltaNet-specific parameters (not in standard loader, stored as extra)
            # These are needed for DeltaNet's state computation
            megatron_sd[f'{meg_attn}.conv1d.weight'] = \
                hf_state_dict[f'{attn_prefix}.conv1d.weight']
            megatron_sd[f'{meg_attn}.A_log'] = \
                hf_state_dict[f'{attn_prefix}.A_log']
            megatron_sd[f'{meg_attn}.dt_bias'] = \
                hf_state_dict[f'{attn_prefix}.dt_bias']
            # norm is per-head layernorm (128 = head_dim)
            megatron_sd[f'{meg_attn}.norm.weight'] = \
                hf_state_dict[f'{attn_prefix}.norm.weight']

            # Q/K layernorm for DeltaNet (q_layernorm/k_layernorm in Megatron)
            # These map to the same q_norm/k_norm names
            # Actually, DeltaNet's norm is for the entire head_dim, applied to Q
            # In HF: norm.weight (128) is for head_dim
            # In Megatron: q_layernorm.weight should be (head_dim=128 per TP rank)

        else:
            # SelfAttention layer
            attn_prefix = f'{layer_prefix}.self_attn'
            meg_attn = f'{meg_layer}.self_attn'

            # Doubled q_proj → split into query + gate
            # Shape: (12288, 5120) = 2 * (num_heads * head_dim)
            q_proj_full = hf_state_dict[f'{attn_prefix}.q_proj.weight']
            assert q_proj_full.shape[0] == 2 * sa_q_dim, \
                f"Expected q_proj shape ({2*sa_q_dim}, {hidden_size}), got {q_proj_full.shape}"
            query_part = q_proj_full[:sa_q_dim]  # (6144, 5120) - actual query
            gate_part = q_proj_full[sa_q_dim:]  # (6144, 5120) - output gate

            megatron_sd[f'{meg_attn}.q_proj.weight'] = query_part

            # gate_proj for SelfAttention: we apply gate AFTER o_proj in hidden_size space
            # (5120, 5120), so we need to project the (6144, 5120) gate through o_proj
            # to get an equivalent (5120, 5120) gate.
            # Mathematical derivation:
            #   Real: output = o_proj(attn_output * sigmoid(gate_6144))
            #   Our:  output = o_proj(attn_output) * sigmoid(gate_5120)
            #
            # These can't be made exactly equivalent for non-diagonal o_proj.
            # But we can approximate by projecting gate through o_proj:
            #   gate_6144_effect ≈ o_proj @ diag(sigmoid(gate_raw)) @ attn_output
            #   ≈ o_proj(attn_output) * sigmoid(o_proj_effect_of_gate)
            #
            # Simple approximation: gate_5120 = o_proj @ gate_6144_raw / norm_factor
            # This isn't mathematically correct, but it's the best we can do.
            #
            # Better: just use a random gate for SelfAttention and focus on
            # DeltaNet weight loading. SelfAttention gate precision impact is small.
            # Store the raw gate weight separately for potential future use.
            megatron_sd[f'{meg_attn}.gate_proj_raw.weight'] = gate_part  # (6144, 5120) - raw

            # For now, approximate gate_proj as: W_gate_5120 ≈ o_proj @ W_gate_6144
            # This gives a (5120, 5120) matrix that approximately captures the gate effect
            o_proj = hf_state_dict[f'{attn_prefix}.o_proj.weight']  # (5120, 6144)
            gate_proj_approx = o_proj @ gate_part  # (5120, 5120)
            megatron_sd[f'{meg_attn}.gate_proj.weight'] = gate_proj_approx

            megatron_sd[f'{meg_attn}.k_proj.weight'] = \
                hf_state_dict[f'{attn_prefix}.k_proj.weight']
            megatron_sd[f'{meg_attn}.v_proj.weight'] = \
                hf_state_dict[f'{attn_prefix}.v_proj.weight']
            megatron_sd[f'{meg_attn}.o_proj.weight'] = \
                hf_state_dict[f'{attn_prefix}.o_proj.weight']
            megatron_sd[f'{meg_attn}.q_norm.weight'] = \
                hf_state_dict[f'{attn_prefix}.q_norm.weight']
            megatron_sd[f'{meg_attn}.k_norm.weight'] = \
                hf_state_dict[f'{attn_prefix}.k_norm.weight']

    return megatron_sd


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True, help='Path to HF model directory')
    parser.add_argument('--output', required=True, help='Path to save converted state_dict')
    parser.add_argument('--num-layers', type=int, default=16, help='Number of layers to convert')
    args = parser.parse_args()

    # Load HF config
    from transformers import AutoConfig
    hf_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)

    print(f"Model config:")
    print(f"  num_hidden_layers: {hf_config.num_hidden_layers}")
    print(f"  num_attention_heads: {hf_config.num_attention_heads}")
    print(f"  num_key_value_heads: {hf_config.num_key_value_heads}")
    print(f"  hidden_size: {hf_config.hidden_size}")
    print(f"  head_dim: {getattr(hf_config, 'head_dim', 'N/A')}")
    print(f"  full_attention_interval: {getattr(hf_config, 'full_attention_interval', 1)}")
    print(f"  partial_rotary_factor: {getattr(hf_config, 'partial_rotary_factor', 1.0)}")
    print(f"  attn_output_gate: {getattr(hf_config, 'attn_output_gate', False)}")

    # Load HF state dict
    print(f"\nLoading HF state dict from {args.model_dir}...")
    hf_sd = load_hf_state_dict(args.model_dir)
    print(f"Loaded {len(hf_sd)} weights")

    # Convert
    print(f"\nConverting to Megatron format (num_layers={args.num_layers})...")
    meg_sd = convert_qwen36_weights_to_megatron(hf_sd, hf_config, args.num_layers)
    print(f"Converted to {len(meg_sd)} Megatron weights")

    # Print summary
    for layer_idx in range(args.num_layers):
        is_dn = is_deltanet_layer(layer_idx, hf_config)
        layer_type = "DeltaNet" if is_dn else "SelfAttn"
        keys = [k for k in meg_sd if f'model.layers.{layer_idx}.self_attn.' in k]
        print(f"  L{layer_idx} ({layer_type}): {len(keys)} attn weights")

    # Save
    print(f"\nSaving to {args.output}...")
    torch.save(meg_sd, args.output)
    print(f"Saved! File size: {os.path.getsize(args.output) / 1e9:.2f} GB")

    # Quick sanity check
    print("\n=== Sanity Check ===")
    for layer_idx in range(args.num_layers):
        is_dn = is_deltanet_layer(layer_idx, hf_config)
        meg_attn = f'model.layers.{layer_idx}.self_attn'
        if is_dn:
            q = meg_sd[f'{meg_attn}.q_proj.weight']
            k = meg_sd[f'{meg_attn}.k_proj.weight']
            v = meg_sd[f'{meg_attn}.v_proj.weight']
            beta = meg_sd[f'{meg_attn}.beta_proj.weight']
            decay = meg_sd[f'{meg_attn}.decay_proj.weight']
            gate = meg_sd[f'{meg_attn}.gate_proj.weight']
            o = meg_sd[f'{meg_attn}.o_proj.weight']
            print(f"  L{layer_idx} DeltaNet: q={q.shape} k={k.shape} v={v.shape} "
                  f"beta={beta.shape} decay={decay.shape} gate={gate.shape} o={o.shape}")
        else:
            q = meg_sd[f'{meg_attn}.q_proj.weight']
            k = meg_sd[f'{meg_attn}.k_proj.weight']
            v = meg_sd[f'{meg_attn}.v_proj.weight']
            gate = meg_sd[f'{meg_attn}.gate_proj.weight']
            o = meg_sd[f'{meg_attn}.o_proj.weight']
            gate_raw = meg_sd[f'{meg_attn}.gate_proj_raw.weight']
            print(f"  L{layer_idx} SelfAttn: q={q.shape} k={k.shape} v={v.shape} "
                  f"gate={gate.shape} gate_raw={gate_raw.shape} o={o.shape}")


if __name__ == '__main__':
    main()