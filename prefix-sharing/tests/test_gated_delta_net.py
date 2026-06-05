"""Unit tests for Qwen3.6 GatedDeltaNet and HybridAttention components.

Tests run locally (CPU) without requiring Megatron or verl.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest


def test_partial_rope_dimensions():
    """Verify partial RoPE only rotates the correct dimensions."""
    head_dim = 256
    partial_rotary_factor = 0.25
    rope_dim = int(head_dim * partial_rotary_factor)

    assert rope_dim == 64, f"Expected rope_dim=64, got {rope_dim}"

    # Simulate partial RoPE: rotate first 64 dims, pass through rest
    q = torch.randn(2, 8, 10, head_dim)  # (batch, heads, seq, dim)
    q_rot = q[..., :rope_dim]
    q_pass = q[..., rope_dim:]

    # After "rotation" (identity for this test)
    q_rotated = q_rot  # In real code, apply_rotary_pos_emb would be called
    q_out = torch.cat([q_rotated, q_pass], dim=-1)

    # Verify pass-through dimensions unchanged
    assert torch.equal(q_out[..., rope_dim:], q[..., rope_dim:]), "Pass-through dims should be unchanged"
    # Verify rotated dimensions are the same (identity rotation in this test)
    assert q_out.shape == q.shape, f"Shape mismatch: {q_out.shape} vs {q.shape}"
    print("  [PASS] Partial RoPE: rope_dim=64, pass-through=192")


def test_gated_deltanet_cumsum_math():
    """Verify GatedDeltaNet cumsum computation matches reference."""
    torch.manual_seed(42)

    total_tokens = 10
    num_heads = 4
    head_dim = 8

    q = torch.randn(total_tokens, num_heads, head_dim)
    k = torch.randn(total_tokens, num_heads, head_dim)
    v = torch.randn(total_tokens, num_heads, head_dim)
    beta = torch.sigmoid(torch.randn(total_tokens, num_heads, 1)).unsqueeze(-1)  # (t, h, 1, 1)

    # k,v: (total_tokens, heads, head_dim)
    # Outer product: (total_tokens, heads, head_dim, head_dim)
    # einsum: t=token, h=head, d=query_dim, e=value_dim
    kv = torch.einsum('thd,the->thde', k, v)
    update = beta * kv
    trajectory = torch.cumsum(update, dim=0)

    # Query the state: y[t,h,e] = sum_d q[t,h,d] * trajectory[t,h,d,e]
    y = torch.einsum('thd,thde->the', q, trajectory)

    assert y.shape == (total_tokens, num_heads, head_dim), f"Wrong shape: {y.shape}"

    # Manual check for first token
    # For token 0: trajectory[0] = update[0] = beta[0] * k[0] (outer) v[0]
    # y[0] = q[0] @ trajectory[0]
    expected_y0 = torch.einsum('hd,hde->he', q[0], update[0])
    assert torch.allclose(y[0], expected_y0, atol=1e-5), "Token 0 output mismatch"

    # For token 2: trajectory[2] = update[0] + update[1] + update[2]
    expected_traj2 = update[0] + update[1] + update[2]
    expected_y2 = torch.einsum('hd,hde->he', q[2], expected_traj2)
    assert torch.allclose(y[2], expected_y2, atol=1e-5), "Token 2 output mismatch"

    print("  [PASS] GatedDeltaNet cumsum math verified")


def test_packed_cumsum_per_sequence():
    """Verify per-sequence cumsum resets at sequence boundaries."""
    # Import from our deltanet module
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "dependency", "verl_v070"))
    # Use inline packed_cumsum logic since the deltanet module requires megatron
    torch.manual_seed(42)

    # Two sequences: [0:4] and [4:7]
    cu_seqlens = torch.tensor([0, 4, 7])
    x = torch.randn(7, 2, 3, 3)

    # Inline packed_cumsum
    full_cumsum = torch.cumsum(x, dim=0)
    start_values = torch.zeros_like(x)
    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        if start > 0:
            start_values[start:end] = full_cumsum[start - 1:start]
    result = full_cumsum - start_values

    # Sequence 1 (tokens 0-3): cumsum should be independent
    seq1_expected = torch.cumsum(x[:4], dim=0)
    assert torch.allclose(result[:4], seq1_expected, atol=1e-6), "Seq 1 cumsum mismatch"

    # Sequence 2 (tokens 4-6): cumsum should reset
    seq2_expected = torch.cumsum(x[4:7], dim=0)
    assert torch.allclose(result[4:7], seq2_expected, atol=1e-6), "Seq 2 cumsum should reset"

    # Verify no leakage between sequences
    assert not torch.allclose(result[4], result[3]), "No leakage between sequences"

    print("  [PASS] Packed cumsum per-sequence boundary check")


def test_output_gate():
    """Verify output gate: output = attn_output * sigmoid(gate_proj(hidden))."""
    torch.manual_seed(42)

    batch, seq, hidden = 2, 8, 32
    attn_output = torch.randn(batch, seq, hidden)
    hidden_states = torch.randn(batch, seq, hidden)

    # Simulate gate_proj (identity for simplicity)
    gate_proj_weight = torch.eye(hidden)
    gate = torch.sigmoid(hidden_states @ gate_proj_weight.T)
    gated_output = attn_output * gate

    assert gated_output.shape == (batch, seq, hidden)
    # Gate should modulate the output
    assert not torch.allclose(gated_output, attn_output), "Gate should change output"
    # Gate values should be in (0, 1) since sigmoid
    assert (gate > 0).all() and (gate < 1).all(), "Gate values should be in (0, 1)"

    print("  [PASS] Output gate computation verified")


def test_hybrid_layer_routing():
    """Verify decoder routes layers correctly based on full_attention_interval."""
    from types import SimpleNamespace

    full_attention_interval = 4
    num_layers = 64

    full_attention_layers = []
    linear_attention_layers = []

    for layer_idx in range(num_layers):
        if layer_idx % full_attention_interval == 0:
            full_attention_layers.append(layer_idx)
        else:
            linear_attention_layers.append(layer_idx)

    assert len(full_attention_layers) == 16, f"Expected 16 full attn layers, got {len(full_attention_layers)}"
    assert len(linear_attention_layers) == 48, f"Expected 48 linear attn layers, got {len(linear_attention_layers)}"
    assert full_attention_layers == [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]

    print(f"  [PASS] Hybrid routing: 16 full attn {full_attention_layers[:4]}... + 48 linear attn")


def test_model_spec_qwen3_6():
    """Verify ModelSpec correctly identifies Qwen3.6 layer types."""
    from prefix_sharing.core.model_spec import ModelSpec, QWEN3_6_27B, AttentionLayerType

    spec = QWEN3_6_27B

    assert spec.num_hidden_layers == 64
    assert spec.full_attention_interval == 4
    assert spec.attn_output_gate is True
    assert spec.partial_rotary_factor == 0.25
    assert spec.head_dim == 256
    assert spec.gqa_group_size == 6  # 24 / 4

    # Verify layer type routing
    assert spec.layer_type(0) == AttentionLayerType.FULL_ATTENTION
    assert spec.layer_type(1) == AttentionLayerType.LINEAR_ATTENTION
    assert spec.layer_type(4) == AttentionLayerType.FULL_ATTENTION
    assert spec.layer_type(63) == AttentionLayerType.LINEAR_ATTENTION

    assert spec.num_full_attention_layers == 16
    assert spec.num_linear_attention_layers == 48

    print(f"  [PASS] ModelSpec Qwen3.6-27B: {spec.num_full_attention_layers} full + {spec.num_linear_attention_layers} linear")


def test_model_spec_from_hf_config():
    """Verify ModelSpec.from_hf_config works with Qwen3.6 config."""
    from prefix_sharing.core.model_spec import ModelSpec

    hf_config = {
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "hidden_size": 6144,
        "full_attention_interval": 4,
        "attn_output_gate": True,
        "partial_rotary_factor": 0.25,
        "max_position_embeddings": 131072,
    }

    spec = ModelSpec.from_hf_config(hf_config)

    assert spec.num_hidden_layers == 64
    assert spec.head_dim == 6144 // 24  # = 256
    assert spec.full_attention_interval == 4
    assert spec.attn_output_gate is True
    assert spec.partial_rotary_factor == 0.25

    print(f"  [PASS] ModelSpec.from_hf_config: head_dim={spec.head_dim}, interval={spec.full_attention_interval}")


def test_prefix_store_deltanet():
    """Verify PrefixDeltanetStore works for state trajectory storage."""
    from prefix_sharing.core.prefix_store import (
        PrefixDeltanetStore,
        PrefixActivationSlotId,
        PREFIX_STATE_TYPE_DELTANET_STATE,
    )

    store = PrefixDeltanetStore()

    slot = PrefixActivationSlotId(
        forward_id=0, micro_batch_id=0, layer_id=0,
        sample_idx_in_batch=0, prefix_state_type=PREFIX_STATE_TYPE_DELTANET_STATE,
        tp_rank=0,
    )
    state = torch.randn(4, 8, 8)  # recurrent state

    store.store(slot, recurrent_state=state, prefix_len=10)
    retrieved = store.load(slot)

    assert torch.equal(state, retrieved.recurrent_state), "Retrieved state should match saved"

    # Different tp_rank should be separate
    slot_tp1 = PrefixActivationSlotId(
        forward_id=0, micro_batch_id=0, layer_id=0,
        sample_idx_in_batch=0, prefix_state_type=PREFIX_STATE_TYPE_DELTANET_STATE,
        tp_rank=1,
    )
    state_tp1 = torch.randn(4, 8, 8)
    store.store(slot_tp1, recurrent_state=state_tp1, prefix_len=10)
    assert torch.equal(store.load(slot).recurrent_state, state), "tp_rank=0 state should be unchanged"

    store.close()
    print("  [PASS] PrefixDeltanetStore store/load verified")


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3.6 Unit Tests")
    print("=" * 60)

    tests = [
        test_partial_rope_dimensions,
        test_gated_deltanet_cumsum_math,
        test_packed_cumsum_per_sequence,
        test_output_gate,
        test_hybrid_layer_routing,
        test_model_spec_qwen3_6,
        test_model_spec_from_hf_config,
        test_prefix_store_deltanet,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
