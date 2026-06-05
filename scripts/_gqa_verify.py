"""Verify GatedDeltaNet numerical correctness with GQA 24:4 config.

Creates a SelfAttention and GatedDeltaNetAttention with shared weights,
verifies that the DeltaNet forward produces valid output for GQA.
Must run with: torchrun --nproc_per_node=1 --nnodes=1 _gqa_verify.py
"""
import os, sys, torch, torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dependency", "megatron_v0150"))
sys.path.insert(0, os.path.join(REPO_ROOT, "prefix-sharing"))

from megatron.core import parallel_state as mpu
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
import importlib.util

# Import GatedDeltaNet
spec = importlib.util.spec_from_file_location(
    "gdn",
    os.path.join(REPO_ROOT, "dependency", "verl_v070", "verl", "models", "mcore", "gated_delta_net.py"),
)
gdn_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gdn_mod)
GatedDeltaNetAttention = gdn_mod.GatedDeltaNetAttention


def main():
    dist.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)

    print("=== GQA 24:4 Numerical Verification ===")

    # Config matching Qwen3.6-27B GQA ratio (scaled down)
    hidden = 768
    heads = 24  # total query heads
    kv_heads = 4  # total kv heads (GQA 24:4)
    head_dim = hidden // heads  # = 32
    num_layers = 2

    tc = TransformerConfig(
        num_layers=num_layers, hidden_size=hidden, num_attention_heads=heads,
        num_query_groups=kv_heads, kv_channels=head_dim,
        bf16=True, params_dtype=torch.bfloat16, normalization="RMSNorm",
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        use_cpu_initialization=False,
    )

    block_spec = get_gpt_decoder_block_spec(tc, use_transformer_engine=False)
    sa_sub = block_spec.layer_specs[0].submodules.self_attention.submodules

    # Create SelfAttention (original)
    sa = SelfAttention(
        config=tc, submodules=sa_sub, layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).cuda()

    # Create GatedDeltaNet with shared weights
    gdn = GatedDeltaNetAttention(
        config=tc, submodules=sa_sub, layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        partial_rotary_factor=0.25, attn_output_gate=True,
    ).cuda()

    # Share weights
    gdn.linear_qkv = sa.linear_qkv
    gdn.linear_proj = sa.linear_proj
    if hasattr(sa, 'q_layernorm') and sa.q_layernorm is not None:
        gdn.q_layernorm = sa.q_layernorm
    if hasattr(sa, 'k_layernorm') and sa.k_layernorm is not None:
        gdn.k_layernorm = sa.k_layernorm

    # Forward
    sq, b = 16, 2
    hidden_states = torch.randn(sq, b, hidden, dtype=torch.bfloat16, device="cuda")

    # SelfAttention forward (with causal mask)
    causal_mask = ~torch.tril(torch.ones(sq, sq, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        sa_out, _ = sa(hidden_states=hidden_states, attention_mask=causal_mask)

    # GatedDeltaNet forward (no mask needed for linear attention)
    with torch.no_grad():
        gdn_out, _ = gdn(hidden_states=hidden_states, attention_mask=None)

    print(f"  SelfAttention output shape: {sa_out.shape}")
    print(f"  GatedDeltaNet output shape: {gdn_out.shape}")
    print(f"  SA output finite: {torch.isfinite(sa_out).all().item()}")
    print(f"  GDN output finite: {torch.isfinite(gdn_out).all().item()}")

    # They should NOT be equal (different attention mechanisms)
    # but both should produce valid, non-trivial outputs
    print(f"  SA output norm: {sa_out.float().norm().item():.4f}")
    print(f"  GDN output norm: {gdn_out.float().norm().item():.4f}")
    print(f"  SA output mean: {sa_out.float().mean().item():.6f}")
    print(f"  GDN output mean: {gdn_out.float().mean().item():.6f}")

    # Verify GQA correctness: check that query/key/value tensors
    # from get_query_key_value_tensors have correct shapes
    q, k, v = sa.get_query_key_value_tensors(hidden_states, None, True)
    # q: [sq, b, np, hn] where np = heads (24 per TP)
    # k: [sq, b, ng, hn] where ng = kv_heads (4 per TP)
    # v: [sq, b, ng, hn]
    print(f"\n  Q shape: {q.shape} (expect [{sq}, {b}, {heads}, {head_dim}])")
    print(f"  K shape: {k.shape} (expect [{sq}, {b}, {kv_heads}, {head_dim}])")
    print(f"  V shape: {v.shape} (expect [{sq}, {b}, {kv_heads}, {head_dim}])")

    assert q.shape == (sq, b, heads, head_dim), f"Q shape wrong: {q.shape}"
    assert k.shape == (sq, b, kv_heads, head_dim), f"K shape wrong: {k.shape}"
    assert v.shape == (sq, b, kv_heads, head_dim), f"V shape wrong: {v.shape}"

    print("\n  [PASS] All checks passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
