"""Tensor parallel test for HybridAttention model.

Verifies GatedDeltaNet and hybrid model work correctly with TP=2.

Must run with: torchrun --nproc_per_node=2 --nnodes=1 gpu_tp_test.py
"""
import os, sys, torch, torch.distributed as dist, importlib.util, time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(REPO_ROOT, "dependency")
sys.path.insert(0, os.path.join(DEPS, "megatron_v0150"))
sys.path.insert(0, os.path.join(REPO_ROOT, "prefix-sharing"))

from megatron.core import parallel_state as mpu
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


def import_mcore_module(name):
    path = os.path.join(DEPS, "verl_v070", "verl", "models", "mcore", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"verl.mcore.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GatedDeltaNetAttention = import_mcore_module("gated_delta_net").GatedDeltaNetAttention

TP_SIZE = 2


def make_tfconfig(hidden=512, num_layers=4, heads=8, kv_heads=4, tp=TP_SIZE):
    return TransformerConfig(
        num_layers=num_layers, hidden_size=hidden,
        num_attention_heads=heads, num_query_groups=kv_heads,
        kv_channels=hidden // heads,
        bf16=True, params_dtype=torch.bfloat16,
        normalization="RMSNorm", init_method_std=0.02,
        hidden_dropout=0.0, attention_dropout=0.0,
        tensor_model_parallel_size=tp, pipeline_model_parallel_size=1,
        use_cpu_initialization=False,
    )


def build_hybrid_model(tfconfig, vocab_size, max_seq_len, interval, partial_rot=0.25, gate=True):
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_sub = block_spec.layer_specs[0].submodules.self_attention.submodules

    model = GPTModel(
        config=tfconfig, transformer_layer_spec=block_spec,
        vocab_size=vocab_size, max_sequence_length=max_seq_len,
        pre_process=True, post_process=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope", rotary_base=10000.0,
    )
    if interval <= 1:
        return model

    for i, layer in enumerate(model.decoder.layers):
        if i % interval != 0:
            old = layer.self_attention
            new = GatedDeltaNetAttention(
                config=tfconfig, submodules=sa_sub,
                layer_number=old.layer_number, attn_mask_type=old.attn_mask_type,
                partial_rotary_factor=partial_rot, attn_output_gate=gate,
            )
            new.to(next(old.parameters()).device)
            new.linear_qkv = old.linear_qkv
            new.linear_proj = old.linear_proj
            if hasattr(old, 'q_layernorm') and old.q_layernorm is not None:
                new.q_layernorm = old.q_layernorm
            if hasattr(old, 'k_layernorm') and old.k_layernorm is not None:
                new.k_layernorm = old.k_layernorm
            layer.self_attention = new
    return model


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size == TP_SIZE, f"Expected {TP_SIZE} processes, got {world_size}"

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=TP_SIZE,
        pipeline_model_parallel_size=1,
    )
    torch.manual_seed(42 + rank)
    model_parallel_cuda_manual_seed(42 + rank)
    torch.cuda.set_device(rank)

    tp_rank = mpu.get_tensor_model_parallel_rank()

    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            if rank == 0:
                print(f"  [PASS] {name}")
        else:
            failed += 1
            if rank == 0:
                print(f"  [FAIL] {name} {detail}")

    if rank == 0:
        print("=" * 60)
        print(f"TP={TP_SIZE} HybridAttention Test")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    # Test 1: Hybrid model with TP=2
    if rank == 0:
        print("\n--- Test 1: Hybrid Model TP=2 ---")

    try:
        # heads=8, kv_heads=4 → each TP rank gets 4 query heads, 2 kv heads
        tfconfig = make_tfconfig(hidden=512, num_layers=4, heads=8, kv_heads=4)

        model = build_hybrid_model(
            tfconfig, vocab_size=32000, max_seq_len=512,
            interval=2, partial_rot=0.5, gate=True,
        ).cuda().bfloat16()

        check("TP model created", True)

        # Verify layer routing
        gdn_layers = [i for i, l in enumerate(model.decoder.layers)
                      if isinstance(l.self_attention, GatedDeltaNetAttention)]
        full_layers = [i for i, l in enumerate(model.decoder.layers)
                       if isinstance(l.self_attention, SelfAttention)
                       and not isinstance(l.self_attention, GatedDeltaNetAttention)]
        check("Full attn [0,2]", full_layers == [0, 2], f"got {full_layers}")
        check("Linear attn [1,3]", gdn_layers == [1, 3], f"got {gdn_layers}")

        # Verify TP head partitioning
        gdn_attn = model.decoder.layers[1].self_attention
        check("GDN num_heads_per_tp", gdn_attn.num_heads_per_tp == 4,
              f"got {gdn_attn.num_heads_per_tp}")
        check("GDN num_kv_heads_per_tp", gdn_attn.num_kv_heads_per_tp == 2,
              f"got {gdn_attn.num_kv_heads_per_tp}")
        check("GDN beta_proj exists", hasattr(gdn_attn, 'beta_proj'))
        check("GDN gate_proj exists", hasattr(gdn_attn, 'gate_proj'))

        # Verify beta_proj weight shape is TP-sharded
        beta_weight_shape = gdn_attn.beta_proj.weight.shape[0]
        expected_per_rank = tfconfig.num_attention_heads // TP_SIZE
        check("beta_proj TP sharded", beta_weight_shape == expected_per_rank,
              f"got {beta_weight_shape}, expected {expected_per_rank}")

        # Forward pass
        bsz, seq_len = 2, 16
        input_ids = torch.randint(0, 32000, (bsz, seq_len), device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

        with torch.no_grad():
            hidden = model.embedding(input_ids, position_ids)
            mask = ~torch.tril(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            out = model.decoder(hidden, mask)
        check("Forward finite", torch.isfinite(out).all().item())
        check("Forward shape", out.shape[0] == seq_len and out.shape[1] == bsz,
              f"got {out.shape}")

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        check("TP hybrid model", False, str(e))
        import traceback; traceback.print_exc()

    # Test 2: Dense model with TP=2 (no hybrid)
    if rank == 0:
        print("\n--- Test 2: Dense Model TP=2 ---")

    try:
        tfconfig = make_tfconfig(hidden=512, num_layers=2, heads=8, kv_heads=4)
        model = build_hybrid_model(
            tfconfig, vocab_size=32000, max_seq_len=512,
            interval=1,  # No hybrid
        ).cuda().bfloat16()

        gdn_layers = [i for i, l in enumerate(model.decoder.layers)
                      if isinstance(l.self_attention, GatedDeltaNetAttention)]
        check("No GDN layers", gdn_layers == [], f"got {gdn_layers}")

        bsz, seq_len = 2, 16
        input_ids = torch.randint(0, 32000, (bsz, seq_len), device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

        with torch.no_grad():
            hidden = model.embedding(input_ids, position_ids)
            mask = ~torch.tril(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            out = model.decoder(hidden, mask)
        check("Dense forward finite", torch.isfinite(out).all().item())

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        check("TP dense model", False, str(e))
        import traceback; traceback.print_exc()

    # Test 3: Gradient flow with TP=2
    if rank == 0:
        print("\n--- Test 3: Gradient Flow TP=2 ---")

    try:
        tfconfig = make_tfconfig(hidden=512, num_layers=4, heads=8, kv_heads=4)
        model = build_hybrid_model(
            tfconfig, vocab_size=32000, max_seq_len=512,
            interval=2, partial_rot=0.5, gate=True,
        ).cuda().bfloat16()

        bsz, seq_len = 2, 8
        input_ids = torch.randint(0, 32000, (bsz, seq_len), device="cuda")
        position_ids = torch.arange(8, device="cuda").unsqueeze(0).expand(bsz, -1)

        hidden = model.embedding(input_ids, position_ids)
        mask = ~torch.tril(torch.ones(8, 8, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        out = model.decoder(hidden, mask)
        loss = out.sum()
        loss.backward()

        gdn_layers = [i for i, l in enumerate(model.decoder.layers)
                      if isinstance(l.self_attention, GatedDeltaNetAttention)]
        check("Has GDN layers", len(gdn_layers) > 0)

        for i in gdn_layers:
            attn = model.decoder.layers[i].self_attention
            check(f"Layer {i} beta_proj grad", attn.beta_proj.weight.grad is not None)
            check(f"Layer {i} linear_qkv grad", attn.linear_qkv.weight.grad is not None)

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        check("TP gradient flow", False, str(e))
        import traceback; traceback.print_exc()

    # Summary
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"TP={TP_SIZE} Results: {passed} passed, {failed} failed")
        print(f"{'='*60}")

    dist.destroy_process_group()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
