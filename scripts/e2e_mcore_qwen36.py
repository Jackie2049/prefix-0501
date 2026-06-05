"""End-to-end mcore Qwen3.6 integration test.

Verifies the full model flow: hybrid model construction, layer routing,
forward pass, gradient flow, regular Qwen3 fallback, and weight converter.

Must run with: torchrun --nproc_per_node=1 --nnodes=1 e2e_mcore_qwen36.py
"""
import os, sys, torch, torch.distributed as dist, importlib.util, time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(REPO_ROOT, "dependency")
sys.path.insert(0, os.path.join(DEPS, "megatron_v0150"))
sys.path.insert(0, os.path.join(REPO_ROOT, "prefix-sharing"))

from megatron.core import parallel_state as mpu
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
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


_gdn = import_mcore_module("gated_delta_net")
GatedDeltaNetAttention = _gdn.GatedDeltaNetAttention


def make_tfconfig(hidden=512, num_layers=8, heads=8, kv_heads=2):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_query_groups=kv_heads,
        kv_channels=hidden // heads,
        bf16=True,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        init_method_std=0.02,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        use_cpu_initialization=False,
    )


def build_hybrid_gpt_model(tfconfig, vocab_size, max_seq_len, full_attention_interval,
                            partial_rotary_factor=1.0, attn_output_gate=False):
    """Build a GPTModel with GatedDeltaNet replacing linear attention layers."""
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_modspec = block_spec.layer_specs[0].submodules.self_attention
    sa_submodules = sa_modspec.submodules

    model = GPTModel(
        config=tfconfig,
        transformer_layer_spec=block_spec,
        vocab_size=vocab_size,
        max_sequence_length=max_seq_len,
        pre_process=True,
        post_process=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=10000.0,
    )

    if full_attention_interval <= 1:
        return model

    for i, layer in enumerate(model.decoder.layers):
        if i % full_attention_interval != 0:
            old_attn = layer.self_attention
            new_attn = GatedDeltaNetAttention(
                config=tfconfig,
                submodules=sa_submodules,
                layer_number=old_attn.layer_number,
                attn_mask_type=old_attn.attn_mask_type,
                partial_rotary_factor=partial_rotary_factor,
                attn_output_gate=attn_output_gate,
            )
            new_attn.to(next(old_attn.parameters()).device)
            new_attn.linear_qkv = old_attn.linear_qkv
            new_attn.linear_proj = old_attn.linear_proj
            if hasattr(old_attn, 'q_layernorm') and old_attn.q_layernorm is not None:
                new_attn.q_layernorm = old_attn.q_layernorm
            if hasattr(old_attn, 'k_layernorm') and old_attn.k_layernorm is not None:
                new_attn.k_layernorm = old_attn.k_layernorm
            layer.self_attention = new_attn

    return model


def main():
    dist.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)

    print("=" * 60)
    print("E2E mcore Qwen3.6 Integration Test")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name} {detail}")

    # ==============================
    # Test 1: Full HybridAttention model (8 layers, interval=4)
    # ==============================
    print("\n--- Test 1: HybridAttention Model (8 layers) ---")
    try:
        tfconfig = make_tfconfig(hidden=512, num_layers=8, heads=8, kv_heads=2)

        init_start = time.time()
        model = build_hybrid_gpt_model(
            tfconfig, vocab_size=32000, max_seq_len=4096,
            full_attention_interval=4,
            partial_rotary_factor=0.25, attn_output_gate=True,
        ).cuda()
        init_time = time.time() - init_start
        check("Model initialization", True)
        print(f"    Init time: {init_time:.2f}s")

        # Layer routing
        full_layers = [i for i, l in enumerate(model.decoder.layers)
                       if isinstance(l.self_attention, SelfAttention)
                       and not isinstance(l.self_attention, GatedDeltaNetAttention)]
        linear_layers = [i for i, l in enumerate(model.decoder.layers)
                         if isinstance(l.self_attention, GatedDeltaNetAttention)]
        check("Full attn [0,4]", full_layers == [0, 4], f"got {full_layers}")
        check("Linear attn [1,2,3,5,6,7]", linear_layers == [1, 2, 3, 5, 6, 7], f"got {linear_layers}")

        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        check("Total params > 0", total_params > 0)
        print(f"    Total params: {total_params:,}")

        # Forward pass
        bsz, seq_len = 2, 32
        input_ids = torch.randint(0, 32000, (bsz, seq_len), device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

        with torch.no_grad():
            hidden = model.embedding(input_ids, position_ids)
            mask = ~torch.tril(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            fwd_start = time.time()
            decoder_out = model.decoder(hidden, mask)
            fwd_time = time.time() - fwd_start
        check("Forward pass", torch.isfinite(decoder_out).all().item())
        check("Output shape", decoder_out.shape[0] == seq_len and decoder_out.shape[1] == bsz,
              f"got {decoder_out.shape}")
        print(f"    Forward time: {fwd_time*1000:.1f}ms")

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        check("Model init+forward", False, str(e))
        import traceback; traceback.print_exc()

    # ==============================
    # Test 2: Regular Qwen3 (no HybridAttention, interval=1)
    # ==============================
    print("\n--- Test 2: Regular Qwen3 (no HybridAttention) ---")
    try:
        tfconfig = make_tfconfig(hidden=512, num_layers=4, heads=8, kv_heads=2)

        model = build_hybrid_gpt_model(
            tfconfig, vocab_size=32000, max_seq_len=4096,
            full_attention_interval=1,  # No hybrid
            partial_rotary_factor=1.0, attn_output_gate=False,
        ).cuda()

        # With interval=1, ALL layers should be full attention (no GatedDeltaNet)
        gdn_layers = [i for i, l in enumerate(model.decoder.layers)
                      if isinstance(l.self_attention, GatedDeltaNetAttention)]
        check("No GatedDeltaNet layers (interval=1)", gdn_layers == [], f"got {gdn_layers}")

        # Forward pass
        input_ids = torch.randint(0, 32000, (2, 16), device="cuda")
        position_ids = torch.arange(16, device="cuda").unsqueeze(0).expand(2, -1)
        with torch.no_grad():
            hidden = model.embedding(input_ids, position_ids)
            mask = ~torch.tril(torch.ones(16, 16, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            out = model.decoder(hidden, mask)
        check("Regular Qwen3 forward", torch.isfinite(out).all().item())

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        check("Regular Qwen3", False, str(e))
        import traceback; traceback.print_exc()

    # ==============================
    # Test 3: Gradient flow through full model
    # ==============================
    print("\n--- Test 3: Gradient Flow Through Full Model ---")
    try:
        tfconfig = make_tfconfig(hidden=256, num_layers=4, heads=4, kv_heads=2)

        model = build_hybrid_gpt_model(
            tfconfig, vocab_size=32000, max_seq_len=512,
            full_attention_interval=2,
            partial_rotary_factor=0.5, attn_output_gate=True,
        ).cuda()

        input_ids = torch.randint(0, 32000, (2, 8), device="cuda")
        position_ids = torch.arange(8, device="cuda").unsqueeze(0).expand(2, -1)

        hidden = model.embedding(input_ids, position_ids)
        mask = ~torch.tril(torch.ones(8, 8, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        out = model.decoder(hidden, mask)
        loss = out.sum()
        loss.backward()

        # Check gradients flow through GatedDeltaNet layers
        gdn_layers = [i for i, l in enumerate(model.decoder.layers)
                      if isinstance(l.self_attention, GatedDeltaNetAttention)]
        check("Has GatedDeltaNet layers", len(gdn_layers) > 0, "expected at least 1")

        for i in gdn_layers:
            attn = model.decoder.layers[i].self_attention
            check(f"Layer {i} beta_proj grad", attn.beta_proj.weight.grad is not None)
            check(f"Layer {i} linear_qkv grad", attn.linear_qkv.weight.grad is not None)
            check(f"Layer {i} linear_proj grad", attn.linear_proj.weight.grad is not None)

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        check("Gradient flow", False, str(e))
        import traceback; traceback.print_exc()

    # ==============================
    # Test 4: Weight converter compatibility
    # ==============================
    print("\n--- Test 4: Weight Converter ---")
    try:
        from types import SimpleNamespace
        config_mod = import_mcore_module("config_converter")
        converter_mod = import_mcore_module("weight_converter")
        hf_to_mcore_config_dense = config_mod.hf_to_mcore_config_dense
        DenseConverter = converter_mod.McoreToHFWeightConverterDense

        hf_config_wc = SimpleNamespace(
            architectures=["Qwen3ForCausalLM"],
            hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, intermediate_size=512,
            max_position_embeddings=512, rms_norm_eps=1e-6,
            rope_theta=10000.0, vocab_size=32000,
            tie_word_embeddings=False, rope_scaling=None,
            full_attention_interval=2, partial_rotary_factor=0.5,
            attn_output_gate=True, qk_layernorm=True,
            hidden_act="silu", model_type="qwen3",
            attention_dropout=0.0, head_dim=None,
        )

        tfconfig = hf_to_mcore_config_dense(hf_config_wc, torch.bfloat16)
        tfconfig.tensor_model_parallel_size = 1
        tfconfig.pipeline_model_parallel_size = 1

        # Test converter with new param names
        converter = DenseConverter(hf_config_wc, tfconfig)
        test_names = [
            "decoder.layers.1.self_attention.beta_proj.weight",
            "decoder.layers.1.self_attention.beta_proj.bias",
            "decoder.layers.1.self_attention.decay_proj.weight",
            "decoder.layers.1.self_attention.decay_proj.bias",
            "decoder.layers.1.self_attention.gate_proj.weight",
        ]
        for name in test_names:
            try:
                result = converter.convert_param(name, [torch.randn(4, 256)])
                check(f"Convert {name.split('.')[-2]}", len(result[0]) == 1)
            except Exception as e:
                check(f"Convert {name.split('.')[-2]}", False, str(e))

        del converter
        torch.cuda.empty_cache()
    except Exception as e:
        check("Weight converter", False, str(e))
        import traceback; traceback.print_exc()

    # ==============================
    # Summary
    # ==============================
    print(f"\n{'='*60}")
    print(f"E2E Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    dist.destroy_process_group()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
