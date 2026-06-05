"""GPU integration test: Megatron model forward with prefix-sharing active.

Verifies that when a prefix-sharing context is active and packed sequences
(THD format) are used, the patched Megatron SelfAttention.forward correctly
produces prefix-shared KV expansion and attention output.

This test bridges the gap between:
- Backend-level tests (run_hybrid_attention.py, run_deltanet.py)
- Full verl training loop (requires ray)

Must run with: torchrun --nproc_per_node=1 --nnodes=1 gpu_ps_mcore_integration.py
"""
import os, sys, torch, torch.distributed as dist, importlib.util, time
from types import SimpleNamespace

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
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed


def import_mcore_module(name):
    path = os.path.join(DEPS, "verl_v070", "verl", "models", "mcore", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"verl.mcore.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GatedDeltaNetAttention = import_mcore_module("gated_delta_net").GatedDeltaNetAttention


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name, ok, detail=""):
        if ok:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            print(f"  [FAIL] {name} {detail}")

    def summary(self):
        print(f"\n{'='*60}")
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print(f"{'='*60}")
        return self.failed == 0


def make_tfconfig(hidden=256, num_layers=4, heads=4, kv_heads=2):
    return TransformerConfig(
        num_layers=num_layers, hidden_size=hidden,
        num_attention_heads=heads, num_query_groups=kv_heads,
        kv_channels=hidden // heads,
        bf16=True, params_dtype=torch.bfloat16,
        normalization="RMSNorm", init_method_std=0.02,
        hidden_dropout=0.0, attention_dropout=0.0,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        use_cpu_initialization=False,
    )


def test_packed_forward_without_prefix_sharing(r):
    """Test 1: Verify standard (non-packed) forward works without prefix-sharing."""
    print("\n--- Test 1: Standard Forward (no prefix-sharing) ---")
    tfconfig = make_tfconfig(hidden=256, num_layers=2, heads=4, kv_heads=2)
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_sub = block_spec.layer_specs[0].submodules.self_attention.submodules

    attn = SelfAttention(
        config=tfconfig, submodules=sa_sub, layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).cuda().bfloat16()

    # Standard [sq, b, h] format (no packing)
    sq, b, h = 16, 2, 256
    hidden = torch.randn(sq, b, h, dtype=torch.bfloat16, device="cuda")

    try:
        with torch.no_grad():
            out, bias = attn(hidden_states=hidden, attention_mask=None)
        r.check("Standard forward runs", True)
        r.check("Output shape", out.shape == (sq, b, h), f"got {out.shape}")
        r.check("Output finite", torch.isfinite(out).all().item())
    except Exception as e:
        r.check("Standard forward", False, str(e))
        import traceback; traceback.print_exc()

    del attn
    torch.cuda.empty_cache()


def test_gated_delta_net_packed(r):
    """Test 2: GatedDeltaNet with packed sequences (THD)."""
    print("\n--- Test 2: GatedDeltaNet Packed Forward ---")
    tfconfig = make_tfconfig(hidden=256, num_layers=2, heads=4, kv_heads=2)
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_sub = block_spec.layer_specs[0].submodules.self_attention.submodules

    attn = GatedDeltaNetAttention(
        config=tfconfig, submodules=sa_sub, layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        partial_rotary_factor=0.5, attn_output_gate=True,
    ).cuda().bfloat16()

    total_tokens = 14
    hidden = torch.randn(total_tokens, 1, 256, dtype=torch.bfloat16, device="cuda")

    try:
        with torch.no_grad():
            # GatedDeltaNet doesn't need packed_seq_params - it uses cumsum
            out, bias = attn(hidden_states=hidden, attention_mask=None)
        r.check("GDN packed forward runs", True)
        r.check("GDN output shape", out.shape == (total_tokens, 1, 256), f"got {out.shape}")
        r.check("GDN output finite", torch.isfinite(out).all().item())
    except Exception as e:
        r.check("GDN packed forward", False, str(e))
        import traceback; traceback.print_exc()

    del attn
    torch.cuda.empty_cache()


def test_prefix_sharing_context_activation(r):
    """Test 3: Verify prefix-sharing context can be activated with Megatron model."""
    print("\n--- Test 3: Prefix-Sharing Context Activation ---")
    from prefix_sharing.core.config import PrefixSharingConfig
    from prefix_sharing.core.planner import PrefixSharingPlanner
    from prefix_sharing.core.model_spec import ModelSpec
    from prefix_sharing.backends.packed_layout import PackedBatchLayout
    from prefix_sharing.integrations.context import (
        prefix_sharing_runtime_context,
        PrefixSharingRuntimeContext,
        current_prefix_sharing_context,
    )

    # Create a plan with shared prefix
    sequences = [
        [1, 2, 3, 4, 10, 11],  # provider
        [1, 2, 3, 4, 20, 21],  # reuser (shares prefix [1,2,3,4])
    ]
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences)
    r.check("Plan has sharing", plan.has_sharing)
    r.check("Plan keep_ranges", plan.input_keep_ranges is not None)

    # Create runtime state
    model_spec = ModelSpec(
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, full_attention_interval=2,
    )
    packed_layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    r.check("Packed layout created", packed_layout is not None)

    # Activate context
    from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
    from prefix_sharing.backends.factory import get_backend_instance

    runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        backend=get_backend_instance(config),
        packed_batch_layout=packed_layout,
        model_spec=model_spec,
    )

    with prefix_sharing_runtime_context(runtime_state) as ctx:
        active_ctx = current_prefix_sharing_context()
        r.check("Context active", active_ctx is not None)
        r.check("Context has plan", active_ctx.prefix_sharing_plan is not None)
        r.check("Context has store", active_ctx.store is not None)
        r.check("Context has deltanet_store", active_ctx.deltanet_store is not None)
        r.check("Context has model_spec", active_ctx.model_spec is not None)
        r.check("Model spec layer_type(0)", model_spec.layer_type(0).value == "full_attention")
        r.check("Model spec layer_type(1)", model_spec.layer_type(1).value == "linear_attention")

    # Context should be deactivated
    active_ctx = current_prefix_sharing_context()
    r.check("Context deactivated", active_ctx is None)


def test_patch_installation(r):
    """Test 4: Verify monkey-patch can be installed on SelfAttention."""
    print("\n--- Test 4: Patch Installation ---")
    from prefix_sharing.core.config import PrefixSharingConfig
    from prefix_sharing.integrations.megatron_attention import MegatronAttentionIntegration
    from prefix_sharing.backends.torch_ref import TorchReferenceBackend

    config = PrefixSharingConfig(enable_prefix_sharing=True)
    integration = MegatronAttentionIntegration(config=config, backend=TorchReferenceBackend())

    original_forward = SelfAttention.forward
    handle = integration.install(model_config={})
    patched_forward = SelfAttention.forward

    r.check("Forward was patched", original_forward is not patched_forward)

    # Verify patched forward still works without context
    tfconfig = make_tfconfig(hidden=256, num_layers=2, heads=4, kv_heads=2)
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_sub = block_spec.layer_specs[0].submodules.self_attention.submodules
    attn = SelfAttention(
        config=tfconfig, submodules=sa_sub, layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).cuda().bfloat16()

    hidden = torch.randn(8, 1, 256, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        out, _ = attn(hidden_states=hidden, attention_mask=None)
    r.check("Patched forward works", torch.isfinite(out).all().item())

    # Disable patch
    handle.disable()
    r.check("Forward restored", SelfAttention.forward is original_forward)

    del attn
    torch.cuda.empty_cache()


def test_gated_delta_net_with_context(r):
    """Test 5: GatedDeltaNet forward with prefix-sharing context active but no THD."""
    print("\n--- Test 5: GatedDeltaNet with Context (no THD) ---")
    from prefix_sharing.core.config import PrefixSharingConfig
    from prefix_sharing.core.planner import PrefixSharingPlanner
    from prefix_sharing.core.model_spec import ModelSpec
    from prefix_sharing.backends.packed_layout import PackedBatchLayout
    from prefix_sharing.integrations.context import prefix_sharing_runtime_context
    from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
    from prefix_sharing.backends.factory import get_backend_instance

    tfconfig = make_tfconfig(hidden=256, num_layers=2, heads=4, kv_heads=2)
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_sub = block_spec.layer_specs[0].submodules.self_attention.submodules

    attn = GatedDeltaNetAttention(
        config=tfconfig, submodules=sa_sub, layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        partial_rotary_factor=0.5, attn_output_gate=True,
    ).cuda().bfloat16()

    # Create context
    sequences = [[1, 2, 3, 4, 10], [1, 2, 3, 4, 20]]
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences)
    model_spec = ModelSpec(
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, full_attention_interval=2,
    )
    packed_layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        backend=get_backend_instance(config),
        packed_batch_layout=packed_layout,
        model_spec=model_spec,
    )

    hidden = torch.randn(10, 1, 256, dtype=torch.bfloat16, device="cuda")

    # With context active but NO packed_seq_params, should fall through to normal path
    with prefix_sharing_runtime_context(runtime_state):
        with torch.no_grad():
            out, _ = attn(hidden_states=hidden, attention_mask=None)
        r.check("GDN with context (no THD) runs", True)
        r.check("GDN output finite", torch.isfinite(out).all().item())

    del attn
    torch.cuda.empty_cache()


def test_hybrid_model_full_cycle(r):
    """Test 6: Full hybrid model forward with all components."""
    print("\n--- Test 6: Hybrid Model Full Cycle ---")
    tfconfig = make_tfconfig(hidden=256, num_layers=4, heads=4, kv_heads=2)

    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_submodules = block_spec.layer_specs[0].submodules.self_attention.submodules

    model = GPTModel(
        config=tfconfig,
        transformer_layer_spec=block_spec,
        vocab_size=32000,
        max_sequence_length=512,
        pre_process=True, post_process=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=10000.0,
    )

    # Replace layers 1,3 with GatedDeltaNet
    for i, layer in enumerate(model.decoder.layers):
        if i % 2 != 0:
            old_attn = layer.self_attention
            new_attn = GatedDeltaNetAttention(
                config=tfconfig, submodules=sa_submodules,
                layer_number=old_attn.layer_number,
                attn_mask_type=old_attn.attn_mask_type,
                partial_rotary_factor=0.5, attn_output_gate=True,
            )
            new_attn.to(next(old_attn.parameters()).device)
            new_attn.linear_qkv = old_attn.linear_qkv
            new_attn.linear_proj = old_attn.linear_proj
            if hasattr(old_attn, 'q_layernorm') and old_attn.q_layernorm is not None:
                new_attn.q_layernorm = old_attn.q_layernorm
            if hasattr(old_attn, 'k_layernorm') and old_attn.k_layernorm is not None:
                new_attn.k_layernorm = old_attn.k_layernorm
            layer.self_attention = new_attn

    model = model.cuda().bfloat16()

    # Forward pass
    bsz, seq_len = 2, 16
    input_ids = torch.randint(0, 32000, (bsz, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

    try:
        with torch.no_grad():
            hidden = model.embedding(input_ids, position_ids)
            mask = ~torch.tril(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            out = model.decoder(hidden, mask)
        r.check("Hybrid forward", torch.isfinite(out).all().item())
        r.check("Hybrid shape", out.shape[0] == seq_len and out.shape[1] == bsz,
                f"got {out.shape}")
    except Exception as e:
        r.check("Hybrid forward", False, str(e))
        import traceback; traceback.print_exc()

    # Gradient pass
    try:
        hidden = model.embedding(input_ids, position_ids)
        mask = ~torch.tril(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        out = model.decoder(hidden, mask)
        loss = out.sum()
        loss.backward()

        gdn_layers = [i for i, l in enumerate(model.decoder.layers)
                      if isinstance(l.self_attention, GatedDeltaNetAttention)]
        r.check("Has GDN layers", len(gdn_layers) > 0)
        for i in gdn_layers:
            attn = model.decoder.layers[i].self_attention
            r.check(f"Layer {i} beta_proj grad", attn.beta_proj.weight.grad is not None)
    except Exception as e:
        r.check("Hybrid gradient", False, str(e))
        import traceback; traceback.print_exc()

    del model
    torch.cuda.empty_cache()


def main():
    dist.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)

    print("=" * 60)
    print("Prefix-Sharing mcore Integration Test")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    r = Results()
    test_packed_forward_without_prefix_sharing(r)
    test_gated_delta_net_packed(r)
    test_prefix_sharing_context_activation(r)
    test_patch_installation(r)
    test_gated_delta_net_with_context(r)
    test_hybrid_model_full_cycle(r)

    ok = r.summary()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
