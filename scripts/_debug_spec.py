"""Debug: inspect ModuleSpec structure for SelfAttention submodules."""
import os, sys, torch, torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dependency", "megatron_v0150"))

dist.init_process_group(backend="nccl")
from megatron.core import parallel_state as mpu
mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

from megatron.core.transformer import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

tc = TransformerConfig(
    num_layers=4, hidden_size=512, num_attention_heads=8, num_query_groups=2,
    kv_channels=64, bf16=True, params_dtype=torch.bfloat16,
    normalization="RMSNorm", tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
)
spec = get_gpt_decoder_block_spec(tc, use_transformer_engine=False)
sa = spec.layer_specs[0].submodules.self_attention
print("type:", type(sa))
print("module:", sa.module)
print("params:", sa.params)
print("submodules type:", type(sa.submodules))
print("submodules:", sa.submodules)
if hasattr(sa.submodules, 'core_attention'):
    print("core_attention:", sa.submodules.core_attention)

dist.destroy_process_group()
