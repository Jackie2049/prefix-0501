"""verl 0.8.x + megatron-core 0.16.x + megatron-bridge 0.4.x + mindspeed 0.15.3

Patch 目标：
1. Attention.forward          → prefix-sharing attention 拦截
2. vocab_parallel_log_probs   → 自动 logprob restore
3. MegatronEngineWithLMHead.forward_step → 微批次重组 + runtime context
"""

from prefix_sharing.setup.registry import PatchSpec
from .attention import make_attention_patch
from .vocab_logprobs import make_vocab_logprobs_patch
from .forward_step import make_forward_step_patch

PATCH_SET: list[PatchSpec] = [
    PatchSpec(
        module_name="megatron.core.transformer.attention",
        target_getter=lambda mod: (getattr(mod, "Attention"), "forward"),
        patch_factory=make_attention_patch,
        description="Attention.forward → prefix-sharing intercept (mcore 0.16.x)",
    ),
    PatchSpec(
        module_name="verl.utils.megatron.tensor_parallel",
        target_getter=lambda mod: (mod, "vocab_parallel_log_probs_from_logits"),
        patch_factory=make_vocab_logprobs_patch,
        description="vocab_parallel_log_probs → auto logprob restore (verl 0.8.x)",
    ),
    PatchSpec(
        module_name="verl.workers.engine.megatron.transformer_impl",
        target_getter=lambda mod: (
            getattr(mod, "MegatronEngineWithLMHead"),
            "forward_step",
        ),
        patch_factory=make_forward_step_patch,
        description="forward_step → micro-batch reorg + context (verl 0.8.x)",
    ),
]