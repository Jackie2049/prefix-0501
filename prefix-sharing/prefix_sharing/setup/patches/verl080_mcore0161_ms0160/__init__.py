"""verl 0.8.0.dev + megatron-core 0.16.1 + mindspeed 0.16.0（Qwen3.5 NPU 配套）

Patch 目标：
1. MegatronEngineWithLMHead.forward_step → 微批次重组 + runtime context 注入
2. Attention.forward                     → prefix-sharing attention 拦截
3. vocab_parallel_log_probs_from_logits  → 自动 logprob restore

所有业务逻辑由 integrations 层处理，本 patch set 只负责 thin wrapper 编排。
"""

from prefix_sharing.setup.registry import PatchSpec
from .forward_step import patch_verl_forward_step
from .attention import patch_megatron_attention
from .vocab_logprobs import patch_megatron_vocab

PATCH_SET: list[PatchSpec] = [
    PatchSpec(
        module_name="verl.workers.engine.megatron.transformer_impl",
        target_getter=lambda mod: (
            getattr(mod, "MegatronEngineWithLMHead"),
            "forward_step",
        ),
        patch_factory=patch_verl_forward_step,
        description="MegatronEngineWithLMHead.forward_step → "
                    "micro-batch reorg + context (verl 0.8.0 engine)",
    ),
    PatchSpec(
        module_name="megatron.core.transformer.attention",
        target_getter=lambda mod: (getattr(mod, "Attention"), "forward"),
        patch_factory=patch_megatron_attention,
        description="Attention.forward → prefix-sharing intercept (mcore 0.16.1)",
    ),
    PatchSpec(
        module_name="verl.utils.megatron.tensor_parallel",
        target_getter=lambda mod: (mod, "vocab_parallel_log_probs_from_logits"),
        patch_factory=patch_megatron_vocab,
        description="vocab_parallel_log_probs → auto logprob restore (verl 0.8.0)",
    ),
]