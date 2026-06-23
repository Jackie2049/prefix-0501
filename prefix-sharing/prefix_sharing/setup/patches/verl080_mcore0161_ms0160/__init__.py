"""verl 0.8.0.dev + megatron-core 0.16.1 + mindspeed 0.16.0（Qwen3.5 NPU 配套）

Patch 目标：
1. MegatronEngineWithLMHead.forward_step → 微批次重组 + runtime context 注入
2. Attention.forward                     → prefix-sharing attention 拦截
3. vocab_parallel_log_probs_from_logits  → 自动 logprob restore
4. no_padding_2_padding                  → PS 物理裁剪后修正序列长度
   (module-level + 所有 from...import 引用)
5. main_ppo.run_ppo                      → fixed rollout + synthetic prefix injection

所有业务逻辑由 integrations 层处理，本 patch set 只负责 thin wrapper 编排。
"""

from prefix_sharing.setup.registry import PatchSpec
from .forward_step import patch_verl_forward_step
from .attention import patch_megatron_attention
from .vocab_logprobs import patch_megatron_vocab
from .nopadding import patch_no_padding_2_padding
from .fit_hooks import create_run_ppo_patch

# no_padding_2_padding 被 4 个模块用 from...import 直接引用
_NOPADDING_PATCH_MODULES = [
    "verl.workers.utils.padding",
    "verl.workers.utils.losses",
    "verl.trainer.distillation.losses",
    "verl.trainer.ppo.ray_trainer",
]

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
    PatchSpec(
        module_name="verl.trainer.main_ppo",
        target_getter=lambda mod: (mod, "run_ppo"),
        patch_factory=create_run_ppo_patch,
        description="run_ppo → fixed rollout + synthetic prefix injection (after init_workers)",
    ),
] + [
    PatchSpec(
        module_name=mod_name,
        target_getter=lambda mod: (mod, "no_padding_2_padding"),
        patch_factory=patch_no_padding_2_padding,
        description=f"no_padding_2_padding in {mod_name} → PS trimming-aware",
    )
    for mod_name in _NOPADDING_PATCH_MODULES
]
