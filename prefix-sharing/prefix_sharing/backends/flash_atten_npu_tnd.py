"""Ascend NPU Flash Attention backend — TND varlen + sparse_mode=3 (rightDownCausal).

这是**推荐的 NPU 后端**，与 OFF baseline（MindSpeed ``dot_product_attention``）使用
相同的 TND varlen 约定，配合 ``sparse_mode=3``。老的 BSH 后端
(:mod:`prefix_sharing.backends.flash_atten_npu`) 保留作对照/回退，不要删。

为什么用 sparse_mode=3（核心）
-------------------------------
``sparse_mode=3`` 即 **rightDownCausal**："以右下顶点划分的下三角"。对一个 Q 比
KV 短的 segment（Sq < Skv），它把 Q **右对齐到 KV 末端**：query i（局部）见 kv j
iff ``j <= (Skv - Sq) + i``。

代到 prefix-sharing 的 reuser（Q=suffix、KV=prefix+suffix、Skv-Sq=prefix_len）：
query i 见 kv j iff ``j <= prefix_len + i`` = **全部 prefix KV 可见 + suffix KV causal**，
正是 reuser 的正确语义。而对 provider（Sq==Skv）退化成 ``j <= i`` = 标准 causal，
与 baseline（``sparse_mode=2`` leftUpCausal）一致。

因此**整个 batch（providers + reusers）一次 varlen 调用即可**：kernel 按每个 segment
的 actual_seq 自动判 provider（标准 causal）还是 reuser（右对齐 causal）。reuser 的
Q 仍是 suffix-only（省算力核心收益不变），mask 用现成的压缩 ``[2048,2048]`` 下三角
（与 baseline ``get_attention_mask`` 完全一样），**无需自建 mask、无需按 prefix_len
分组拆调用**。

为什么不用老 backend 的 BSH + sparse_mode=1
-------------------------------------------
老 backend 用 BSH + per-sample B1SS mask + ``sparse_mode=1``，实测在 reuser 上结果
错（mode 1=allMask 对 B1SS 自建 mask 的处理不对）。改 ``sparse_mode=3`` 又崩
（"attenmask compression requires [2048,2048]"，因为 mode 3 要压缩 ``[2048,2048]``，
不是 B1SS）。TND varlen + 压缩 mask 才是 mode 3 的正确用法。

曾经担心 varlen 反向有 128-tile 约束（见老 backend 的 docstring），实测
（``flash_atten_npu_test.py`` Probe D）已证伪——本 CANN 版本 varlen 反向正常。

选用方式
--------
配置里设 ``backend="flash_atten_npu_tnd"`` 即可指向本后端。

Select via config: ``backend="flash_atten_npu_tnd"``.
"""

from __future__ import annotations

import importlib
import math
from functools import lru_cache
from typing import Any

from prefix_sharing.backends.base import BackendCapabilities
from prefix_sharing.backends.flash_atten_base import (
    FlashAttentionMixin,
    FlashBackendValidationError,
)
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlan


_CANDIDATES = [
    ("mindspeed.ops.fusion_attention_v2", "npu_fusion_attention"),
    ("mindspeed.ops", "npu_fusion_attention"),
]


@lru_cache(maxsize=None)
def _import_npu_fusion_attention():
    last_err = None
    for module_name, attr in _CANDIDATES:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except ImportError as e:
            last_err = e
    raise RuntimeError(
        "NpuFlashAttentionTndBackend requires MindSpeed (mindspeed.ops). "
        "Install MindSpeed matching your CANN version."
    ) from last_err


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("NpuFlashAttentionTndBackend requires PyTorch") from exc
    return torch


# 压缩 [2048,2048] 下三角 mask（True=masked）按 device 缓存。
# sparse_mode 2/3/4 共用这张压缩 mask；kernel 拿 actual_seq 重建每段 causal，
# 故不限 seq 长（实测 seq>2048 正常，见 flash_atten_npu_test.py Probe E）。
_COMPRESSED_MASK: dict[Any, Any] = {}


def _compressed_causal_mask(device: Any) -> Any:
    torch = _torch()
    cached = _COMPRESSED_MASK.get(device)
    if cached is None:
        cached = torch.triu(
            torch.ones([2048, 2048], dtype=torch.bool, device=device), diagonal=1
        )
        _COMPRESSED_MASK[device] = cached
    return cached


class NpuFlashAttentionTndBackend(FlashAttentionMixin):
    """Ascend NPU backend via ``npu_fusion_attention`` (TND varlen, sparse_mode=3).

    单次 varlen 调用覆盖整个 batch（providers + reusers）：

    - **Provider 段**（Sq==Skv）：``sparse_mode=3`` 退化成标准 causal。
    - **Reuser 段**（Sq<Skv）：``sparse_mode=3`` 的 rightDownCausal 右对齐 Q，
      得到 "prefix KV 全可见 + suffix KV causal"。

    Q 对 reuser 仍是 suffix-only（省算力）；mask 用压缩 ``[2048,2048]`` 下三角，
    不自建。详见模块 docstring。
    """

    capabilities = BackendCapabilities(
        name="flash_atten_npu_tnd",
        supports_cpu=False,
        supports_cuda=False,
        supports_cann=True,
        supports_different_q_kv_lengths=True,
        supports_prefix_last_restore=True,
        supports_gated_attention=False,
        supports_deltanet_state_reuse=False,
    )

    def __init__(self) -> None:
        self._torch_ref = TorchReferenceBackend()

    def validate(self, config: PrefixSharingConfig, model_config: Any | None = None) -> None:
        config.validate(model_config=model_config)
        _import_npu_fusion_attention()

    def apply_rope(
        self,
        query: Any,
        key: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        return self._torch_ref.apply_rope(query, key, prefix_sharing_plan, **kwargs)

    def build_kv(
        self,
        key: Any,
        value: Any,
        store: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        *,
        packed_batch_layout: Any | None = None,
        layer_id: int,
        tp_rank: int = 0,
        stats: Any | None = None,
    ) -> tuple[Any, Any]:
        """KV 展开委托给 torch reference（与其它后端一致）。"""
        return self._torch_ref.build_kv(
            key,
            value,
            store,
            prefix_sharing_plan,
            packed_batch_layout=packed_batch_layout,
            layer_id=layer_id,
            tp_rank=tp_rank,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # attention — TND varlen + sparse_mode=3，单次调用全 batch
    # ------------------------------------------------------------------
    def attention(
        self,
        query: Any,
        key: Any,
        value: Any,
        prefix_sharing_plan: PrefixSharingPlan,
        **kwargs: Any,
    ) -> Any:
        """Run prefix-sharing attention via TND-varlen ``npu_fusion_attention``.

        流程：
        1. ``_prepare_flash_inputs`` 剥 Q 的 TP padding、产出 cu_seqlens_q/kv
           （batch+1，带前导 0，取自 plan）。
        2. 一次 ``npu_fusion_attention``，``input_layout="TND"``、
           ``sparse_mode=3``（rightDownCausal）、压缩 ``[2048,2048]`` mask。
        3. 必要时把输出按 pad_layout 回填 TP padding，恢复原 Q 形状。

        K/V 来自 ``build_kv``，已是展开后的 TND（reuser 是 prefix+suffix），
        无 padding，跟随 ``plan.expanded_lengths_kv``。
        """
        layer_id = kwargs.get("layer_id", "?")
        packed_batch_layout = kwargs.get("packed_batch_layout")
        if packed_batch_layout is None:
            raise FlashBackendValidationError(
                "flash_atten_npu_tnd.attention requires packed_batch_layout kwarg."
            )

        print(
            f"[PS][backend] flash_atten_npu_tnd attention: "
            f"layer={layer_id}, "
            f"q_shape={tuple(query.shape)}, k_shape={tuple(key.shape)}, "
            f"v_shape={tuple(value.shape)}"
        )

        npu_fusion_attention = _import_npu_fusion_attention()

        # Step 1: 剥 Q 的 TP padding + 取 cu_seqlens_q/kv（plan 语义长度，带前导 0）。
        # _prepare_flash_inputs 返回 8 元组：q, k, v, cu_seqlens_q, cu_seqlens_kv,
        # max_seqlen_q, max_seqlen_kv, pad_layout（max_seqlen_* 这里不用）。
        q, k, v, cu_seqlens_q, cu_seqlens_kv, _, _, pad_layout = (
            self._prepare_flash_inputs(
                query,
                key,
                value,
                prefix_sharing_plan,
                attention_mask=kwargs.get("attention_mask"),
                packed_batch_layout=packed_batch_layout,
            )
        )

        num_q_heads = q.shape[1]
        head_dim = q.shape[-1]
        scale = kwargs.get("softmax_scale") or (1.0 / math.sqrt(head_dim))
        dropout_p = kwargs.get("dropout_p", 0.0)
        keep_prob = kwargs.get("keep_prob", 1.0 - dropout_p)

        # Step 2: 单次 varlen 调用，sparse_mode=3（rightDownCausal）。
        # mode 3 下 pre/next_tokens 不生效，走默认；atten_mask 用压缩 [2048,2048]。
        try:
            result = npu_fusion_attention(
                q,
                k,
                v,
                num_q_heads,
                "TND",
                atten_mask=_compressed_causal_mask(q.device),
                scale=scale,
                keep_prob=keep_prob,
                sparse_mode=3,
                actual_seq_qlen=cu_seqlens_q.tolist(),
                actual_seq_kvlen=cu_seqlens_kv.tolist(),
            )
        except Exception as exc:
            raise FlashBackendValidationError(
                f"npu_fusion_attention (TND, sparse_mode=3) failed: "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, "
                f"cu_seqlens_q={cu_seqlens_q.tolist()}, "
                f"cu_seqlens_kv={cu_seqlens_kv.tolist()}"
            ) from exc

        output = result[0] if isinstance(result, (tuple, list)) else result

        # Step 3: 回填 TP padding，恢复原 Q 形状（TP=1 时 pad_layout 为 None，no-op）。
        if pad_layout is not None:
            output = self._repad_output(output, pad_layout)

        return output
