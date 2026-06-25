"""TND (varlen) + sparse_mode=3 (rightDownCausal) 可行性实验（NPU only）。

背景
----
当前 ON 路径 ([flash_atten_npu.py](prefix_sharing/backends/flash_atten_npu.py)) 用
BSH + per-sample B1SS mask + sparse_mode=1，在 reuser 上结果错；改 sparse_mode=3 又崩
（"attenmask compression requires [2048,2048]"，因为 mode 3 要的是压缩 [2048,2048]，
不是 B1SS）。

调研 Ascend 60RC2 官方文档后发现：**sparse_mode=3 (rightDownCausal) 原生支持 Q<KV 的
右对齐 causal**——query i 见 kv j iff ``j <= (Skv-Sq) + i``。代到 reuser（Q=suffix、
KV=prefix+suffix）正好得到"全部 prefix KV 可见 + suffix KV causal"，且对 provider
（Q==KV）退化成标准 causal。

本文件验证：把 ON 路径 kernel 换成 **TND varlen + sparse_mode=3 + 压缩 [2048,2048]
mask + actual_seq**、**单次调用全 batch**、**reuser Q 仍 suffix-only** 是否成立。

用法
----
- pytest：``pytest prefix_sharing/backends/flash_atten_npu_test.py``（非 NPU 自动 skip）。
- 直接跑：``python prefix_sharing/backends/flash_atten_npu_test.py``，顺序跑 Probe A→E，
  打印每个 probe 的 pass/fail + out_diff/grad_diff + 推荐生产分支（决策树）。

ground truth = :class:`TorchReferenceBackend` 的 attention（其 reuser mask 即目标语义，
见 ``torch_ref._causal_q_kv_mask`` 用 ``q_start=prefix_len``）。
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

# Allow running this file directly (``python flash_atten_npu_test.py``): put the
# package root (the ``prefix-sharing/`` dir that contains ``prefix_sharing/``) on
# sys.path so the ``prefix_sharing`` imports below resolve. pytest adds rootdir
# automatically, so this bootstrap only matters for direct execution.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

# ── NPU kernel / 设备探测；非 NPU 环境把所有 probe skip 掉 ──────────────
try:
    from prefix_sharing.backends.flash_atten_npu import (  # noqa: E402
        _import_npu_fusion_attention,
    )
    _import_npu_fusion_attention()
    _HAS_NPU_KERNEL = True
except (RuntimeError, ImportError):
    _HAS_NPU_KERNEL = False

from prefix_sharing.backends.packed_layout import PackedBatchLayout  # noqa: E402
from prefix_sharing.backends.torch_ref import TorchReferenceBackend  # noqa: E402
from prefix_sharing.core.config import PrefixSharingConfig  # noqa: E402
from prefix_sharing.core.planner import PrefixSharingPlanner  # noqa: E402
from prefix_sharing.core.prefix_store import PrefixAttentionStore  # noqa: E402

try:
    _HAS_NPU_DEVICE = torch.npu.is_available()
except (AttributeError, RuntimeError):
    _HAS_NPU_DEVICE = False

DEVICE = torch.device("npu" if _HAS_NPU_DEVICE else "cpu")
_HAS_NPU = _HAS_NPU_KERNEL and _HAS_NPU_DEVICE

# ── 维度常量（小尺寸，目的是验证正确性，不是性能） ─────────────────────
NUM_HEADS = 2
NUM_KV_HEADS = 2  # 本实验先不测 GQA；要测改这里
HEAD_DIM = 64
SCALE = 1.0 / math.sqrt(HEAD_DIM)

_ATOL_FP16 = 5e-2
_ATOL_GRAD_FP16 = 2e-1


# ═══════════════════════════════════════════════════════════════════════
# 复用自 tests/integrated_test/optional/test_npu_flash_backend.py（拷贝，不 import）
# ═══════════════════════════════════════════════════════════════════════
def _make_plan(batch_sizes: list[int], prefix_lens: list[int]) -> Any:
    """构造一个结构可控的 PrefixSharingPlan。

    - batch_sizes[i] = 样本 i 的原始长度
    - prefix_lens[i] = 样本 i 的共享前缀长度（0 表示 provider / 无共享）
    """
    config = PrefixSharingConfig(enable_prefix_sharing=True, backend="flash_atten_npu")
    planner = PrefixSharingPlanner(config)
    input_ids = [list(range(s)) for s in batch_sizes]
    plan = planner.plan(input_ids)

    object.__setattr__(plan, "batch_size", len(batch_sizes))
    object.__setattr__(plan, "original_lengths", batch_sizes)
    object.__setattr__(plan, "prefix_lens", prefix_lens)
    object.__setattr__(plan, "kept_lengths_q", [b - p for b, p in zip(batch_sizes, prefix_lens)])
    object.__setattr__(plan, "expanded_lengths_kv", list(batch_sizes))
    object.__setattr__(plan, "q_position_offsets", prefix_lens)
    object.__setattr__(plan, "kv_position_offsets", [0] * len(batch_sizes))

    cu_seqlens_q = [0]
    cu_seqlens_kv = [0]
    max_seqlen_q = 0
    max_seqlen_kv = 0
    for b, p in zip(batch_sizes, prefix_lens):
        q_len = b - p
        kv_len = b
        cu_seqlens_q.append(cu_seqlens_q[-1] + q_len)
        cu_seqlens_kv.append(cu_seqlens_kv[-1] + kv_len)
        max_seqlen_q = max(max_seqlen_q, q_len)
        max_seqlen_kv = max(max_seqlen_kv, kv_len)

    object.__setattr__(plan, "cu_seqlens_q", cu_seqlens_q)
    object.__setattr__(plan, "cu_seqlens_kv", cu_seqlens_kv)
    object.__setattr__(plan, "max_seqlen_q", max_seqlen_q)
    object.__setattr__(plan, "max_seqlen_kv", max_seqlen_kv)
    object.__setattr__(plan, "provider_index", [0] * len(batch_sizes))
    object.__setattr__(plan, "is_provider", [p == 0 for p in prefix_lens])
    object.__setattr__(plan, "reuse_specs", ())
    object.__setattr__(plan, "prefix_last_restore", [])
    return plan


def _make_layout(kept_lengths_q: list[int]) -> PackedBatchLayout:
    return PackedBatchLayout.from_valid_lengths(kept_lengths_q)


def _random_qkv(total_q: int, total_kv: int, seed: int = 42):
    """随机生成 THD Q / 原始(未展开) K/V。原始 K/V 长度 = kept_lengths_q 之和。"""
    torch.manual_seed(seed)
    q = torch.randn(total_q, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE) * 0.02
    k = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE) * 0.02
    v = torch.randn(total_kv, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=DEVICE) * 0.02
    return q, k, v


# ═══════════════════════════════════════════════════════════════════════
# 本实验新增 helper
# ═══════════════════════════════════════════════════════════════════════
_COMPRESSED_MASK: torch.Tensor | None = None


def _compressed_causal_mask() -> torch.Tensor:
    """压缩 [2048,2048] 下三角 mask（True=masked），与 baseline get_attention_mask 一致。

    sparse_mode 2/3/4 共用这一张压缩 mask；区别只在 kernel 锚定方式。
    构建一次缓存。理论上覆盖任意 seq 长（压缩模式由 actual_seq 重建每段 causal）。
    """
    global _COMPRESSED_MASK
    if _COMPRESSED_MASK is None or _COMPRESSED_MASK.device != DEVICE:
        _COMPRESSED_MASK = torch.triu(
            torch.ones([2048, 2048], dtype=torch.bool, device=DEVICE), diagonal=1
        )
    return _COMPRESSED_MASK


def _build_inputs(plan: Any, seed: int = 42):
    """构造一次实验所需的全部张量 + ground truth。

    返回:
        q: [总Q, NUM_HEADS, HEAD_DIM] —— reuser suffix-only
        ek, ev: [总KV展开, NUM_KV_HEADS, HEAD_DIM] —— build_kv 展开后的 K/V（reuser 是 prefix+suffix）
        ref_out: torch_ref 在相同 (q, ek, ev, plan) 上的输出，作为 ground truth
        layout: PackedBatchLayout
    """
    layout = _make_layout(plan.kept_lengths_q)
    total_q = int(sum(plan.kept_lengths_q))
    # 原始(未展开) K/V 长度 = kept_lengths_q 之和（裁剪后 provider 全长、reuser suffix）
    q, k_raw, v_raw = _random_qkv(total_q, total_q, seed=seed)

    store = PrefixAttentionStore()
    ref = TorchReferenceBackend()
    ek, ev = ref.build_kv(
        k_raw, v_raw, store, plan,
        packed_batch_layout=layout, layer_id=0, tp_rank=0,
    )
    ref_out = ref.attention(q, ek, ev, plan, packed_batch_layout=layout)
    store.close()
    return q, ek, ev, ref_out, layout


def _varlen_tnd_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_q: list[int],
    cu_kv: list[int],
    sparse_mode: int,
) -> torch.Tensor:
    """TND varlen npu_fusion_attention 调用。

    - input_layout="TND"
    - actual_seq_qlen/kvlen 用 cu_seqlens（**带前导 0**，对齐 verl util.py:69 的 zeros(batch+1)）
    - atten_mask = 压缩 [2048,2048] 下三角
    - scale / keep_prob = 1/√d / 1.0
    - mode 2/3 下 pre/next_tokens 不生效，走默认
    """
    fn = _import_npu_fusion_attention()
    result = fn(
        q, k, v,
        NUM_HEADS,
        "TND",
        atten_mask=_compressed_causal_mask(),
        scale=SCALE,
        keep_prob=1.0,
        sparse_mode=sparse_mode,
        actual_seq_qlen=list(cu_q),
        actual_seq_kvlen=list(cu_kv),
    )
    return result[0] if isinstance(result, (tuple, list)) else result


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


# ═══════════════════════════════════════════════════════════════════════
# Probe 矩阵。每个返回 dict: {name, pass, detail, out_diff, grad_diff}
# ═══════════════════════════════════════════════════════════════════════
def probe_a_provider_only_mode2() -> dict:
    """A. provider-only + mode 2（baseline sanity）。

    两个 provider，无共享，Q==KV。验证 TND varlen + 压缩 mask + mode 2 的接线正确
    （与 baseline 同款），并与 torch_ref 对齐。A 不过 = 环境/接线坏，后续 probe 都不可信。
    """
    plan = _make_plan(batch_sizes=[4, 6], prefix_lens=[0, 0])
    q, ek, ev, ref_out, _ = _build_inputs(plan)
    try:
        out = _varlen_tnd_call(
            q, ek, ev,
            cu_q=plan.cu_seqlens_q, cu_kv=plan.cu_seqlens_kv,
            sparse_mode=2,
        )
        diff = _max_abs_diff(out, ref_out)
        ok = diff < _ATOL_FP16
        return {"name": "A", "pass": ok, "detail": f"provider-only mode2 out_diff={diff:.4e}",
                "out_diff": diff, "grad_diff": None}
    except Exception as e:  # noqa: BLE001
        return {"name": "A", "pass": False, "detail": f"exception: {type(e).__name__}: {e}",
                "out_diff": None, "grad_diff": None}


def probe_b_single_reuser_mode3() -> dict:
    """B. 单 reuser + mode 3（核心 probe）。

    从 [8,8]/[0,4] 的 batch 里取 reuser(index=1)：Q=suffix(4)、KV=prefix+suffix(8)、
    prefix_len=4。单独喂给 TND varlen + mode 3，与 torch_ref 的 reuser 段输出对齐。

    B 过 = mode 3 在 varlen 对 Q<Skv 真的右对齐（reuser 语义正确）——核心结论成立。
    B 失败（意外）= mode 3 在本 CANN 版本 varlen 下没右对齐 → 退路 mode 1(allMask) 自建 mask。
    """
    plan = _make_plan(batch_sizes=[8, 8], prefix_lens=[0, 4])
    q, ek, ev, ref_out, _ = _build_inputs(plan)
    # reuser = index 1
    q_lo, q_hi = plan.cu_seqlens_q[1], plan.cu_seqlens_q[2]
    kv_lo, kv_hi = plan.cu_seqlens_kv[1], plan.cu_seqlens_kv[2]
    q_reuser = q[q_lo:q_hi]
    ek_reuser = ek[kv_lo:kv_hi]
    ev_reuser = ev[kv_lo:kv_hi]
    ref_reuser = ref_out[q_lo:q_hi]
    try:
        out = _varlen_tnd_call(
            q_reuser, ek_reuser, ev_reuser,
            cu_q=[0, q_reuser.shape[0]],
            cu_kv=[0, ek_reuser.shape[0]],
            sparse_mode=3,
        )
        diff = _max_abs_diff(out, ref_reuser)
        ok = diff < _ATOL_FP16
        return {"name": "B", "pass": ok,
                "detail": f"single-reuser mode3 (Q=4,KV=8,prefix=4) out_diff={diff:.4e}",
                "out_diff": diff, "grad_diff": None}
    except Exception as e:  # noqa: BLE001
        return {"name": "B", "pass": False, "detail": f"exception: {type(e).__name__}: {e}",
                "out_diff": None, "grad_diff": None}


def probe_c_full_batch_mode3() -> dict:
    """C. 全 batch（provider+reuser）mode 3 单次调用。

    一次 npu_fusion_attention 覆盖 provider 段（Q==KV）和 reuser 段（Q<KV），mode 3。
    C 过 = 生产"单次调用全 batch"方案可行（provider 标准 causal + reuser 右对齐同调用正确）。
    C 失败 → 退路：拆两次调用（providers mode 2 / reusers mode 3）。
    """
    plan = _make_plan(batch_sizes=[8, 8], prefix_lens=[0, 4])
    q, ek, ev, ref_out, _ = _build_inputs(plan)
    try:
        out = _varlen_tnd_call(
            q, ek, ev,
            cu_q=plan.cu_seqlens_q, cu_kv=plan.cu_seqlens_kv,
            sparse_mode=3,
        )
        diff = _max_abs_diff(out, ref_out)
        ok = diff < _ATOL_FP16
        return {"name": "C", "pass": ok,
                "detail": f"full-batch mode3 out_diff={diff:.4e}",
                "out_diff": diff, "grad_diff": None}
    except Exception as e:  # noqa: BLE001
        return {"name": "C", "pass": False, "detail": f"exception: {type(e).__name__}: {e}",
                "out_diff": None, "grad_diff": None}


def probe_d_full_batch_backward() -> dict:
    """D. C 的反向（128-tile 约束）。

    全 batch mode 3 forward + sum().backward()，对齐 torch_ref 的 Q/K/V 梯度。
    D 过 = varlen 反向通，128-tile 约束不挡路（或被满足）。
    D 崩 tiling → 需要把 max_q/max_kv 补到 128 倍数重试（生产改写时处理）。
    """
    plan = _make_plan(batch_sizes=[8, 8], prefix_lens=[0, 4])

    def _run(forward_fn):
        q, ek, ev, _, _ = _build_inputs(plan)
        q = q.clone().detach().requires_grad_(True)
        ek = ek.clone().detach().requires_grad_(True)
        ev = ev.clone().detach().requires_grad_(True)
        out = forward_fn(q, ek, ev)
        out.sum().backward()
        return {"q": q.grad, "k": ek.grad, "v": ev.grad}

    # FA 反向
    def fa_fwd(qq, kk, vv):
        return _varlen_tnd_call(qq, kk, vv, cu_q=plan.cu_seqlens_q,
                                cu_kv=plan.cu_seqlens_kv, sparse_mode=3)
    # ref 反向（torch_ref）
    layout = _make_layout(plan.kept_lengths_q)
    ref_backend = TorchReferenceBackend()

    def ref_fwd(qq, kk, vv):
        return ref_backend.attention(qq, kk, vv, plan, packed_batch_layout=layout)

    try:
        grads_fa = _run(fa_fwd)
        grads_ref = _run(ref_fwd)
        max_grad_diff = 0.0
        for name in ("q", "k", "v"):
            d = _max_abs_diff(grads_fa[name], grads_ref[name])
            max_grad_diff = max(max_grad_diff, d)
        ok = max_grad_diff < _ATOL_GRAD_FP16
        return {"name": "D", "pass": ok,
                "detail": f"full-batch mode3 backward grad_diff={max_grad_diff:.4e}",
                "out_diff": None, "grad_diff": max_grad_diff}
    except Exception as e:  # noqa: BLE001
        return {"name": "D", "pass": False,
                "detail": f"backward exception (可能 128-tile): {type(e).__name__}: {e}",
                "out_diff": None, "grad_diff": None}


def probe_e_long_seq_gt_2048() -> dict:
    """E.（可选）mode 3 + 某段 seq>2048。

    把 provider 段拉到 >2048，确认压缩 [2048,2048] mask 不限 seq 长（压缩模式由
    actual_seq 重建每段 causal，理论上支持任意 seq）。这是用户关心的点。
    """
    plan = _make_plan(batch_sizes=[2100, 2100], prefix_lens=[0, 100])
    q, ek, ev, ref_out, _ = _build_inputs(plan)
    try:
        out = _varlen_tnd_call(
            q, ek, ev,
            cu_q=plan.cu_seqlens_q, cu_kv=plan.cu_seqlens_kv,
            sparse_mode=3,
        )
        diff = _max_abs_diff(out, ref_out)
        ok = diff < _ATOL_FP16
        return {"name": "E", "pass": ok,
                "detail": f"seq>2048 mode3 out_diff={diff:.4e}",
                "out_diff": diff, "grad_diff": None}
    except Exception as e:  # noqa: BLE001
        return {"name": "E", "pass": False, "detail": f"exception: {type(e).__name__}: {e}",
                "out_diff": None, "grad_diff": None}


_PROBES = [
    probe_a_provider_only_mode2,
    probe_b_single_reuser_mode3,
    probe_c_full_batch_mode3,
    probe_d_full_batch_backward,
    probe_e_long_seq_gt_2048,
]


# ═══════════════════════════════════════════════════════════════════════
# pytest 包装（每个 probe 一个 test，非 NPU skip）
# ═══════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not _HAS_NPU, reason="requires NPU device + mindspeed")
def test_probe_a_provider_only_mode2():
    r = probe_a_provider_only_mode2()
    assert r["pass"], r["detail"]


@pytest.mark.skipif(not _HAS_NPU, reason="requires NPU device + mindspeed")
def test_probe_b_single_reuser_mode3():
    r = probe_b_single_reuser_mode3()
    assert r["pass"], r["detail"]


@pytest.mark.skipif(not _HAS_NPU, reason="requires NPU device + mindspeed")
def test_probe_c_full_batch_mode3():
    r = probe_c_full_batch_mode3()
    assert r["pass"], r["detail"]


@pytest.mark.skipif(not _HAS_NPU, reason="requires NPU device + mindspeed")
def test_probe_d_full_batch_backward():
    r = probe_d_full_batch_backward()
    assert r["pass"], r["detail"]


@pytest.mark.skipif(not _HAS_NPU, reason="requires NPU device + mindspeed")
def test_probe_e_long_seq_gt_2048():
    r = probe_e_long_seq_gt_2048()
    assert r["pass"], r["detail"]


# ═══════════════════════════════════════════════════════════════════════
# 决策树（__main__ 用）
# ═══════════════════════════════════════════════════════════════════════
def _decide(results: dict[str, dict]) -> str:
    a, b, c, d = results["A"], results["B"], results["C"], results["D"]
    if not a["pass"]:
        return ("A 失败 → 环境/接线坏（input_layout 字符串/cu_seqlens 格式/scale 等）。"
                "先修 A，后续 probe 都不可信。")
    if not b["pass"]:
        return ("B 失败 → mode 3 在本 CANN 版本的 varlen 下没对 Q<Skv 右对齐（意外）。"
                "退路：sparse_mode=1(allMask) 自建右移 mask，或暂留 BSH。需进一步查 CANN 版本行为。")
    if not c["pass"]:
        return ("C 失败 → 单次调用混 batch（provider+reuser）mode 3 行为异常。"
                "退路：拆两次调用（providers mode 2 / reusers mode 3）。")
    if not d["pass"]:
        if "tiling" in d["detail"].lower() or "128" in d["detail"]:
            return ("D 失败(疑似 128-tile) → 生产改写时把 max_q/max_kv 补到 128 倍数。"
                    "若补完仍不过 → mode 3 反向不可行，退路 mode 3 仅前向/反向回退或留 BSH。")
        return ("D 失败(反向) → mode 3 varlen 反向有问题。退路：mode 3 仅前向/反向回退，或留 BSH。")
    return ("A/B/C/D 全过 → 生产设计确认：把 flash_atten_npu.py 改写为"
            " TND varlen + sparse_mode=3 单次调用（reuser Q 仍 suffix-only）。下一轮可动手。")


def _run_all() -> None:
    if not _HAS_NPU:
        print("[skip] 无 NPU 设备或 mindspeed 内核，本实验只能在 NPU 上跑。")
        return
    print(f"[env] DEVICE={DEVICE} NUM_HEADS={NUM_HEADS} NUM_KV_HEADS={NUM_KV_HEADS} "
          f"HEAD_DIM={HEAD_DIM} SCALE={SCALE:.4f}")
    print("=" * 70)
    results: dict[str, dict] = {}
    for probe in _PROBES:
        r = probe()
        results[r["name"]] = r
        tag = "PASS" if r["pass"] else "FAIL"
        diff_str = []
        if r["out_diff"] is not None:
            diff_str.append(f"out_diff={r['out_diff']:.4e}")
        if r["grad_diff"] is not None:
            diff_str.append(f"grad_diff={r['grad_diff']:.4e}")
        diff_txt = f" [{', '.join(diff_str)}]" if diff_str else ""
        print(f"Probe {r['name']}: {tag}{diff_txt}")
        print(f"    {r['detail']}")
    print("=" * 70)
    print("[决策] " + _decide(results))


if __name__ == "__main__":
    _run_all()
