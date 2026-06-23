"""verl080 精度比较 — ON vs OFF（packed + 2D 整合对比）。

对比项分两类：

  **packed（suffix 对齐，复用 v070 cmp_diag）**
    - attention_output per-layer cos   每层 attention 输出余弦相似度
    - first_token                       packed[0]（attn[0] + logits[0]）
    - logits packed                     全 packed logits suffix 对齐对比

  **2D（v080 特有，restore 后 ``[B, L_max]``）**
    - logprobs / entropy                直接逐元素对比（ON/OFF 同坐标系）

为什么 packed 部分能直接复用 v070 ``cmp_diag``：
  v080 的 attn/logits dump 沿用 ``diagnostic_dump.py`` 的 ``dump_attn_on/off`` /
  ``dump_logits``，文件格式与 v070 完全一致（packed ``[N,…]`` + ``cu_seqlens_q.pt`` +
  ``prefix_lens.pt``）。suffix 对齐逻辑（OFF 的 cu_seqlens + ON 的 prefix_lens 构建
  1D mask，从 OFF 完整 packed 提取 ON suffix 对应段）通用，直接 import ``cmp_diag``
  的对比函数即可。

为什么 2D 部分要新写：
  v080 的 logprobs/entropy 是 restore 后展成的 2D ``[B, L_max]``（ON=restore 后含
  完整 prompt+response / OFF=原始 forward），ON/OFF 同 shape 同坐标系，直接逐元素
  对比，**不需要 suffix 对齐**。v070 ``cmp_diag.cmp_2d`` 没有 shape diagnostics，
  v080 场景下 shape 可能不一致（dump 端坐标系错配），需要增强：开头打印 ON/OFF
  所有文件 shape 定位根因，shape 不匹配时不 crash 而是报告。

dump 文件约定（``diagnostic_dump_verl080`` + ``diagnostic_dump``）::

    logprobs_{tag}.pt       [B, L_max]       restore 后 2D log_probs
    entropy_{tag}.pt        [B, L_max]       2D entropy
    attention_mask_{tag}.pt [B, L_max] bool  [0,L_i-1) 所有 predict 有效位
    label_mask_{tag}.pt     [B, L_max] bool  [prompt-last,L_i-1) PPO loss 范围
    logits.pt               [N, V//tp]       packed logits（ON 裁剪后 / OFF 完整）
    attn_outputs.pt         dict {layer: [N, hidden]}  per-layer packed attn output
    prefix_lens.pt          [B]              ON=plan.prefix_lens / OFF=全0
    cu_seqlens_q.pt         [B+1]            NestedTensor offsets（ON 裁剪后 / OFF 完整）

Usage:
    # 完整对比（attn per-layer + first_token + logits + logprobs + entropy）
    python cmp_diag_verl080.py --dir-on ./dump_on --dir-off ./dump_off --tag old

    # 只看某一层 attention（1-indexed）
    python cmp_diag_verl080.py --dir-on ./dump_on --dir-off ./dump_off \\
        --tag old --layer 12

    # 2D top-K 误差最大位置
    python cmp_diag_verl080.py --dir-on ./dump_on --dir-off ./dump_off \\
        --tag old --topk 20

    # OFF vs OFF baseline（噪声底）
    python cmp_diag_verl080.py --dir-on ./dump_off --dir-off ./dump_off2 \\
        --tag old

Parameters:
    --dir-on     (必需) ON dump 目录
    --dir-off    (必需) OFF dump 目录
    --dir-off2   (可选) 第二个 OFF 目录，OFF-vs-OFF baseline
    --tag        (必需) 2D 文件标签 old / train
    --mask       (可选) 2D 对比 mask: label(默认) / attention / none
    --layer      (可选) 只对比指定层 attention (1-indexed)，不传则所有层
    --atol       (可选) 2D 对比绝对容差，默认 1e-5
    --topk       (可选) 2D top-K 误差位置（0=关闭）
    --sort-err   (可选) 2D top-K 排序: abs(默认) / rel / val
    -o, --output (可选) JSON 报告
"""

from __future__ import annotations

import argparse
import os

import torch

# packed 对比（attn / logits / first_token）复用 v070 cmp_diag —— v080 的 attn/logits
# dump 沿用 diagnostic_dump.py 的 dump_attn_on/off / dump_logits，文件格式与 v070 一致
# （packed [N,…] + cu_seqlens_q.pt + prefix_lens.pt），suffix 对齐逻辑通用。
from prefix_sharing.tools.cmp_diag import (
    CheckResult,
    _SEP_DOUBLE, _SEP_THIN, _CHECK, _CROSS,
    # packed 对比函数
    cmp_attn_layer, cmp_first_token, cmp_logits_packed,
    # packed 结果打印
    _print_per_layer, _print_first_token, _print_logits_packed,
    # 通用输出（兼容 packed + 2D 混合结果）
    _print_summary, _dump_json,
    # 通用 loader
    _load_tensor,
)


# ════════════════════════════════════════════════════════════════
#  2D metric helpers（logprobs/entropy 标量逐元素对比）
# ════════════════════════════════════════════════════════════════

def _error_abs_rel(a: torch.Tensor, b: torch.Tensor,
                   mask: torch.Tensor | None = None) -> dict:
    diff = (a - b).abs()
    rel = diff / torch.maximum(a.abs(), b.abs()).clamp(min=1e-8)
    if mask is not None:
        diff, rel = diff[mask], rel[mask]
    if diff.numel() == 0:
        return {"abs_max": 0.0, "abs_mean": 0.0, "rel_max": 0.0, "rel_mean": 0.0}
    return {"abs_max": float(diff.max()), "abs_mean": float(diff.mean()),
            "rel_max": float(rel.max()), "rel_mean": float(rel.mean())}


def _pearson_r(t1: torch.Tensor, t2: torch.Tensor,
               mask: torch.Tensor | None = None) -> float:
    x = (t1[mask] if mask is not None else t1.flatten()).to(torch.float64)
    y = (t2[mask] if mask is not None else t2.flatten()).to(torch.float64)
    if x.numel() < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).sum()
    sx = ((x - mx) ** 2).sum().sqrt()
    sy = ((y - my) ** 2).sum().sqrt()
    return float("nan") if sx == 0 or sy == 0 else float(cov / (sx * sy))


# ════════════════════════════════════════════════════════════════
#  2D mask loading
# ════════════════════════════════════════════════════════════════

def _load_mask_2d(dir_path: str, mask_kind: str, tag: str) -> torch.Tensor | None:
    """加载 2D mask：``label_mask_{tag}.pt`` / ``attention_mask_{tag}.pt``。"""
    if mask_kind == "none":
        return None
    fname = f"{mask_kind}_mask_{tag}.pt"  # label_mask_{tag} / attention_mask_{tag}
    fp = os.path.join(dir_path, fname)
    if not os.path.exists(fp):
        return None
    return torch.load(fp, weights_only=True).to(torch.bool)


def _resolve_mask(dir_off: str, mask_kind: str, tag: str,
                  ref_shape: tuple[int, ...]) -> torch.Tensor | None:
    """加载 mask（取 OFF 侧 = ground truth 坐标系）并校验 shape。

    若 mask 与 logprobs shape 不一致，打印警告并返回 None（回退到全位置对比）。
    """
    if mask_kind == "none":
        return None
    mask = _load_mask_2d(dir_off, mask_kind, tag)
    if mask is None:
        print(f"  {_CROSS} {mask_kind}_mask_{tag}.pt not found in OFF, "
              f"comparing all positions\n")
        return None
    if tuple(mask.shape) != ref_shape:
        print(f"  {_CROSS} {mask_kind}_mask shape {tuple(mask.shape)} != "
              f"logprobs {ref_shape}, comparing all positions\n")
        return None
    return mask


# ════════════════════════════════════════════════════════════════
#  2D comparison（v080 增强：返回 tensors + shape mismatch 不 crash）
# ════════════════════════════════════════════════════════════════

def cmp_2d(dir_on: str, dir_off: str, filename: str, name: str,
           mask: torch.Tensor | None, atol: float
           ) -> tuple[CheckResult, torch.Tensor | None, torch.Tensor | None]:
    """对比 ON/OFF 的 2D 张量（logprobs / entropy）。

    Returns ``(result, on_tensor, off_tensor)`` —— 后两者供 top-K 打印复用。
    """
    t1 = _load_tensor(dir_on, filename)
    t2 = _load_tensor(dir_off, filename)
    if t1 is None or t2 is None:
        return (CheckResult(name=name, passed=False,
                            metrics={"error": "file missing"}), t1, t2)
    if t1.shape != t2.shape:
        return (CheckResult(name=name, passed=False,
                            metrics={"error": "shape mismatch",
                                     "s_on": tuple(t1.shape),
                                     "s_off": tuple(t2.shape)}), t1, t2)
    m = mask.to(t1.device) if mask is not None else None
    if m is not None:
        m = m & ~torch.isnan(t1) & ~torch.isnan(t2)
    err = _error_abs_rel(t1, t2, m)
    n_act = int(m.sum()) if m is not None else t1.numel()
    return (CheckResult(name=name,
                        passed=n_act == 0 or err["abs_max"] <= atol,
                        metrics={"shape": tuple(t1.shape),
                                 "active": n_act,
                                 "abs_max": err["abs_max"],
                                 "abs_mean": err["abs_mean"],
                                 "rel_max": err["rel_max"],
                                 "rel_mean": err["rel_mean"],
                                 "pearson_r": _pearson_r(t1, t2, m),
                                 "atol": atol}), t1, t2)


# ════════════════════════════════════════════════════════════════
#  Shape diagnostics（v080 特有：定位 mismatch 根因）
# ════════════════════════════════════════════════════════════════

def _shape_of(dir_path: str, filename: str) -> str:
    fp = os.path.join(dir_path, filename)
    if not os.path.exists(fp):
        return "(missing)"
    try:
        return str(tuple(torch.load(fp, weights_only=True).shape))
    except Exception:
        return "(error)"


def _print_shapes(dir_on: str, dir_off: str, tag: str):
    """打印 ON/OFF 各 .pt 文件 shape —— 定位 shape mismatch 根因的第一手信息。

    logprobs ON vs OFF shape 不同，通常意味着 dump 端两侧坐标系不一致
    （例如 ON restore 后含完整 prompt+response，OFF 只 dump 了 response 段）。
    """
    print("-" * 70 + "\n  [shapes]  ON vs OFF dump shapes")
    print("-" * 70)
    files = [
        f"logprobs_{tag}.pt",
        f"entropy_{tag}.pt",
        f"label_mask_{tag}.pt",
        f"attention_mask_{tag}.pt",
        "logits.pt",
        "attn_outputs.pt",
        "prefix_lens.pt",
        "cu_seqlens_q.pt",
    ]
    print(f"  {'FILE':<28s} {'ON':<16s} {'OFF':<16s} {'STATUS'}")
    print(f"  {'─' * 28} {'─' * 16} {'─' * 16} {'─' * 10}")
    for fname in files:
        s_on, s_off = _shape_of(dir_on, fname), _shape_of(dir_off, fname)
        if s_on == "(missing)" or s_off == "(missing)":
            status = "—"
        elif s_on == s_off:
            status = "OK"
        else:
            status = f"{_CROSS} DIFF"
        print(f"  {fname:<28s} {s_on:<16s} {s_off:<16s} {status}")
    print()


# ════════════════════════════════════════════════════════════════
#  Output（2D 结果 + header + top-K）
# ════════════════════════════════════════════════════════════════

def _print_header(dir_on, dir_off, dir_off2, tag, mask_kind, layer):
    print(_SEP_DOUBLE)
    print("  verl080 Prefix-Sharing Diag Report")
    print(f"  ON :  {dir_on}\n  OFF:  {dir_off}")
    if dir_off2:
        print(f"  OFF2: {dir_off2}")
    detail = f"TAG: {tag}    MASK: {mask_kind}"
    if layer is not None:
        detail += f"    LAYER: {layer}"
    print(f"  {detail}")
    print(_SEP_DOUBLE + "\n")


def _print_2d_result(r: CheckResult):
    print(_SEP_THIN)
    m = r.metrics
    if "error" in m:
        print(f"  [{r.name}]  {_CROSS} {m['error']}")
        if m.get("s_on") or m.get("s_off"):
            print(f"    ON: {m.get('s_on')}   OFF: {m.get('s_off')}")
        print()
        return
    print(f"  [{r.name}]  shape={m.get('shape')}  active={m.get('active')}  "
          f"abs_max={m.get('abs_max', 0):.6e}  rel_max={m.get('rel_max', 0):.6e}  "
          f"pearson={m.get('pearson_r', float('nan')):.8f}")
    print(f"  abs_mean={m.get('abs_mean', 0):.6e}  rel_mean={m.get('rel_mean', 0):.6e}  "
          f"{_CHECK if r.passed else _CROSS} {'PASS' if r.passed else 'FAIL'}")
    print()


def _print_topk_2d(on_t: torch.Tensor, off_t: torch.Tensor,
                   mask: torch.Tensor | None, topk: int, sort_by: str,
                   label: str):
    abs_err = (on_t - off_t).abs()
    rel_err = abs_err / torch.maximum(on_t.abs(), off_t.abs()).clamp(min=1e-8)
    if sort_by == "abs":
        sort_key = abs_err
    elif sort_by == "rel":
        sort_key = rel_err
    else:  # "val"
        sort_key = torch.maximum(on_t.abs(), off_t.abs())
    if mask is not None:
        sort_key = sort_key.clone()
        sort_key[~mask.to(sort_key.device)] = float("-inf")
    flat, idx = sort_key.flatten().topk(min(topk, sort_key.numel()))
    rows = idx // sort_key.shape[1]
    cols = idx % sort_key.shape[1]
    print(f"\n  [{label}]  top-{topk} positions (sort by {sort_by})")
    print(f"  {'Seq_idx':>7s} {'POS':>6s}  {'ON':>14s}  {'OFF':>14s}  "
          f"{'ABS_ERR':>12s}  {'REL_ERR':>12s}")
    for k in range(len(flat)):
        r, c = int(rows[k]), int(cols[k])
        print(f"  {r:>7d} {c:>6d}  {float(on_t[r, c]):>14.6e}  "
              f"{float(off_t[r, c]):>14.6e}  {float(abs_err[r, c]):>12.6e}  "
              f"{float(rel_err[r, c]):>12.6e}")


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="verl080 precision comparison — ON vs OFF (packed + 2D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dir-on", required=True, help="ON dump directory")
    ap.add_argument("--dir-off", required=True, help="OFF dump directory")
    ap.add_argument("--dir-off2", default=None,
                    help="Second OFF dir for OFF-vs-OFF baseline")
    ap.add_argument("--tag", required=True,
                    help="2D file tag (old / train) — logprobs_{tag}.pt, mask_{tag}.pt")
    ap.add_argument("--mask", choices=["label", "attention", "none"],
                    default="label", help="2D mask type (default: label)")
    ap.add_argument("--layer", type=int, default=None,
                    help="Compare specific attn layer 1-indexed (default: all)")
    ap.add_argument("--atol", type=float, default=1e-5,
                    help="Absolute tolerance for 2D (default: 1e-5)")
    ap.add_argument("--topk", type=int, default=0,
                    help="2D top-K worst positions (0 = disabled)")
    ap.add_argument("--sort-err", choices=["abs", "rel", "val"], default="abs",
                    help="2D top-K sort: abs / rel / val (default: abs)")
    ap.add_argument("--output", "-o", default=None, help="Save report as JSON")
    args = ap.parse_args()

    _print_header(args.dir_on, args.dir_off, args.dir_off2,
                  args.tag, args.mask, args.layer)

    # ── shape diagnostics：直接暴露 ON/OFF 各文件 shape ──
    _print_shapes(args.dir_on, args.dir_off, args.tag)

    # ── resolve 2D mask（以 OFF logprobs shape 为参考校验） ──
    ref = _load_tensor(args.dir_off, f"logprobs_{args.tag}.pt")
    mask = None
    if ref is not None:
        mask = _resolve_mask(args.dir_off, args.mask, args.tag, tuple(ref.shape))
        if mask is not None:
            print(f"  mask: {args.mask}  shape={tuple(mask.shape)}  "
                  f"active={int(mask.sum())}\n")

    all_results: list[CheckResult] = []

    # ── packed: attention_output per-layer cos（suffix 对齐，复用 cmp_diag） ──
    r = cmp_attn_layer(args.dir_on, args.dir_off, args.layer)
    if r:
        all_results.append(r)
        _print_per_layer(r)

    # ── packed: first_token（attn[0] + logits[0]，packed[0] 直接对比） ──
    for r in cmp_first_token(args.dir_on, args.dir_off):
        all_results.append(r)
        _print_first_token(r)

    # ── packed: logits（suffix 对齐，per-token cosine） ──
    r = cmp_logits_packed(args.dir_on, args.dir_off)
    if r:
        all_results.append(r)
        _print_logits_packed(r)

    # ── 2D: logprobs + entropy（v080 restore 后，直接逐元素对比） ──
    for fname, cname in [("logprobs", "logp"), ("entropy", "entropy")]:
        fn = f"{fname}_{args.tag}.pt"
        r, t1, t2 = cmp_2d(args.dir_on, args.dir_off, fn,
                           f"{cname}_{args.tag}", mask, args.atol)
        all_results.append(r)
        _print_2d_result(r)
        if (args.topk > 0 and t1 is not None and t2 is not None
                and t1.shape == t2.shape):
            _print_topk_2d(t1.cpu(), t2.cpu(), mask, args.topk,
                           args.sort_err, r.name)

    # ── OFF vs OFF baseline（2D 噪声底） ──
    if args.dir_off2:
        print(_SEP_DOUBLE + "\n  [BASELINE]  OFF vs OFF2 Noise Floor\n" + _SEP_DOUBLE)
        for fname, cname in [("logprobs", "logp"), ("entropy", "entropy")]:
            fn = f"{fname}_{args.tag}.pt"
            r, _, _ = cmp_2d(args.dir_off, args.dir_off2, fn,
                             f"bl_{cname}_{args.tag}", mask, args.atol)
            all_results.append(r)
            _print_2d_result(r)

    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, args.dir_on, args.dir_off,
                   args.tag, args.dir_off2)


if __name__ == "__main__":
    main()
