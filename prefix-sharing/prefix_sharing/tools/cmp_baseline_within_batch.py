"""GEMM Precision Baseline — Within-Batch Pairwise Comparison.

Compares N identical copies WITHIN a single forward pass to verify that
the same GEMM kernel produces bit-identical results for the same data.

Expected result: max_abs == 0.0 for all pairs (same batch size → same kernel).
Non-zero results indicate non-determinism beyond batch-size effects.

Usage::

    export PREFIX_SHARING_DIAG_DUMP=/dump_multi
    # forward with batch=[A x N]

    python cmp_baseline_within_batch.py --dir-multi /dump_multi --num-copies 4
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field

import torch

from prefix_sharing.tools.cmp_diag_verl080 import (
    CheckResult,
    _SEP_DOUBLE,
    _SEP_SINGLE,
    _CHECK,
    _CROSS,
    _cosine_sim,
    _error_abs_rel,
    _pearson_r,
    _dump_json,
)


# ══════════════════════════════════════════════════════════════════
#  Local helpers (same as cross_batch)
# ══════════════════════════════════════════════════════════════════

def _load_per_layer_dict(dir_path: str, filename: str) -> dict | None:
    fp = os.path.join(dir_path, filename)
    if not os.path.exists(fp):
        return None
    d = torch.load(fp, weights_only=True)
    return d if isinstance(d, dict) else None


def _load_cu_seqlens(dir_path: str) -> torch.Tensor | None:
    fp = os.path.join(dir_path, "cu_seqlens_q.pt")
    if not os.path.exists(fp):
        return None
    return torch.load(fp, weights_only=True)


def _extract_copy(packed: torch.Tensor, cu: torch.Tensor, copy_idx: int) -> torch.Tensor:
    return packed[int(cu[copy_idx]) : int(cu[copy_idx + 1])]


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a_f = a.reshape(a.shape[0], -1).float()
    b_f = b.reshape(b.shape[0], -1).float()
    err = _error_abs_rel(a_f, b_f)
    cos = _cosine_sim(a_f, b_f, dim=-1)
    pr = _pearson_r(a_f, b_f)
    return {
        "max_abs": err["abs_max"],
        "mean_abs": err["abs_mean"],
        "rel_max": err["rel_max"],
        "rel_mean": err["rel_mean"],
        "cos_avg": float(cos.mean()),
        "cos_min": float(cos.min()),
        "pearson": pr,
        "n_tokens": a_f.shape[0],
    }


def _get_layers(data: dict) -> list[int]:
    return sorted(int(k) for k in data.keys())


# ══════════════════════════════════════════════════════════════════
#  Comparison drivers
# ══════════════════════════════════════════════════════════════════

def _compare_plain_file_within(dir_multi: str, filename: str,
                               cu: torch.Tensor, n_copies: int,
                               layer: int | None,
                               label: str) -> CheckResult:
    """Pairwise comparison of copies within a single dump for ``filename``."""
    d = _load_per_layer_dict(dir_multi, filename)
    if d is None:
        return CheckResult(name=label, passed=False,
                           metrics={"error": f"{filename} missing"})

    layers = _get_layers(d)
    if layer is not None:
        layers = [l for l in layers if l == layer]
    if not layers:
        return CheckResult(name=label, passed=False,
                           metrics={"error": "no layers"})

    n_pairs = n_copies * (n_copies - 1) // 2
    per_layer: dict = {}
    worst_md = 0.0
    worst_pair: tuple | None = None
    worst_layer: int | None = None

    for lyr in layers:
        mt = d[lyr].float()
        copies = [_extract_copy(mt, cu, i) for i in range(n_copies)]
        T = copies[0].shape[0]
        layer_worst = 0.0
        layer_pair: tuple | None = None
        for i in range(n_copies):
            for j in range(i + 1, n_copies):
                m = _tensor_metrics(copies[i], copies[j])
                if m["max_abs"] > layer_worst:
                    layer_worst = m["max_abs"]
                    layer_pair = (i, j)
        per_layer[lyr] = {
            "max_abs": layer_worst,
            "worst_pair": list(layer_pair) if layer_pair else None,
            "n_pairs": n_pairs,
            "n_tokens": T,
        }
        if layer_worst > worst_md:
            worst_md = layer_worst
            worst_pair = layer_pair
            worst_layer = lyr

    return CheckResult(
        name=label,
        passed=worst_md == 0.0,
        metrics={
            "layers": per_layer,
            "worst_max_abs": worst_md,
            "worst_layer": worst_layer,
            "worst_pair": worst_pair,
            "n_pairs": n_pairs,
        },
    )


def _compare_rope_preqk_within(dir_multi: str, cu: torch.Tensor,
                               n_copies: int, layer: int | None,
                               label: str, fname: str = "rope_preqk.pt") -> CheckResult:
    """Pairwise comparison of ``{layer: {"query","key"}}`` dict within a single dump."""
    d = _load_per_layer_dict(dir_multi, fname)
    if d is None:
        return CheckResult(name=label, passed=False,
                           metrics={"error": f"{fname} missing"})

    layers = _get_layers(d)
    if layer is not None:
        layers = [l for l in layers if l == layer]
    if not layers:
        return CheckResult(name=label, passed=False, metrics={"error": "no layers"})

    n_pairs = n_copies * (n_copies - 1) // 2
    per_layer: dict = {}
    worst_md = 0.0
    worst_pair: tuple | None = None
    worst_layer: int | None = None

    for lyr in layers:
        mq = d[lyr]["query"].float()
        mk = d[lyr]["key"].float()
        q_copies = [_extract_copy(mq, cu, i) for i in range(n_copies)]
        k_copies = [_extract_copy(mk, cu, i) for i in range(n_copies)]
        layer_worst = 0.0
        layer_pair: tuple | None = None
        for i in range(n_copies):
            for j in range(i + 1, n_copies):
                qm = _tensor_metrics(q_copies[i], q_copies[j])
                km = _tensor_metrics(k_copies[i], k_copies[j])
                w = max(qm["max_abs"], km["max_abs"])
                if w > layer_worst:
                    layer_worst = w
                    layer_pair = (i, j)
        per_layer[lyr] = {
            "max_abs": layer_worst,
            "worst_pair": list(layer_pair) if layer_pair else None,
            "n_pairs": n_pairs,
        }
        if layer_worst > worst_md:
            worst_md = layer_worst
            worst_pair = layer_pair
            worst_layer = lyr

    return CheckResult(
        name=label,
        passed=worst_md == 0.0,
        metrics={
            "layers": per_layer,
            "worst_max_abs": worst_md,
            "worst_layer": worst_layer,
            "worst_pair": worst_pair,
            "n_pairs": n_pairs,
        },
    )


# ══════════════════════════════════════════════════════════════════
#  Output
# ══════════════════════════════════════════════════════════════════

def _print_header(dir_multi: str, cu: torch.Tensor, n_copies: int):
    n_pair = n_copies * (n_copies - 1) // 2
    T = int(cu[1]) - int(cu[0])
    ok = (cu.numel() == n_copies + 1
          and all(int(cu[i + 1]) - int(cu[i]) == T for i in range(n_copies)))
    print(_SEP_DOUBLE)
    print("  GEMM Precision Baseline — Within-Batch Pairwise Comparison")
    print(f"  Directory: {dir_multi}")
    print(f"  Copies: {n_copies}  →  {n_pair} pair(s)")
    print(f"  Tokens per copy: {T}")
    print(_SEP_DOUBLE)
    print(f"  cu_seqlens: {cu.tolist()}"
          f"  {' ' + _CHECK if ok else ' ' + _CROSS + ' copies not uniform'}")
    print()


def _print_within_plain_table(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [{r.name}]  Max abs error across all C(N,2) pairs")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {_CROSS} {m['error']}\n"); return
    layers = m.get("layers", {})
    n_pairs = m.get("n_pairs", "—")
    print(f"  {'LAYER':>6s}  {'PAIRS':>6s}  {'MAX_ABS':>12s}  {'WORST_PAIR':>12s}  "
          f"{'COS_MIN':>10s}  {'STATUS':>8s}")
    print(f"  {'─' * 6}  {'─' * 6}  {'─' * 12}  {'─' * 12}  {'─' * 10}  {'─' * 8}")
    for lyr in sorted(layers):
        d = layers[lyr]
        md = d["max_abs"]
        wp = d.get("worst_pair", "—")
        wp_s = str(wp) if wp else "—"
        ok = md == 0.0
        # cos_min not tracked per-layer in simple mode; use "—"
        print(f"  {lyr:>6d}  {n_pairs:>6}  {md:>12.3e}  {wp_s:>12}  "
              f"{'—':>10}  {'PASS' if ok else 'DIFF':>8s}")
    print()


def _print_within_verdict(r: CheckResult):
    print(_SEP_DOUBLE)
    m = r.metrics
    if m.get("worst_max_abs", 1.0) == 0.0:
        print(f"  RESULT: ALL max_abs == 0.0  {_CHECK}")
        print("  → No within-batch non-determinism detected.")
        print("  → Any non-zero diff in cross-batch baseline is purely from")
        print("    batch-size-induced GEMM kernel selection.")
    else:
        print(f"  RESULT: max_abs = {m.get('worst_max_abs', '?'):.3e}  {_CROSS}")
        print(f"  → Non-determinism detected!")
        print(f"     Layer: {m.get('worst_layer', '?')}")
        print(f"     Pair:  {m.get('worst_pair', '?')}")
        print("  → Check: dropout disabled? model.eval()? non-deterministic CUDA?")
    print(_SEP_DOUBLE)
    print()


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="GEMM precision baseline — within-batch pairwise comparison",
        epilog=__doc__,
    )
    ap.add_argument("--dir-multi", required=True,
                    help="Multi-copy dump directory (batch=[A x N])")
    ap.add_argument("--num-copies", type=int, required=True,
                    help="Number of copies N")
    ap.add_argument("--layer", type=int, default=None,
                    help="Compare specific layer 1-indexed (default: all)")
    ap.add_argument("--output", "-o", default=None,
                    help="Write JSON report to this path")
    args = ap.parse_args()

    # ── Load cu_seqlens & validate ──
    cu = _load_cu_seqlens(args.dir_multi)
    if cu is None:
        print(f"{_CROSS} cu_seqlens_q.pt missing")
        return 1
    _print_header(args.dir_multi, cu, args.num_copies)

    all_results: list[CheckResult] = []

    # ── hidden_states ──
    r = _compare_plain_file_within(args.dir_multi, "hidden_states.pt",
                                   cu, args.num_copies, args.layer,
                                   "hidden_states")
    all_results.append(r)
    _print_within_plain_table(r)

    # ── build_kv_input_v ──
    r = _compare_plain_file_within(args.dir_multi, "build_kv_input_v.pt",
                                   cu, args.num_copies, args.layer,
                                   "build_kv_input_v")
    all_results.append(r)
    _print_within_plain_table(r)

    # ── rope_preqk ──
    r = _compare_rope_preqk_within(args.dir_multi, cu, args.num_copies,
                                   args.layer, "rope_preqk")
    all_results.append(r)
    _print_within_plain_table(r)

    # ── rope_freqs ──
    r = _compare_plain_file_within(args.dir_multi, "rope_freqs.pt",
                                   cu, args.num_copies, args.layer,
                                   "rope_freqs")
    all_results.append(r)
    _print_within_plain_table(r)

    # ── rope_postqk ──
    r = _compare_rope_preqk_within(args.dir_multi, cu, args.num_copies,
                                   args.layer, "rope_postqk",
                                   fname="rope_postqk.pt")
    all_results.append(r)
    _print_within_plain_table(r)

    # ── attn_outputs ──
    r = _compare_plain_file_within(args.dir_multi, "attn_outputs.pt",
                                   cu, args.num_copies, args.layer,
                                   "attn_outputs")
    all_results.append(r)
    _print_within_plain_table(r)

    # ── verdict ──
    all_passed = all(r.passed for r in all_results)
    worst_md = max(r.metrics.get("worst_max_abs", 0) for r in all_results)
    combined = CheckResult(name="WITHIN_BATCH_OVERALL", passed=all_passed,
                           metrics={"worst_max_abs": worst_md})
    _print_within_verdict(combined)

    if args.output:
        _dump_json(all_results, args.output, "—", args.dir_multi,
                   tag=f"within_batch_N{args.num_copies}", dir_off2=None)


if __name__ == "__main__":
    main()
