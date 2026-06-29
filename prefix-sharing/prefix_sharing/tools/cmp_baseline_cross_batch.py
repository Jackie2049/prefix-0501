"""GEMM Precision Baseline — Cross-Batch-Size Comparison.

Compares the SAME data processed at DIFFERENT batch sizes (1 copy vs N copies)
to quantify GEMM floating-point noise independent of prefix-sharing logic.

Usage::

    # Run 1: single copy
    export PREFIX_SHARING_DIAG_DUMP=/dump_single
    # forward with batch=[A]

    # Run 2: stacked copies
    export PREFIX_SHARING_DIAG_DUMP=/dump_stacked
    # forward with batch=[A x N]

    python cmp_baseline_cross_batch.py \
        --dir-single /dump_single --dir-stacked /dump_stacked --num-copies 4
"""

from __future__ import annotations

import argparse
import json
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
#  Local helpers
# ══════════════════════════════════════════════════════════════════

def _load_per_layer_dict(dir_path: str, filename: str) -> dict | None:
    """Load a per-layer ``{layer_idx: ...}`` dict from ``dir_path/filename``."""
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
    """Slice copy ``copy_idx`` from packed: ``packed[cu[i] : cu[i+1]]``."""
    return packed[int(cu[copy_idx]) : int(cu[copy_idx + 1])]


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compute per-token cosine + global error for ``[T, ...]`` tensors.

    Unlike :func:`_vec_metrics` (which expects 1-D vectors), this handles
    multi-token tensors by flattening all non-token dims.
    """
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

def _compare_plain_file(dir_single: str, dir_stacked: str, filename: str,
                        cu_single: torch.Tensor, cu_stacked: torch.Tensor,
                        n_copies: int, layer: int | None,
                        label: str) -> CheckResult:
    """Compare ``filename`` (``{layer: [T, ...]}``) across copies."""
    sd = _load_per_layer_dict(dir_single, filename)
    md = _load_per_layer_dict(dir_stacked, filename)
    if sd is None or md is None:
        return CheckResult(name=label, passed=False,
                           metrics={"error": f"{filename} missing in one or both dirs"})

    layers = _get_layers(sd)
    if layer is not None:
        layers = [l for l in layers if l == layer]
    if not layers:
        return CheckResult(name=label, passed=False,
                           metrics={"error": "no common layers"})

    per_layer: dict = {}
    worst_md = 0.0
    for lyr in layers:
        st = sd[lyr].float()
        mt = md[lyr].float()
        T = st.shape[0]
        copies: list[dict] = []
        copy_max = 0.0
        for i in range(n_copies):
            ct = _extract_copy(mt, cu_stacked, i)
            if ct.shape[0] != T:
                copies.append({"copy": i, "error":
                              f"length mismatch: single={T} copy={ct.shape[0]}"})
                continue
            m = _tensor_metrics(st, ct)
            m["copy"] = i
            copies.append(m)
            copy_max = max(copy_max, m["max_abs"])
        per_layer[lyr] = {"copies": copies, "max_across_copies": copy_max}
        worst_md = max(worst_md, copy_max)

    return CheckResult(name=label, passed=worst_md < 1e-5,
                       metrics={"layers": per_layer, "worst_max_abs": worst_md})


def _compare_rope_preqk(dir_single: str, dir_stacked: str,
                        cu_single: torch.Tensor, cu_stacked: torch.Tensor,
                        n_copies: int, layer: int | None,
                        label: str, fname: str = "rope_preqk.pt") -> CheckResult:
    """Compare ``{layer: {"query", "key"}}`` dict file across copies."""
    sd = _load_per_layer_dict(dir_single, fname)
    md = _load_per_layer_dict(dir_stacked, fname)
    if sd is None or md is None:
        return CheckResult(name=label, passed=False,
                           metrics={"error": f"{fname} missing"})

    layers = _get_layers(sd)
    if layer is not None:
        layers = [l for l in layers if l == layer]
    if not layers:
        return CheckResult(name=label, passed=False, metrics={"error": "no common layers"})

    per_layer: dict = {}
    worst_md = 0.0
    for lyr in layers:
        sq = sd[lyr]["query"].float()
        sk = sd[lyr]["key"].float()
        mq = md[lyr]["query"].float()
        mk = md[lyr]["key"].float()
        Tq, Tk = sq.shape[0], sk.shape[0]
        layer_worst = 0.0
        layer_copies: list[dict] = []
        for i in range(n_copies):
            cq = _extract_copy(mq, cu_stacked, i)
            ck = _extract_copy(mk, cu_stacked, i)
            if cq.shape[0] != Tq or ck.shape[0] != Tk:
                layer_copies.append({"copy": i, "error":
                                    f"length mismatch Q: single={Tq} copy={cq.shape[0]}"
                                    f" K: single={Tk} copy={ck.shape[0]}"})
                continue
            qm = _tensor_metrics(sq, cq)
            km = _tensor_metrics(sk, ck)
            layer_copies.append({
                "copy": i,
                "Q": qm,
                "K": km,
            })
            layer_worst = max(layer_worst, qm["max_abs"], km["max_abs"])
        per_layer[lyr] = {"copies": layer_copies, "max_across_copies": layer_worst}
        worst_md = max(worst_md, layer_worst)

    return CheckResult(name=label, passed=worst_md < 1e-5,
                       metrics={"layers": per_layer, "worst_max_abs": worst_md})


# ══════════════════════════════════════════════════════════════════
#  Output
# ══════════════════════════════════════════════════════════════════

def _print_header(dir_single: str, dir_stacked: str, n_copies: int,
                  cu_single: torch.Tensor, cu_stacked: torch.Tensor):
    T = int(cu_single[-1])
    ok_single = cu_single.numel() == 2  # B=1
    ok_stacked = (cu_stacked.numel() == n_copies + 1
                  and all(int(cu_stacked[i + 1]) - int(cu_stacked[i]) == T
                          for i in range(n_copies)))
    print(_SEP_DOUBLE)
    print("  GEMM Precision Baseline — Cross-Batch-Size Comparison")
    print(f"  Single :  {dir_single}  (1 sequence, {T} tokens)")
    print(f"  Stacked:  {dir_stacked}  ({n_copies} copies, {n_copies * T} tokens)")
    print(f"  Copies :  {n_copies}")
    print(_SEP_DOUBLE)
    print(f"  cu_seqlens single:  {cu_single.tolist()}"
          f"  {' ' + _CHECK if ok_single else ' ' + _CROSS + ' expected B=1'}")
    print(f"  cu_seqlens stacked: {cu_stacked.tolist()}"
          f"  {' ' + _CHECK if ok_stacked else ' ' + _CROSS + ' expected B=' + str(n_copies)}")
    print()


def _print_plain_table(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [{r.name}]  Per-layer max_abs across copies")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {_CROSS} {m['error']}\n"); return
    layers = m.get("layers", {})
    n_copies = max(len(v.get("copies", [])) for v in layers.values()) if layers else 0
    hdr = (f"  {'LAYER':>6s}  " +
           "  ".join(f"{'COPY_'+str(i):>11s}" for i in range(n_copies)) +
           f"  {'MAX':>11s}  {'MEAN':>11s}")
    print(hdr)
    print(f"  {'─' * 6}  " + "  ".join("─" * 11 for _ in range(n_copies + 2)))
    for lyr in sorted(layers):
        d = layers[lyr]
        copies = d.get("copies", [])
        vals = [c.get("max_abs", float("nan")) for c in copies]
        mx = max(v for v in vals if not math.isnan(v)) if vals else float("nan")
        mn = sum(v for v in vals if not math.isnan(v)) / max(1, sum(1 for v in vals if not math.isnan(v)))
        row = f"  {lyr:>6d}  " + "  ".join(f"{v:>11.3e}" for v in vals) + \
              f"  {mx:>11.3e}  {mn:>11.3e}"
        print(row)
    print()


def _print_rope_preqk_table(r: CheckResult, component: str):
    print(_SEP_SINGLE + f"\n  [{r.name}]  {component}  Per-layer max_abs across copies")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {_CROSS} {m['error']}\n"); return
    layers = m.get("layers", {})
    n_copies = 0
    for v in layers.values():
        n_copies = max(n_copies, len(v.get("copies", [])))
    if n_copies == 0:
        print("  no data\n"); return
    hdr = (f"  {'LAYER':>6s}  " +
           "  ".join(f"{'COPY_'+str(i):>11s}" for i in range(n_copies)) +
           f"  {'MAX':>11s}  {'MEAN':>11s}")
    print(hdr)
    print(f"  {'─' * 6}  " + "  ".join("─" * 11 for _ in range(n_copies + 2)))
    for lyr in sorted(layers):
        d = layers[lyr]
        copies = d.get("copies", [])
        vals = [c.get(component, {}).get("max_abs", float("nan")) for c in copies]
        mx = max(v for v in vals if not math.isnan(v)) if vals else float("nan")
        mn = sum(v for v in vals if not math.isnan(v)) / max(1, sum(1 for v in vals if not math.isnan(v)))
        row = f"  {lyr:>6d}  " + "  ".join(f"{v:>11.3e}" for v in vals) + \
              f"  {mx:>11.3e}  {mn:>11.3e}"
        print(row)
    print()


def _print_summary(all_results: list[CheckResult]):
    print(_SEP_DOUBLE + "\n  AGGREGATE SUMMARY  (worst max_abs across all layers & copies)")
    print(_SEP_DOUBLE)
    hdr = f"  {'FILE':<24s} {'WORST_MAX_ABS':>14s}  {'PASS?':>8s}"
    print(hdr + "\n  " + "─" * (len(hdr) - 2))
    for r in all_results:
        wm = r.metrics.get("worst_max_abs", "—")
        wm_s = f"{wm:>14.3e}" if isinstance(wm, float) else f"{wm:>14s}"
        s = f"  {_CHECK} PASS" if r.passed else f"  {_CROSS} FAIL"
        print(f"  {r.name:<24s} {wm_s}  {s}")
    print(_SEP_DOUBLE)
    print()


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="GEMM precision baseline — cross-batch-size (single vs N copies)",
        epilog=__doc__,
    )
    ap.add_argument("--dir-single", required=True,
                    help="Single-copy dump directory (batch=[A])")
    ap.add_argument("--dir-stacked", required=True,
                    help="Stacked-copies dump directory (batch=[A x N])")
    ap.add_argument("--num-copies", type=int, required=True,
                    help="Number of stacked copies N")
    ap.add_argument("--layer", type=int, default=None,
                    help="Compare specific layer 1-indexed (default: all)")
    ap.add_argument("--output", "-o", default=None,
                    help="Write JSON report to this path")
    args = ap.parse_args()

    # ── Load cu_seqlens & validate ──
    cu_single  = _load_cu_seqlens(args.dir_single)
    cu_stacked = _load_cu_seqlens(args.dir_stacked)
    if cu_single is None or cu_stacked is None:
        print(f"{_CROSS} cu_seqlens_q.pt missing in one or both dump dirs")
        return 1
    _print_header(args.dir_single, args.dir_stacked, args.num_copies,
                  cu_single, cu_stacked)

    all_results: list[CheckResult] = []

    # ── hidden_states ──
    r = _compare_plain_file(args.dir_single, args.dir_stacked,
                            "hidden_states.pt", cu_single, cu_stacked,
                            args.num_copies, args.layer, "hidden_states")
    all_results.append(r)
    _print_plain_table(r)

    # ── build_kv_input_v ──
    r = _compare_plain_file(args.dir_single, args.dir_stacked,
                            "build_kv_input_v.pt", cu_single, cu_stacked,
                            args.num_copies, args.layer, "build_kv_input_v")
    all_results.append(r)
    _print_plain_table(r)

    # ── rope_preqk ──
    r = _compare_rope_preqk(args.dir_single, args.dir_stacked,
                            cu_single, cu_stacked,
                            args.num_copies, args.layer, "rope_preqk")
    all_results.append(r)
    _print_rope_preqk_table(r, "Q")
    _print_rope_preqk_table(r, "K")

    # ── rope_freqs ──
    r = _compare_plain_file(args.dir_single, args.dir_stacked,
                            "rope_freqs.pt", cu_single, cu_stacked,
                            args.num_copies, args.layer, "rope_freqs")
    all_results.append(r)
    _print_plain_table(r)

    # ── rope_postqk ──
    r = _compare_rope_preqk(args.dir_single, args.dir_stacked,
                            cu_single, cu_stacked,
                            args.num_copies, args.layer, "rope_postqk",
                            fname="rope_postqk.pt")
    all_results.append(r)
    _print_rope_preqk_table(r, "Q")
    _print_rope_preqk_table(r, "K")

    # ── attn_outputs ──
    r = _compare_plain_file(args.dir_single, args.dir_stacked,
                            "attn_outputs.pt", cu_single, cu_stacked,
                            args.num_copies, args.layer, "attn_outputs")
    all_results.append(r)
    _print_plain_table(r)

    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, args.dir_single, args.dir_stacked,
                   tag=f"cross_batch_N{args.num_copies}", dir_off2=None)


if __name__ == "__main__":
    main()
