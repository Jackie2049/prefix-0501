"""Benchmark analyzer: compare prefix-sharing ON vs OFF results.

Consumes the profiling artifacts produced by ``PROFILE_OUTPUT_DIR`` and the
prefix-sharing audit log, then emits a comparison summary.

Two analysis passes:

1. **PS-on self-check** — parses ``[PS][audit]`` lines from the PS-on run
   log and verifies that the detected prefix reuse matches the dataset
   manifest's expectations (reused_valid_token_ratio, provider/reuser
   counts). Warns on mismatch.

2. **PS-on vs PS-off comparison** — aggregates the timing CSV
   (``microbatch_fwd_bwd`` per-phase stats) and memory CSV (peak
   allocated/reserved) from both runs and prints a side-by-side table with
   absolute delta and percentage change.

Usage::

    python -m prefix_sharing.tools.analyze \\
        --ps-off-dir results/ps_off \\
        --ps-on-dir  results/ps_on \\
        --manifest   prefix_sharing/tools/data/manifest.json \\
        --log-on     results/ps_on/train.log

The result directories are expected to contain ``timing_trace_rank*.csv``
and ``memory_trace_rank*.csv`` produced by the profiling wiring.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def _load_timing_csvs(result_dir: str) -> dict[str, list[float]]:
    """Load all timing_trace_rank*.csv and aggregate durations per phase."""
    phases: dict[str, list[float]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(result_dir, "timing_trace_rank*.csv"))):
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                phases[row["phase"]].append(float(row["duration_s"]))
    return phases


def _load_memory_csvs(result_dir: str) -> dict[str, float]:
    """Load all memory_trace_rank*.csv and return peak allocated/reserved."""
    peak_alloc = 0.0
    peak_reserved = 0.0
    found = False
    for path in sorted(glob.glob(os.path.join(result_dir, "memory_trace_rank*.csv"))):
        found = True
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                peak_alloc = max(peak_alloc, float(row["allocated_gb"]))
                peak_reserved = max(peak_reserved, float(row["reserved_gb"]))
    if not found:
        return {}
    return {"peak_allocated_gb": peak_alloc, "peak_reserved_gb": peak_reserved}


def _phase_stats(durations: list[float]) -> dict[str, float]:
    if not durations:
        return {}
    s = sorted(durations)
    n = len(s)
    avg = sum(s) / n
    return {
        "count": n,
        "total_s": round(sum(s), 4),
        "avg_s": round(avg, 4),
        "median_s": round(statistics.median(s), 4),
        "min_s": round(s[0], 4),
        "max_s": round(s[-1], 4),
        "p99_s": round(s[max(0, int(n * 0.99) - 1)], 4),
    }


# ---------------------------------------------------------------------------
# audit log parsing (PS-on self-check)
# ---------------------------------------------------------------------------

_AUDIT_FIELDS = [
    "forward_id",
    "micro_batch_id",
    "batch_size",
    "original_tokens",
    "kept_valid_tokens",
    "kept_padded_tokens",
    "reused_valid_tokens",
    "reused_valid_token_ratio",
    "provider_count",
    "reuser_count",
    "sharing_group_count",
    "expected_restore_count",
    "actual_restore_count",
]


def parse_audit_lines(log_path: str) -> list[dict[str, Any]]:
    """Extract [PS][audit]...summary lines from a training log."""
    if not log_path or not os.path.exists(log_path):
        return []
    entries: list[dict[str, Any]] = []
    with open(log_path, errors="replace") as f:
        for line in f:
            if "[PS][audit]" not in line or "summary:" not in line:
                continue
            entry: dict[str, Any] = {}
            for field in _AUDIT_FIELDS:
                marker = f"{field}="
                idx = line.find(marker)
                if idx < 0:
                    continue
                rest = line[idx + len(marker):].split()[0].rstrip(",")
                try:
                    entry[field] = int(rest)
                except ValueError:
                    try:
                        entry[field] = float(rest)
                    except ValueError:
                        entry[field] = rest
            if entry:
                entries.append(entry)
    return entries


def expected_reuse_from_manifest(manifest_path: str) -> dict[str, Any]:
    """Compute expected reuse stats from the dataset manifest."""
    if not manifest_path or not os.path.exists(manifest_path):
        return {}
    with open(manifest_path) as f:
        manifest = json.load(f)
    total_tokens = 0
    reused_tokens = 0
    for group in manifest["groups"]:
        for chain in group["chains"]:
            for s in chain["samples"]:
                total_tokens += s["seq_len"]
                reused_tokens += s.get("expected_shared_prefix_len", 0)
    ratio = reused_tokens / total_tokens if total_tokens else 0.0
    return {
        "total_samples": manifest["total_samples"],
        "expected_total_tokens": total_tokens,
        "expected_reused_tokens": reused_tokens,
        "expected_reused_valid_token_ratio": round(ratio, 4),
    }


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _fmt_pct(old: float, new: float) -> str:
    if old == 0:
        return "n/a"
    delta = (new - old) / old * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep)
    for row in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))


def compare_timing(ps_off: dict[str, list[float]], ps_on: dict[str, list[float]]) -> None:
    print("\n" + "=" * 70)
    print("TIMING COMPARISON (per microbatch_fwd_bwd, lower is faster)")
    print("=" * 70)
    phases = sorted(set(ps_off) | set(ps_on))
    rows: list[list[str]] = []
    for phase in phases:
        off = _phase_stats(ps_off.get(phase, []))
        on = _phase_stats(ps_on.get(phase, []))
        if not off and not on:
            continue
        for stat in ("avg_s", "median_s", "p99_s"):
            ov = off.get(stat, float("nan"))
            nv = on.get(stat, float("nan"))
            rows.append(
                [
                    f"{phase}.{stat}",
                    f"{ov:.4f}" if off else "-",
                    f"{nv:.4f}" if on else "-",
                    f"{nv - ov:+.4f}" if off and on else "-",
                    _fmt_pct(ov, nv) if off and on else "-",
                ]
            )
    _print_table(rows, ["metric", "PS-off", "PS-on", "delta", "delta%"])
    # headline speedup
    off_med = _phase_stats(ps_off.get("microbatch_fwd_bwd", [])).get("median_s")
    on_med = _phase_stats(ps_on.get("microbatch_fwd_bwd", [])).get("median_s")
    if off_med and on_med:
        speedup = off_med / on_med
        print(f"\n  microbatch_fwd_bwd speedup (PS-off/PS-on median): {speedup:.3f}x")


def compare_memory(ps_off: dict[str, float], ps_on: dict[str, float]) -> None:
    print("\n" + "=" * 70)
    print("MEMORY COMPARISON (peak across ranks, lower is better)")
    print("=" * 70)
    rows: list[list[str]] = []
    for key in ("peak_allocated_gb", "peak_reserved_gb"):
        ov = ps_off.get(key)
        nv = ps_on.get(key)
        rows.append(
            [
                key,
                f"{ov:.3f} GiB" if ov is not None else "-",
                f"{nv:.3f} GiB" if nv is not None else "-",
                f"{nv - ov:+.3f}" if ov is not None and nv is not None else "-",
                _fmt_pct(ov, nv) if ov is not None and nv is not None else "-",
            ]
        )
    _print_table(rows, ["metric", "PS-off", "PS-on", "delta", "delta%"])


def self_check(audit_entries: list[dict[str, Any]], expected: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("PS-ON SELF-CHECK (reuse verification)")
    print("=" * 70)
    if not audit_entries:
        print("  WARNING: no [PS][audit] summary lines found in PS-on log.")
        print("  Prefix sharing may not have activated, or the log path is wrong.")
        return
    # take the last (most complete) audit entry
    last = audit_entries[-1]
    print(f"  audit entries found: {len(audit_entries)}")
    for k in ("batch_size", "original_tokens", "kept_valid_tokens",
              "reused_valid_tokens", "reused_valid_token_ratio",
              "provider_count", "reuser_count", "actual_restore_count"):
        if k in last:
            print(f"    {k} = {last[k]}")

    if expected:
        exp_ratio = expected.get("expected_reused_valid_token_ratio")
        act_ratio = last.get("reused_valid_token_ratio")
        if exp_ratio is not None and act_ratio is not None:
            print(f"\n  expected reused_valid_token_ratio (from manifest): {exp_ratio}")
            print(f"  actual   reused_valid_token_ratio (from audit)   : {act_ratio}")
            # tolerate deviation — planner grouping may differ from manifest's ideal
            if abs(exp_ratio - act_ratio) > 0.15:
                print("  ⚠ ratio deviates >15% from manifest expectation — investigate planner grouping")
            else:
                print("  ✓ reuse ratio within 15% of manifest expectation")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prefix-sharing ON vs OFF benchmark results.")
    parser.add_argument("--ps-off-dir", required=True, help="PS-off results dir (PROFILE_OUTPUT_DIR).")
    parser.add_argument("--ps-on-dir", required=True, help="PS-on results dir (PROFILE_OUTPUT_DIR).")
    parser.add_argument("--manifest", default=None, help="Dataset manifest.json for reuse self-check.")
    parser.add_argument("--log-on", default=None, help="PS-on training stdout log (for audit parsing).")
    parser.add_argument("--json-out", default=None, help="Optional: write summary as JSON to this path.")
    args = parser.parse_args()

    print(f"PS-off dir: {args.ps_off_dir}")
    print(f"PS-on  dir: {args.ps_on_dir}")

    off_timing = _load_timing_csvs(args.ps_off_dir)
    on_timing = _load_timing_csvs(args.ps_on_dir)
    off_mem = _load_memory_csvs(args.ps_off_dir)
    on_mem = _load_memory_csvs(args.ps_on_dir)

    compare_timing(off_timing, on_timing)
    compare_memory(off_mem, on_mem)

    audit = parse_audit_lines(args.log_on)
    expected = expected_reuse_from_manifest(args.manifest)
    self_check(audit, expected)

    if args.json_out:
        summary = {
            "ps_off_timing": {p: _phase_stats(d) for p, d in off_timing.items()},
            "ps_on_timing": {p: _phase_stats(d) for p, d in on_timing.items()},
            "ps_off_memory": off_mem,
            "ps_on_memory": on_mem,
            "audit_entries": audit,
            "expected_reuse": expected,
        }
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to {args.json_out}")


if __name__ == "__main__":
    main()
