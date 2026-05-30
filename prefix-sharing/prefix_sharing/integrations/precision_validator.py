"""Precision validation utilities for prefix sharing.

This module provides tools to compare the numerical output of prefix-sharing
optimized runs against baseline (prefix-sharing disabled) runs on the same
input data and model weights.
"""

import json
import os
import time
from contextlib import contextmanager
from typing import Any

import torch


class PrecisionValidator:
    """Validate prefix sharing precision by comparing optimized vs baseline."""

    def __init__(self, actor: Any):
        self.actor = actor
        self._param_snapshots: dict[str, torch.Tensor] = {}
        self.reports: list[dict[str, Any]] = []

    def save_weights(self) -> None:
        """Snapshot current model parameters."""
        self._param_snapshots = {
            name: param.detach().clone()
            for name, param in self.actor.actor_module.named_parameters()
        }

    def restore_weights(self) -> None:
        """Restore model parameters from snapshots."""
        with torch.no_grad():
            for name, param in self.actor.actor_module.named_parameters():
                if name in self._param_snapshots:
                    param.copy_(self._param_snapshots[name])

    @contextmanager
    def baseline_mode(self):
        """Temporarily disable prefix sharing for the actor."""
        self.actor._force_disable_prefix_sharing = True
        try:
            yield
        finally:
            self.actor._force_disable_prefix_sharing = False

    def compare_tensors(
        self,
        name: str,
        t1: torch.Tensor | None,
        t2: torch.Tensor | None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> dict[str, Any]:
        """Compare two tensors and record a report entry."""

        if t1 is None and t2 is None:
            report: dict[str, Any] = {
                "name": name,
                "status": "skipped",
                "reason": "both_none",
            }
            self.reports.append(report)
            return report

        if t1 is None or t2 is None:
            report = {
                "name": name,
                "status": "error",
                "reason": "one_is_none",
            }
            self.reports.append(report)
            return report

        # Align device
        if t1.device != t2.device:
            t2 = t2.to(t1.device)

        diff = (t1 - t2).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)

        report = {
            "name": name,
            "shape": list(t1.shape),
            "device": str(t1.device),
            "dtype": str(t1.dtype),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "is_close": is_close,
            "rtol": rtol,
            "atol": atol,
        }

        if not is_close:
            flat_diff = diff.flatten()
            k = min(10, flat_diff.numel())
            topk_vals, topk_indices = torch.topk(flat_diff, k)
            report["top_diffs"] = topk_vals.tolist()
            report["top_indices"] = topk_indices.tolist()

            # Relative diff (safe against zeros)
            denom = t1.abs() + t2.abs()
            denom = torch.where(denom == 0, torch.ones_like(denom), denom)
            rel_diff = (diff * 2 / denom).flatten()
            report["max_rel_diff"] = float(rel_diff.max().item())

        self.reports.append(report)
        return report

    def print_report(self) -> None:
        """Print report to stdout and save JSON."""

        print("\n" + "=" * 70)
        print("  PREFIX SHARING PRECISION VALIDATION REPORT")
        print("=" * 70)

        all_pass = True
        for r in self.reports:
            status = "PASS" if r.get("is_close", True) else "FAIL"
            if status == "FAIL":
                all_pass = False

            if r.get("status") == "skipped":
                print(f"  [SKIP] {r['name']}: {r['reason']}")
            elif r.get("status") == "error":
                print(f"  [ERR ] {r['name']}: {r['reason']}")
            else:
                extra = ""
                if "max_rel_diff" in r:
                    extra = f", max_rel_diff={r['max_rel_diff']:.6e}"
                print(
                    f"  [{status}] {r['name']}: shape={r['shape']}, "
                    f"max_diff={r['max_diff']:.6e}, mean_diff={r['mean_diff']:.6e}"
                    f"{extra}"
                )

        print("-" * 70)
        print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        print("=" * 70 + "\n")

        try:
            report_dir = "precision_reports"
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(
                report_dir, f"precision_report_{int(time.time())}.json"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self.reports, f, indent=2, ensure_ascii=False)
            print(f"  Detailed report saved to: {report_path}\n")
        except Exception as exc:
            print(f"  Warning: failed to save report: {exc}\n")
