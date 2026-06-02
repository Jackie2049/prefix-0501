"""Precision validation utilities for prefix sharing.

This module provides tools to compare the numerical output of prefix-sharing
optimized runs against baseline (prefix-sharing disabled) runs on the same
input data and model weights.

AttentionProbe
--------------
Captures per-layer SelfAttention outputs via forward hooks.  The hooks fire
regardless of whether ``maybe_run_prefix_sharing_attention`` intercepted the
call or the normal Megatron attention path was taken, so the same probe works
for both the optimized and baseline forward pass.

Usage::

    from prefix_sharing.tools.precision_validator import (
        AttentionProbe,
        PrecisionValidator,
    )

    validator = PrecisionValidator(actor)
    probe = AttentionProbe(actor.actor_module)

    # Baseline forward
    validator.save_weights()
    with validator.baseline_mode():
        with probe:
            ... model forward (loss.backward(), optimizer.step()) ...
    baseline = dict(probe.collected)

    # Prefix sharing forward (reuse same weights)
    validator.restore_weights()
    probe.clear()
    with probe:
        ... model forward ...
    ps = dict(probe.collected)

    # Compare attention outputs layer-by-layer
    for layer_name in baseline:
        validator.compare_tensors(f"attention/{layer_name}",
                                  baseline[layer_name], ps[layer_name])
    validator.print_report()
"""

import json
import os
import time
from contextlib import contextmanager
from typing import Any

import torch

import logging
precision_logger = logging.getLogger(__file__)


class AttentionProbe:
    """Capture ``SelfAttention`` forward outputs for precision comparison.

    Registers ``register_forward_hook`` on every module whose class name is
    ``"SelfAttention"``.  The hook fires *after* ``forward()`` returns — this
    captures the final attention output (``core_attn_out`` after
    ``linear_proj``) regardless of whether prefix sharing or normal Megatron
    attention was executed.

    Captured tensors are detached and moved to CPU to minimise memory impact.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.collected: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Discard all captured tensors (call between baseline / ps runs)."""
        self.collected.clear()

    def close(self) -> None:
        """Remove all forward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    # Context-manager protocol – alias for clear … forward … collect
    # ------------------------------------------------------------------

    def __enter__(self) -> "AttentionProbe":
        self.collected.clear()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _register(self) -> None:
        def _is_self_attention(mod: torch.nn.Module) -> bool:
            cls_name = type(mod).__name__
            if cls_name == "SelfAttention":
                return True
            return False

        for name, mod in self.model.named_modules():
            if _is_self_attention(mod):
                handle = mod.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

        if not self._handles:
            precision_logger.warning(
                "[AttentionProbe] No SelfAttention modules found in model."
            )

    def _make_hook(
        self, module_name: str
    ) -> Any:
        def _hook(_module: Any, _input: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            self.collected[module_name] = tensor.detach().cpu()
        return _hook


class PrecisionValidator:
    """Validate prefix sharing precision by comparing optimized vs baseline."""

    def __init__(self, actor: Any):
        self.actor = actor
        self._param_snapshots: dict[str, torch.Tensor] = {}
        self.reports: list[dict[str, Any]] = []

    def _get_modules(self):
        """Return a flat list of nn.Module objects from actor_module."""
        actor_module = self.actor.actor_module
        if isinstance(actor_module, (list, tuple, torch.nn.ModuleList)):
            return list(actor_module)
        return [actor_module]

    def save_weights(self) -> None:
        """Snapshot current model parameters."""
        self._param_snapshots = {}
        for idx, module in enumerate(self._get_modules()):
            for name, param in module.named_parameters():
                key = f"{idx}.{name}"
                self._param_snapshots[key] = param.detach().clone()

    def restore_weights(self) -> None:
        """Restore model parameters from snapshots."""
        with torch.no_grad():
            for idx, module in enumerate(self._get_modules()):
                for name, param in module.named_parameters():
                    key = f"{idx}.{name}"
                    if key in self._param_snapshots:
                        param.copy_(self._param_snapshots[key])

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

        precision_logger.warning("\n" + "=" * 70)
        precision_logger.warning("  PREFIX SHARING PRECISION VALIDATION REPORT")
        precision_logger.warning("=" * 70)

        all_pass = True
        for r in self.reports:
            status = "PASS" if r.get("is_close", True) else "FAIL"
            if status == "FAIL":
                all_pass = False

            if r.get("status") == "skipped":
                precision_logger.warning(f"  [SKIP] {r['name']}: {r['reason']}")
            elif r.get("status") == "error":
                precision_logger.warning(f"  [ERR ] {r['name']}: {r['reason']}")
            else:
                extra = ""
                if "max_rel_diff" in r:
                    extra = f", max_rel_diff={r['max_rel_diff']:.6e}"
                precision_logger.warning(
                    f"  [{status}] {r['name']}: shape={r['shape']}, "
                    f"max_diff={r['max_diff']:.6e}, mean_diff={r['mean_diff']:.6e}"
                    f"{extra}"
                )

        precision_logger.warning("-" * 70)
        precision_logger.warning(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        precision_logger.warning("=" * 70 + "\n")

        try:
            report_dir = "precision_reports"
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(
                report_dir, f"precision_report_{int(time.time())}.json"
            )
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(self.reports, f, indent=2, ensure_ascii=False)
            precision_logger.warning(f"  Detailed report saved to: {report_path}\n")
        except Exception as exc:
            precision_logger.warning(f"  Warning: failed to save report: {exc}\n")
