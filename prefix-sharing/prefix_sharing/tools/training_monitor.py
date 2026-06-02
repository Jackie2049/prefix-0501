"""Training monitoring utilities for prefix sharing profiling.

Provides ``MemoryMonitor`` (continuous device memory sampling) and
``Stopwatch`` (lap / start-stop timing) for debugging HBM usage and
bottlenecks during prefix-sharing training.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MemoryMonitor
# ---------------------------------------------------------------------------


@dataclass
class MemorySnapshot:
    """A single memory-poll sample."""

    timestamp: float
    allocated_gb: float  # memory_allocated()  – tensor 占用
    reserved_gb: float   # memory_reserved()  – 分配器保留（含缓存）
    occupied_gb: float   # mem_get_info() → total - free – 设备级已占用


class MemoryMonitor:
    """Background-thread memory sampler for CUDA / NPU devices.

    Each sample records three metrics:

    * **allocated** – memory actively holding tensors (``memory_allocated()``)
    * **reserved**  – memory reserved by the caching allocator (``memory_reserved()``)
    * **occupied**  – real device-level used memory (``mem_get_info()`` total - free)

    Call ``start()`` before the region of interest and ``stop()`` after::

        mon = MemoryMonitor(interval=0.05)   # sample every 50 ms
        mon.start()
        ... forward + backward ...
        mon.stop()
        s = mon.summary()
        logger.warning(
            f"Peak  occupied={s['peak_occupied_gib']:.1f} GiB  "
            f"reserved={s['peak_reserved_gib']:.1f} GiB  "
            f"allocated={s['peak_allocated_gib']:.1f} GiB"
        )
    """

    def __init__(
        self,
        device_type: Literal["cuda", "npu"] | None = None,
        interval: float = 0.1,
    ) -> None:
        self.interval = interval
        self._samples: list[MemorySnapshot] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self.device_type = self._resolve_device(device_type)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin background sampling."""
        self._samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background sampling and join the thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)

    @property
    def total_gb(self) -> float:
        """Total device memory (GiB)."""
        return self._get_total_gb()

    def snapshot(self) -> MemorySnapshot:
        """One-shot query of all memory metrics without starting the thread."""
        return self._sample_once()

    # -- per-metric peaks -------------------------------------------------

    @property
    def peak_allocated_gb(self) -> float:
        if not self._samples:
            return 0.0
        return max(s.allocated_gb for s in self._samples)

    @property
    def peak_reserved_gb(self) -> float:
        if not self._samples:
            return 0.0
        return max(s.reserved_gb for s in self._samples)

    @property
    def peak_occupied_gb(self) -> float:
        if not self._samples:
            return 0.0
        return max(s.occupied_gb for s in self._samples)

    def summary(self) -> dict:
        """Aggregate all three memory metrics.

        Returns keys ``peak_{allocated,reserved,occupied}_gib``,
        ``avg_{allocated,reserved,occupied}_gib``, plus metadata.
        """
        if not self._samples:
            return {"device_type": self.device_type, "num_samples": 0}
        al = [s.allocated_gb for s in self._samples]
        rs = [s.reserved_gb for s in self._samples]
        oc = [s.occupied_gb for s in self._samples]
        return {
            "device_type": self.device_type,
            "num_samples": len(self._samples),
            "interval_s": self.interval,
            "peak_allocated_gib": round(max(al), 3),
            "peak_reserved_gib": round(max(rs), 3),
            "peak_occupied_gib": round(max(oc), 3),
            "avg_allocated_gib": round(sum(al) / len(al), 3),
            "avg_reserved_gib": round(sum(rs) / len(rs), 3),
            "avg_occupied_gib": round(sum(oc) / len(oc), 3),
        }

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _resolve_device(self, device_type: str | None) -> str | None:
        import torch as _torch

        if device_type is not None:
            return device_type
        try:
            if _torch.npu.is_available():
                return "npu"
        except Exception:
            pass
        if _torch.cuda.is_available():
            return "cuda"
        return None  # CPU – memory monitoring disabled

    def _sample_once(self) -> MemorySnapshot:
        import torch as _torch

        if self.device_type == "npu":
            allocated = _torch.npu.memory_allocated()
            reserved = _torch.npu.memory_reserved()
            free, total = _torch.npu.mem_get_info()
        elif self.device_type == "cuda":
            allocated = _torch.cuda.memory_allocated()
            reserved = _torch.cuda.memory_reserved()
            free, total = _torch.cuda.mem_get_info()
        else:
            return MemorySnapshot(time.time(), 0.0, 0.0, 0.0)

        return MemorySnapshot(
            timestamp=time.time(),
            allocated_gb=allocated / (1024**3),
            reserved_gb=reserved / (1024**3),
            occupied_gb=(total - free) / (1024**3),
        )

    def _get_total_gb(self) -> float:
        import torch as _torch

        if self.device_type == "npu":
            _, total = _torch.npu.mem_get_info()
            return total / (1024**3)
        if self.device_type == "cuda":
            _, total = _torch.cuda.mem_get_info()
            return total / (1024**3)
        return 0.0

    def _sample_loop(self) -> None:
        while self._running:
            self._samples.append(self._sample_once())
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# Stopwatch
# ---------------------------------------------------------------------------


class Stopwatch:
    """Lightweight timing for training phases with distribution support.

    Supports three usage modes:

    **Start / stop** – for repeatable phases (e.g. per-microbatch timing)::

        sw = Stopwatch()
        for _ in range(n):
            sw.start("microbatch")
            ... forward + backward ...
            sw.stop("microbatch")
        sw.start("update")
        ... optimizer ...
        sw.stop("update")

        print(sw.summary())
        # {
        #   "phases": {
        #     "microbatch": {"total_s": 8.2, "count": 4, "avg_s": 2.05,
        #                    "min_s": 1.98, "max_s": 2.11, "median_s": 2.04,
        #                    "p99_s": 2.108, "stddev_s": 0.05},
        #     "update":     {"total_s": 4.3, "count": 1, ...}
        #   }
        # }

    **Sub-phase nesting** — time sub-stages within a microbatch::

        sw.start("microbatch")
        sw.start("forward")
        ... forward ...
        sw.stop("forward")          # recorded as a sample of "forward"
        sw.start("backward")
        ... backward ...
        sw.stop("backward")
        sw.stop("microbatch")       # also recorded as a sample of "microbatch"
        # total_s of "microbatch" == sum of all microbatch durations
        # (sub-phases are a separate dimension, not subtracted)

    **Lap** – for linear checkpoints::

        sw.lap("forward_done")
        sw.lap("backward_done")
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = OrderedDict()  # name → [durations]
        self._running: dict[str, float] = OrderedDict()        # name → start_time
        self._laps: list[tuple[str, float]] = []               # (name, timestamp)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def lap(self, name: str = "") -> None:
        """Record an instantaneous checkpoint with label *name*."""
        self._laps.append((name, time.perf_counter()))

    def start(self, name: str) -> None:
        """Begin timing for *name*.

        Supports nesting: calling ``start('forward')`` while ``start('microbatch')``
        is already running works fine — each named timer runs independently.
        """
        self._running[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """End timing for *name*, record an individual sample, and return elapsed seconds."""
        start = self._running.pop(name, None)
        if start is None:
            logger.warning("Stopwatch.stop('%s'): timer was never started", name)
            return 0.0
        elapsed = time.perf_counter() - start
        self._samples.setdefault(name, [])
        self._samples[name].append(elapsed)
        return elapsed

    def reset(self) -> None:
        """Clear all recorded samples, laps, and running timers."""
        self._samples.clear()
        self._laps.clear()
        self._running.clear()

    def elapsed(self, name: str) -> float:
        """Return total cumulative time for phase *name*."""
        return sum(self._samples.get(name, []))

    def raw_samples(self, name: str) -> list[float]:
        """Return the individual duration list for *name* (for custom analysis)."""
        return list(self._samples.get(name, []))

    def summary(self) -> dict:
        """Aggregate timing stats with distribution for each phase.

        Returns::

            {
                "total_s": 12.5,
                "phases": {
                    "microbatch": {"total_s", "count", "avg_s",
                                   "min_s", "max_s", "median_s",
                                   "p99_s", "stddev_s"},
                    ...
                }
            }
        """
        phases: OrderedDict[str, dict] = OrderedDict()
        for name, samples in self._samples.items():
            cnt = len(samples)
            if cnt == 0:
                continue
            total_s = sum(samples)
            sorted_s = sorted(samples)
            avg_s = total_s / cnt
            stddev_s = (sum((s - avg_s) ** 2 for s in samples) / cnt) ** 0.5
            p99_idx = max(0, int(cnt * 0.99) - 1)

            phases[name] = {
                "total_s": round(total_s, 4),
                "count": cnt,
                "avg_s": round(avg_s, 4),
                "min_s": round(sorted_s[0], 4),
                "max_s": round(sorted_s[-1], 4),
                "median_s": round(sorted_s[cnt // 2], 4),
                "p99_s": round(sorted_s[p99_idx], 4),
                "stddev_s": round(stddev_s, 4),
            }

        return {
            "total_s": round(sum(sum(v) for v in self._samples.values()), 4),
            "phases": phases,
        }

    def print_lap_timeline(self, logger_fn=None) -> None:
        """Pretty-print the lap timeline (useful for a single microbatch walk-through).

        *logger_fn* is called instead of ``print`` if given (e.g. ``logger.warning``).
        """
        if len(self._laps) < 2:
            return
        out = []
        t0 = self._laps[0][1]
        for name, ts in self._laps:
            delta = ts - t0
            out.append(f"    [{delta:8.4f}s] {name}")
        text = "\n".join(out)
        if logger_fn:
            logger_fn(text)
        else:
            print(text)
