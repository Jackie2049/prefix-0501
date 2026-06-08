"""Training monitoring utilities for prefix sharing profiling.

Provides three core classes:

* ``PerfRecorder`` – unified performance recorder that combines Stopwatch-style
  hierarchical ``start/stop`` timing with ``time_phase`` context-manager timing.
  Always synchronizes the GPU/NPU device before and after measurement for
  accurate kernel-level timing. Records samples, computes distribution
  statistics, and writes CSV/JSONL output.

* ``MemoryMonitor`` – background-thread memory sampler that periodically polls
  ``memory_allocated()`` / ``memory_reserved()`` on CUDA or NPU.

* ``Stopwatch`` – lightweight hierarchical timer without GPU sync (useful for
  CPU-side orchestration timing where sync overhead would dominate).

Both ``PerfRecorder`` and ``Stopwatch`` support ``save_to_csv()`` for offline
analysis; ``PerfRecorder`` additionally supports ``flush()`` for JSONL output.

Typical usage (in ``update_policy``)::

    from prefix_sharing.tools.training_monitor import (
        PerfRecorder, MemoryMonitor, training_monitor_context,
    )

    mon = MemoryMonitor(interval=0.1)
    recorder = PerfRecorder.from_env(rank=rank)
    mon.start()

    for step_id in range(num_steps):
        recorder.set_step_context(step_id)
        _profile_stopwatch_var.set(recorder)

        recorder.start("minibatch_fwd_bwd")
        for mb_idx in range(num_microbatches):
            recorder.start("microbatch_fwd")
            ...  # Megatron forward on this microbatch
            recorder.stop("microbatch_fwd")
            recorder.start("microbatch_bwd")
            ...  # Megatron backward
            recorder.stop("microbatch_bwd")
        recorder.stop("minibatch_fwd_bwd")

        recorder.start("update")
        ...  # optimizer.step()
        recorder.stop("update")

    _profile_stopwatch_var.set(None)
    mon.stop()
    recorder.save_to_csv(os.path.join(output_dir, f"timing_trace_rank{rank}.csv"))
    recorder.flush()
    mon.save_to_csv(os.path.join(output_dir, f"memory_trace_rank{rank}.csv"))
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean env var (1/true/yes/on → True)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _synchronize_device() -> None:
    """Synchronize GPU/NPU device for accurate kernel-level timing."""
    try:
        import torch
    except ModuleNotFoundError:
        return
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
            return
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _default_profile_dir() -> Path:
    return Path(os.getenv("PREFIX_SHARING_PROFILE_DIR", "prefix_sharing_perf"))


# ---------------------------------------------------------------------------
# ContextVar – propagation without modifying function signatures
# ---------------------------------------------------------------------------


@dataclass
class TrainingMonitorState:
    """Holds the active ``MemoryMonitor`` and ``PerfRecorder`` for the current
    training step.  Accessed via ``current_memory_monitor()`` /
    ``current_recorder()`` from any nested function.
    """

    mon: MemoryMonitor | None
    recorder: PerfRecorder | None


_monitor_ctx: ContextVar[TrainingMonitorState | None] = ContextVar(
    "training_monitor_context",
    default=None,
)


def current_memory_monitor() -> MemoryMonitor | None:
    """Return the active ``MemoryMonitor`` (or ``None`` outside a context)."""
    state = _monitor_ctx.get()
    return state.mon if state is not None else None


def current_recorder() -> PerfRecorder | None:
    """Return the active ``PerfRecorder`` (or ``None`` outside a context)."""
    state = _monitor_ctx.get()
    return state.recorder if state is not None else None


@contextmanager
def training_monitor_context(
    mon: MemoryMonitor | None = None,
    recorder: PerfRecorder | None = None,
) -> Iterator[TrainingMonitorState]:
    """Context manager that makes *mon* and *recorder* available via getters."""
    state = TrainingMonitorState(mon=mon, recorder=recorder)
    token = _monitor_ctx.set(state)
    try:
        yield state
    finally:
        _monitor_ctx.reset(token)


# ---------------------------------------------------------------------------
# MemoryMonitor – background-thread device memory sampler
# ---------------------------------------------------------------------------


@dataclass
class MemorySnapshot:
    """A single memory-poll sample."""

    timestamp: float
    allocated_gb: float  # memory_allocated() – tensor 占用
    reserved_gb: float   # memory_reserved() – 分配器保留（含缓存）


class MemoryMonitor:
    """Background-thread memory sampler for CUDA / NPU devices.

    Each sample records two metrics:

    * **allocated** – memory actively holding tensors (``memory_allocated()``)
    * **reserved**  – memory reserved by the caching allocator (``memory_reserved()``)

    Call ``start()`` before the region of interest and ``stop()`` after::

        mon = MemoryMonitor(interval=0.05)
        mon.start()
        ... forward + backward ...
        mon.stop()
        s = mon.summary()
        logger.warning(
            f"Peak  allocated={s['peak_allocated_gib']:.1f} GiB  "
            f"reserved={s['peak_reserved_gib']:.1f} GiB"
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

    # -- public -----------------------------------------------------------

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

    def snapshot(self) -> MemorySnapshot:
        """One-shot query of all memory metrics without starting the thread."""
        return self._sample_once()

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

    def summary(self) -> dict:
        """Aggregate memory metrics."""
        if not self._samples:
            return {"device_type": self.device_type, "num_samples": 0}
        al = [s.allocated_gb for s in self._samples]
        rs = [s.reserved_gb for s in self._samples]
        return {
            "device_type": self.device_type,
            "num_samples": len(self._samples),
            "interval_s": self.interval,
            "peak_allocated_gib": round(max(al), 3),
            "peak_reserved_gib": round(max(rs), 3),
            "avg_allocated_gib": round(sum(al) / len(al), 3),
            "avg_reserved_gib": round(sum(rs) / len(rs), 3),
        }

    def save_to_csv(self, path: str) -> None:
        """Save all memory samples to CSV (columns: timestamp, allocated_gb, reserved_gb)."""
        if not self._samples:
            raise RuntimeError("[MemoryMonitor] No samples to save. Did you forget to call start()/stop()?")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "allocated_gb", "reserved_gb"])
            for s in self._samples:
                writer.writerow([s.timestamp, s.allocated_gb, s.reserved_gb])
        logger.info("[MemoryMonitor] Saved %d samples to %s", len(self._samples), path)

    # -- internal ---------------------------------------------------------

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
        return None

    def _sample_once(self) -> MemorySnapshot:
        import torch as _torch

        if self.device_type == "npu":
            allocated = _torch.npu.memory_allocated()
            reserved = _torch.npu.memory_reserved()
        elif self.device_type == "cuda":
            allocated = _torch.cuda.memory_allocated()
            reserved = _torch.cuda.memory_reserved()
        else:
            return MemorySnapshot(time.time(), 0.0, 0.0)

        return MemorySnapshot(
            timestamp=time.time(),
            allocated_gb=allocated / (1024**3),
            reserved_gb=reserved / (1024**3),
        )

    def _sample_loop(self) -> None:
        while self._running:
            self._samples.append(self._sample_once())
            time.sleep(self.interval)


# ---------------------------------------------------------------------------
# Stopwatch – lightweight hierarchical timer (NO GPU sync)
# ---------------------------------------------------------------------------


class Stopwatch:
    """Lightweight hierarchical timing for CPU-side orchestration phases.

    Unlike ``PerfRecorder``, this class does **not** synchronize the GPU before
    and after each measurement – the overhead of device sync would dominate for
    short CPU-side phases. Use this for timing data movement, scheduling, and
    other host-side operations where wall-clock time is sufficient.

    Supports three modes: start/stop, sub-phase nesting, and lap checkpoints.
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = OrderedDict()
        self._sample_step_ids: dict[str, list[int | None]] = OrderedDict()
        self._running: dict[str, float] = OrderedDict()
        self._laps: list[tuple[str, float]] = []
        self._current_step_id: int | None = None

    def lap(self, name: str = "") -> None:
        """Record an instantaneous checkpoint."""
        self._laps.append((name, time.perf_counter()))

    def set_step_context(self, step_id: int | None) -> None:
        self._current_step_id = step_id

    def reset_step_context(self) -> None:
        self._current_step_id = None

    def start(self, name: str) -> None:
        """Begin timing for *name* (supports nesting)."""
        self._running[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """End timing for *name*, record sample, return elapsed seconds."""
        start = self._running.pop(name, None)
        if start is None:
            logger.warning("Stopwatch.stop('%s'): timer was never started", name)
            return 0.0
        elapsed = time.perf_counter() - start
        self._samples.setdefault(name, [])
        self._samples[name].append(elapsed)
        self._sample_step_ids.setdefault(name, [])
        self._sample_step_ids[name].append(self._current_step_id)
        return elapsed

    def reset(self) -> None:
        self._samples.clear()
        self._sample_step_ids.clear()
        self._laps.clear()
        self._running.clear()
        self._current_step_id = None

    def elapsed(self, name: str) -> float:
        return sum(self._samples.get(name, []))

    def raw_samples(self, name: str) -> list[float]:
        return list(self._samples.get(name, []))

    def summary(self) -> dict:
        """Aggregate timing stats with distribution for each phase."""
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
        """Pretty-print the lap timeline."""
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

    def save_to_csv(self, path: str) -> None:
        """Save all stopwatch samples to CSV (columns: step_id, phase, sample_index, duration_s)."""
        if not self._samples:
            raise RuntimeError("[Stopwatch] No samples to save.")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step_id", "phase", "sample_index", "duration_s"])
            for phase in self._samples:
                durations = self._samples[phase]
                step_ids = self._sample_step_ids.get(phase, [None] * len(durations))
                for idx, (d, sid) in enumerate(zip(durations, step_ids)):
                    writer.writerow([sid if sid is not None else "", phase, idx, round(d, 6)])
        logger.info("[Stopwatch] Saved %d phase samples to %s",
                     sum(len(v) for v in self._samples.values()), path)


# ---------------------------------------------------------------------------
# PerfRecorder – unified timing with strict GPU sync
# ---------------------------------------------------------------------------


class PerfRecorder:
    """Unified performance recorder combining Stopwatch-style hierarchical
    timing with ``time_phase`` context-manager timing.

    **All measurements include strict GPU/NPU device synchronization** before
    and after each timed region. This guarantees that the recorded duration
    reflects the true kernel execution time, not just the host-side launch time.
    The sync overhead is ~5–50 µs per measurement and is acceptable for
    training-phase timing where durations are typically milliseconds or longer.

    Supports two usage modes:

    **Hierarchical start/stop** (for training loop phases)::

        recorder = PerfRecorder.from_env(rank=0)
        recorder.set_step_context(step_id)
        recorder.start("minibatch_fwd_bwd")
        recorder.start("microbatch_fwd")
        ... forward ...
        recorder.stop("microbatch_fwd")
        recorder.start("microbatch_bwd")
        ... backward ...
        recorder.stop("microbatch_bwd")
        recorder.stop("minibatch_fwd_bwd")

    **Flat time_phase context manager** (for single-use regions)::

        with recorder.time_phase("build_kv", metadata={"layer": 3}):
            backend.build_kv(...)
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        rank: int = 0,
        output_dir: Path | str | None = None,
        log_every: int = 1,
    ) -> None:
        self.enabled = enabled
        self.rank = rank
        self.output_dir = Path(output_dir) if output_dir is not None else _default_profile_dir()
        self.log_every = log_every

        # Stopwatch-style state (hierarchical start/stop)
        self._sw_samples: dict[str, list[float]] = OrderedDict()
        self._sw_step_ids: dict[str, list[int | None]] = OrderedDict()
        self._sw_running: dict[str, float] = OrderedDict()
        self._laps: list[tuple[str, float]] = []
        self._current_step_id: int | None = None

        # JSONL samples (from time_phase and record)
        self._jsonl_samples: list[dict[str, Any]] = []
        self._record_count: int = 0

    @classmethod
    def from_env(cls, *, rank: int | None = None) -> PerfRecorder:
        """Construct from environment variables.

        ``PREFIX_SHARING_PROFILE=1`` enables recording (default: disabled).
        Other env vars: ``PREFIX_SHARING_PROFILE_DIR``, ``PREFIX_SHARING_PROFILE_LOG_EVERY``.
        """
        enabled = _env_flag("PREFIX_SHARING_PROFILE", default=False)
        resolved_rank = rank if rank is not None else int(
            os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
        )
        return cls(
            enabled=enabled,
            rank=resolved_rank,
            output_dir=_default_profile_dir(),
            log_every=max(1, int(os.getenv("PREFIX_SHARING_PROFILE_LOG_EVERY", "1"))),
        )

    # -- hierarchical start/stop (Stopwatch-style) ------------------------

    def set_step_context(self, step_id: int | None) -> None:
        """Tag subsequent ``stop()`` samples with this training step id."""
        self._current_step_id = step_id

    def reset_step_context(self) -> None:
        self._current_step_id = None

    def start(self, name: str) -> None:
        """Begin timing for *name* (with GPU sync for accurate measurement).

        Supports nesting: ``start('microbatch_fwd')`` while
        ``start('minibatch_fwd_bwd')`` is running works fine.
        """
        if not self.enabled:
            return
        _synchronize_device()
        self._sw_running[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """End timing for *name*, sync GPU, record sample, return elapsed seconds."""
        if not self.enabled:
            return 0.0
        _synchronize_device()
        start = self._sw_running.pop(name, None)
        if start is None:
            logger.warning("[PS][perf] PerfRecorder.stop('%s'): timer was never started", name)
            return 0.0
        elapsed = time.perf_counter() - start
        duration_ms = elapsed * 1000.0

        # Record in Stopwatch-style samples (for summary + CSV)
        self._sw_samples.setdefault(name, [])
        self._sw_samples[name].append(elapsed)
        self._sw_step_ids.setdefault(name, [])
        self._sw_step_ids[name].append(self._current_step_id)

        # Record in JSONL samples (for flush)
        payload: dict[str, Any] = {
            "timestamp": time.time(),
            "rank": self.rank,
            "phase": name,
            "duration_ms": round(duration_ms, 6),
            "step_id": self._current_step_id,
        }
        self._jsonl_samples.append(payload)
        self._record_count += 1
        if self._record_count % self.log_every == 0:
            logger.warning(
                "[PS][perf][rank=%s phase=%s step=%s] duration_ms=%.3f",
                self.rank, name, self._current_step_id, duration_ms,
            )

        return elapsed

    def elapsed(self, name: str) -> float:
        """Cumulative time for phase *name*."""
        return sum(self._sw_samples.get(name, []))

    def raw_samples(self, name: str) -> list[float]:
        """Individual duration list for *name*."""
        return list(self._sw_samples.get(name, []))

    def reset(self) -> None:
        """Clear all recorded data."""
        self._sw_samples.clear()
        self._sw_step_ids.clear()
        self._sw_running.clear()
        self._laps.clear()
        self._jsonl_samples.clear()
        self._record_count = 0
        self._current_step_id = None

    # -- flat time_phase context manager -----------------------------------

    @contextmanager
    def time_phase(self, phase: str, *, metadata: Mapping[str, Any] | None = None) -> Iterator[None]:
        """Context manager that times a named phase (with strict GPU sync)."""
        if not self.enabled:
            yield
            return
        _synchronize_device()
        start = time.perf_counter()
        try:
            yield
        finally:
            _synchronize_device()
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.record(phase, duration_ms, metadata=metadata)

    def record(
        self,
        phase: str,
        duration_ms: float,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a timing sample (called automatically by ``time_phase`` and ``stop``)."""
        if not self.enabled:
            return
        payload: dict[str, Any] = {
            "timestamp": time.time(),
            "rank": self.rank,
            "phase": phase,
            "duration_ms": round(float(duration_ms), 6),
        }
        if metadata:
            payload.update(dict(metadata))
        self._jsonl_samples.append(payload)
        self._record_count += 1
        if self._record_count % self.log_every == 0:
            logger.warning(
                "[PS][perf][rank=%s phase=%s] duration_ms=%.3f metadata=%s",
                self.rank, phase, duration_ms,
                {k: v for k, v in payload.items() if k not in {"timestamp", "rank", "phase", "duration_ms"}},
            )

    # -- lap ---------------------------------------------------------------

    def lap(self, name: str = "") -> None:
        """Record an instantaneous checkpoint timestamp."""
        self._laps.append((name, time.perf_counter()))

    # -- aggregation -------------------------------------------------------

    def summary(self) -> dict:
        """Aggregate timing stats with distribution for each phase.

        Returns::

            {
                "total_s": 12.5,
                "phases": {
                    "microbatch_fwd": {"total_s", "count", "avg_s",
                                       "min_s", "max_s", "median_s",
                                       "p99_s", "stddev_s"},
                    ...
                }
            }
        """
        if not self._sw_samples:
            return {"total_s": 0, "phases": {}}
        phases: OrderedDict[str, dict] = OrderedDict()
        for name, samples in self._sw_samples.items():
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
            "total_s": round(sum(sum(v) for v in self._sw_samples.values()), 4),
            "phases": phases,
        }

    def print_lap_timeline(self, logger_fn=None) -> None:
        """Pretty-print the lap timeline."""
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

    # -- output -----------------------------------------------------------

    def save_to_csv(self, path: str) -> None:
        """Save all hierarchical timing samples to CSV.

        Columns: ``step_id``, ``phase``, ``sample_index``, ``duration_s``.
        """
        if not self._sw_samples:
            raise RuntimeError("[PerfRecorder] No samples to save.")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step_id", "phase", "sample_index", "duration_s"])
            for phase in self._sw_samples:
                durations = self._sw_samples[phase]
                step_ids = self._sw_step_ids.get(phase, [None] * len(durations))
                for idx, (d, sid) in enumerate(zip(durations, step_ids)):
                    writer.writerow([sid if sid is not None else "", phase, idx, round(d, 6)])
        logger.info("[PerfRecorder] Saved %d phase samples to %s",
                     sum(len(v) for v in self._sw_samples.values()), path)

    def flush(self) -> None:
        """Write all JSONL samples (from both time_phase and start/stop) to file."""
        if not self.enabled or not self._jsonl_samples:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / f"prefix_sharing_perf_rank{self.rank}.jsonl"
        with output_file.open("a", encoding="utf-8") as f:
            for sample in self._jsonl_samples:
                f.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        self._jsonl_samples.clear()


# ---------------------------------------------------------------------------
# maybe_time_phase – convenience wrapper
# ---------------------------------------------------------------------------


@contextmanager
def maybe_time_phase(
    recorder: PerfRecorder | None,
    phase: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Context manager that safely handles ``None`` recorder."""
    if recorder is None or not recorder.enabled:
        yield
        return
    with recorder.time_phase(phase, metadata=metadata):
        yield