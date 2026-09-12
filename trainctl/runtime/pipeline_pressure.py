"""Cheap, always-on wall-clock phase timing for the Lightning training loop.

Not a profiler -- a small rolling summary of where time goes at the Lightning-hook
level (input gap, forward+loss, backward, post-backward), built from a handful of
`time.perf_counter_ns()` calls per batch. See `trainctl.runtime.current_step` for the
separate concern of retaining a batch/loss/exception across hook boundaries.
"""

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PipelinePressureSample:
    """One batch's phase timing breakdown, in nanoseconds.

    Attributes:
        epoch: Epoch this batch belonged to.
        global_step: Global step at batch end.
        batch_idx: Batch index within the epoch.
        input_gap_ns: Time between the previous batch's end and this batch's start --
            some mix of DataLoader wait, batch transfer, and framework overhead; not
            proven to be pure DataLoader time. `None` for the first batch.
        forward_loss_ns: Time from batch start to just before backward -- training_step,
            forward, and loss construction. `None` if backward never started.
        backward_ns: Time spent inside `backward()`. `None` if backward never finished.
        post_backward_ns: Time from after backward to batch end -- gradient handling,
            optimizer work, bookkeeping. `None` if backward never finished.
        batch_total_ns: Time from batch start to batch end.
        optimizer_step_ns: Time inside `optimizer_step`, when that capability is
            supported for the active strategy; `None` otherwise.
    """

    epoch: int
    global_step: int
    batch_idx: int
    input_gap_ns: int | None
    forward_loss_ns: int | None
    backward_ns: int | None
    post_backward_ns: int | None
    batch_total_ns: int | None
    optimizer_step_ns: int | None = None


class PipelinePressureTracker:
    """Records per-batch phase timestamps and keeps a rolling window of samples.

    Attributes:
        window: Maximum number of samples retained.
    """

    def __init__(self, window: int) -> None:
        self.window = window
        self._samples: deque[PipelinePressureSample] = deque(maxlen=window)
        self._batch_start_ns: int | None = None
        self._before_backward_ns: int | None = None
        self._after_backward_ns: int | None = None
        self._before_optimizer_ns: int | None = None
        self._optimizer_step_ns: int | None = None
        self._prev_batch_end_ns: int | None = None
        self._epoch = 0
        self._global_step = 0
        self._batch_idx = 0

    def batch_start(
        self, now_ns: int, epoch: int, global_step: int, batch_idx: int
    ) -> None:
        """Records the start of a new batch. Call from `on_train_batch_start`."""
        self._batch_start_ns = now_ns
        self._before_backward_ns = None
        self._after_backward_ns = None
        self._before_optimizer_ns = None
        self._optimizer_step_ns = None
        self._epoch = epoch
        self._global_step = global_step
        self._batch_idx = batch_idx

    def before_backward(self, now_ns: int) -> None:
        """Records the point just before `backward()`. Call from `on_before_backward`."""
        self._before_backward_ns = now_ns

    def after_backward(self, now_ns: int) -> None:
        """Records the point just after `backward()`. Call from `on_after_backward`."""
        self._after_backward_ns = now_ns

    def before_optimizer(self, now_ns: int) -> None:
        """Records the point just before the optimizer step.

        Call from `on_before_optimizer_step`.
        """
        self._before_optimizer_ns = now_ns

    def optimizer_step(self, now_ns: int) -> None:
        """Records optimizer-step duration, when that capability is supported."""
        if self._before_optimizer_ns is not None:
            self._optimizer_step_ns = now_ns - self._before_optimizer_ns

    def batch_end(self, now_ns: int) -> PipelinePressureSample:
        """Finalizes the current batch's sample and appends it to the rolling window."""
        input_gap_ns = (
            self._batch_start_ns - self._prev_batch_end_ns
            if self._batch_start_ns is not None and self._prev_batch_end_ns is not None
            else None
        )
        forward_loss_ns = (
            self._before_backward_ns - self._batch_start_ns
            if self._before_backward_ns is not None and self._batch_start_ns is not None
            else None
        )
        backward_ns = (
            self._after_backward_ns - self._before_backward_ns
            if self._after_backward_ns is not None
            and self._before_backward_ns is not None
            else None
        )
        post_backward_ns = (
            now_ns - self._after_backward_ns
            if self._after_backward_ns is not None
            else None
        )
        batch_total_ns = (
            now_ns - self._batch_start_ns if self._batch_start_ns is not None else None
        )
        sample = PipelinePressureSample(
            epoch=self._epoch,
            global_step=self._global_step,
            batch_idx=self._batch_idx,
            input_gap_ns=input_gap_ns,
            forward_loss_ns=forward_loss_ns,
            backward_ns=backward_ns,
            post_backward_ns=post_backward_ns,
            batch_total_ns=batch_total_ns,
            optimizer_step_ns=self._optimizer_step_ns,
        )
        self._samples.append(sample)
        self._prev_batch_end_ns = now_ns
        return sample

    def latest(self) -> PipelinePressureSample | None:
        """Returns the most recent sample, or `None` before the first batch completes."""
        return self._samples[-1] if self._samples else None

    def summary(self) -> dict[str, Any]:
        """Computes rolling phase statistics over the current window, lazily on read."""
        samples = list(self._samples)
        if not samples:
            return {"window": self.window, "samples": 0}
        totals = [s.batch_total_ns for s in samples if s.batch_total_ns is not None]
        phase_stats: dict[str, Any] = {}
        for phase in ("input_gap_ns", "forward_loss_ns", "backward_ns", "post_backward_ns"):
            values = [
                getattr(s, phase) for s in samples if getattr(s, phase) is not None
            ]
            if not values:
                continue
            phase_stats[phase.removesuffix("_ns")] = {
                "p50_ms": _percentile(values, 0.50) / 1e6,
                "p95_ms": _percentile(values, 0.95) / 1e6,
                "max_ms": max(values) / 1e6,
                "fraction": (statistics.mean(values) / statistics.mean(totals))
                if totals
                else None,
            }
        largest_phase = (
            max(phase_stats, key=lambda name: phase_stats[name]["p50_ms"])
            if phase_stats
            else None
        )
        return {
            "window": self.window,
            "samples": len(samples),
            "latest_step": samples[-1].global_step,
            "batch_total_ms": (
                {
                    "p50": _percentile(totals, 0.50) / 1e6,
                    "p95": _percentile(totals, 0.95) / 1e6,
                }
                if totals
                else None
            ),
            "phases": phase_stats,
            "largest_phase": largest_phase,
        }


def _percentile(values: list[int], fraction: float) -> float:
    """Nearest-rank percentile over `values`; not interpolated, fine for a rolling summary."""
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return float(ordered[index])
