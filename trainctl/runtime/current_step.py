"""Live references to the in-flight training batch, retained across hook boundaries.

Single-owner, training-thread-only, mutable scratch state -- never published through
FUSE/REST directly (it can hold live CUDA tensors and Python references with no
serializable form). The public `RuntimeSnapshot` only ever receives scalar/serialized
state derived from it, e.g. `trainctl.runtime.state.ExceptionBreakpointInfo`.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class CurrentStep:
    """The batch/loss/exception belonging to the training step in progress, if any.

    Attributes:
        epoch: Epoch of the in-flight batch.
        global_step: Global step of the in-flight batch.
        batch_idx: Batch index within the epoch.
        started_ns: `time.perf_counter_ns()` value at batch start.
        batch: The live batch object passed to `training_step`. Retained only so a
            backward exception has something to inspect/save; not copied.
        loss: The live loss tensor, set at `on_before_backward`.
        exception: The exception a failed `backward()` raised, if any.
    """

    epoch: int | None = None
    global_step: int | None = None
    batch_idx: int | None = None
    started_ns: int | None = None
    batch: Any = None
    loss: Any = None
    exception: BaseException | None = None

    def begin(
        self, batch: Any, batch_idx: int, epoch: int, global_step: int, started_ns: int
    ) -> None:
        """Starts tracking a new batch, discarding any previous one's references."""
        self.epoch = epoch
        self.global_step = global_step
        self.batch_idx = batch_idx
        self.started_ns = started_ns
        self.batch = batch
        self.loss = None
        self.exception = None

    def clear(self) -> None:
        """Drops live references at batch end, leaving only the scalar step markers."""
        self.batch = None
        self.loss = None
        self.exception = None
