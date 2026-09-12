"""Immutable runtime-state snapshots and a thread-safe publish/read store."""

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ScalarMetric:
    """A single scalar metric value pushed from a Lightning hook.

    Attributes:
        value: The metric value.
        epoch: The epoch the value was produced at.
        step: The global step the value was produced at.
        updated_at: Unix timestamp the value was published.
    """

    value: float
    epoch: int
    step: int
    updated_at: float


@dataclass(frozen=True)
class RankInfo:
    """Per-rank process metadata: mostly gathered once after distributed init.

    Attributes:
        global_rank: Rank across all processes.
        local_rank: Rank within the local node.
        pid: Process ID.
        hostname: Host the rank runs on.
        device: Selected compute device, e.g. `cuda:0` or `cpu`.
        worker_pids: Live DataLoader worker process IDs. Unlike the other fields,
            this is refreshed on every publish rather than gathered once, since
            workers are spawned lazily and non-persistent ones churn per epoch.
    """

    global_rank: int
    local_rank: int
    pid: int
    hostname: str
    device: str
    worker_pids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ExceptionBreakpointInfo:
    """A backward exception currently holding the training thread.

    Diagnostic only: gradient state at this point is not a valid complete gradient
    set, and distributed collectives may be inconsistent across ranks (see
    `TrainctlRuntime.handle_backward_exception`). Not a normal cooperative safe point.

    Attributes:
        exception_type: The intercepted exception's class name.
        exception_message: `str(exception)`.
        epoch: Epoch at the time of the exception.
        global_step: Global step at the time of the exception.
        batch_idx: Batch index at the time of the exception, when known.
        rank: Global rank the exception occurred on.
        started_at: Unix timestamp the breakpoint was entered.
    """

    exception_type: str
    exception_message: str
    epoch: int
    global_step: int
    batch_idx: int | None
    rank: int
    started_at: float


@dataclass(frozen=True)
class RuntimeSnapshot:
    """A fully immutable point-in-time view of a training run, safe for cross-thread reads.

    Attributes:
        generation: Monotonically increasing publish counter.
        updated_at: Unix timestamp of this snapshot's publication.
        run_id: Unique identifier for this training run.
        status: One of `initializing`, `running`, `held`, `exception_held`, `stopping`,
            `finished`, `failed`.
        stage: Lightning trainer stage (`fit`, `validate`, `test`, `predict`), when known.
        phase: Current training phase (`train`, `validate`, `test`, `predict`, `idle`).
        epoch: Current epoch.
        global_step: Current global step.
        batch_idx: Current batch index within the epoch, when known.
        world_size: Number of distributed processes.
        ranks: Static metadata for every known rank.
        metrics: Latest scalar metrics keyed by Lightning log key.
        learning_rates: Current learning rate per optimizer param group.
        pending_commands: Number of commands still queued.
        hold_state: `none` or `held`.
        rest_host: REST bind host, once started.
        rest_port: REST bind port, once started.
        torch_debug_url: PyTorch distributed debug frontend URL, once started.
        artifacts_path: This run's checkpoint/snapshot/debug-capture directory, once
            resolved -- distinct per run (colocated under Lightning's own
            `trainer.log_dir` when available), so distinguishes concurrent runs that
            share a host and launch directory.
        lightning_backend: Selected Lightning backend family name.
        lightning_version: Selected Lightning backend version string.
        exception: The active backward-exception breakpoint, when `status ==
            "exception_held"`; `None` otherwise.
        backward_interception_supported: Whether the active strategy is known to route
            backward through `LightningModule.backward()`, making
            `break_on_backward_exception` effective.
        backward_interception_reason: Why interception is unsupported, when
            `backward_interception_supported` is False.
    """

    generation: int
    updated_at: float

    run_id: str
    status: str
    stage: str | None
    phase: str

    epoch: int
    global_step: int
    batch_idx: int | None

    world_size: int
    ranks: tuple[RankInfo, ...]

    metrics: Mapping[str, ScalarMetric]
    learning_rates: tuple[float, ...]

    pending_commands: int
    hold_state: str

    rest_host: str | None
    rest_port: int | None

    torch_debug_url: str | None
    artifacts_path: str | None

    lightning_backend: str
    lightning_version: str

    exception: ExceptionBreakpointInfo | None
    backward_interception_supported: bool
    backward_interception_reason: str | None


def initial_snapshot(
    run_id: str, lightning_backend: str, lightning_version: str
) -> RuntimeSnapshot:
    """Builds the starting snapshot published before any Lightning hook has run."""
    return RuntimeSnapshot(
        generation=0,
        updated_at=time.time(),
        run_id=run_id,
        status="initializing",
        stage=None,
        phase="idle",
        epoch=0,
        global_step=0,
        batch_idx=None,
        world_size=1,
        ranks=(),
        metrics={},
        learning_rates=(),
        pending_commands=0,
        hold_state="none",
        rest_host=None,
        rest_port=None,
        torch_debug_url=None,
        artifacts_path=None,
        lightning_backend=lightning_backend,
        lightning_version=lightning_version,
        exception=None,
        backward_interception_supported=True,
        backward_interception_reason=None,
    )


class StateStore:
    """Publishes and reads `RuntimeSnapshot` instances under a small lock.

    Attributes:
        None: State is private; use `read`/`publish`/`update`.
    """

    def __init__(self, snapshot: RuntimeSnapshot) -> None:
        self._lock = threading.Lock()
        self._snapshot = snapshot

    def read(self) -> RuntimeSnapshot:
        """Returns the current snapshot."""
        with self._lock:
            return self._snapshot

    def publish(self, snapshot: RuntimeSnapshot) -> None:
        """Replaces the current snapshot outright."""
        with self._lock:
            self._snapshot = snapshot

    def update(self, **changes: object) -> RuntimeSnapshot:
        """Publishes a new snapshot derived from the current one.

        Args:
            **changes: Field overrides applied via `dataclasses.replace`.

        Returns:
            The newly published snapshot.
        """
        with self._lock:
            current = self._snapshot
            new = replace(
                current,
                generation=current.generation + 1,
                updated_at=time.time(),
                **changes,
            )
            self._snapshot = new
            return new
