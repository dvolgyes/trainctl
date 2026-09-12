"""Owns Trainctl's runtime services and dispatches Lightning-hook-driven state
publication and command execution.

The Lightning training thread is the only thread that mutates or deeply inspects
Trainer/model state. REST and FUSE only read published `RuntimeSnapshot`s or enqueue
`Command`s; this module is what actually executes them, from inside Lightning hooks.
"""

import json
import math
import os
import secrets
import shutil
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from loguru import logger

from trainctl.config import TrainctlConfig
from trainctl.hooks.manager import bootstrap_hooks_tree
from trainctl.hooks.session import HandleResult, HookSession
from trainctl.lightning_backend import LightningBackend
from trainctl.logging import LoggingSession
from trainctl.runtime import (
    cuda_memory,
    forward_input,
    graph_inspectors,
    hparam_registry,
    torchinfo_inspector,
    torchlens_inspector,
)
from trainctl.runtime.artifacts import ArtifactStore
from trainctl.runtime.commands import Command, CommandQueue, next_id
from trainctl.runtime.current_step import CurrentStep
from trainctl.runtime.events import SafePoint
from trainctl.runtime.finite_guard import DataloaderFiniteGuard, FiniteCheckReport
from trainctl.runtime.gate import RunGate
from trainctl.runtime.live_capture import LiveCaptureBroker
from trainctl.runtime.optimizer_surgery import find_adapter, resolve_parameters
from trainctl.runtime.pipeline_pressure import PipelinePressureTracker
from trainctl.runtime.profiler import ProfilerRun
from trainctl.runtime.state import (
    ExceptionBreakpointInfo,
    RankInfo,
    ScalarMetric,
    StateStore,
    initial_snapshot,
)
from trainctl.torch_debug.manager import TorchDebugManager

_ALLOWED_DURING_HOLD = frozenset(
    {
        "checkpoint",
        "snapshot",
        "set_learning_rate",
        "reset_momentum",
        "set_dataloader_finite_check",
        "set_hparam",
        "stop",
        "torchinfo_capture",
        "profile",
        "torchview_capture",
        "torchviz_capture",
        "torchlens_capture",
        "set_lightning_profiler",
    }
)
_ALLOWED_DURING_EXCEPTION = frozenset(
    {
        "save_current_batch",
        "save_current_loss",
        "save_partial_gradients",
        "save_optimizer_summary",
        "release_exception",
        "torchviz_capture",
    }
)
_KNOWN_SAFE_BACKWARD_STRATEGIES = frozenset(
    {"SingleDeviceStrategy", "DDPStrategy", "FSDPStrategy", "DDPSpawnStrategy"}
)


class TrainctlRuntime:
    """Coordinates state publication, command execution, and Trainctl's services.

    Attributes:
        config: Effective `TrainctlConfig` for this run.
        backend: The Lightning backend the composed model uses.
        run_id: Unique id for this training run.
        started_at: Unix timestamp this runtime object was constructed.
        state: Published `RuntimeSnapshot` store.
        command_queue: Submitted control-plane commands.
        gate: Cooperative hold gate.
        artifacts: Checkpoint/snapshot/debug-capture disk store; created once
            `setup_environment` resolves its final directory, not before.
        torch_debug: PyTorch distributed debug server manager.
        pipeline: Rolling per-batch phase-timing tracker.
        current_step: Live batch/loss/exception for the in-flight training step.
        finite_guard: Opt-in dataloader-batch finite-value checker.
        cuda_memory_history: Current CUDA allocator-history recording configuration.
        profiler_run: The armed/active bounded `torch.profiler` run, if any.
        lightning_profiler: The Lightning profiler instance currently installed as
            `trainer.profiler` by `set_lightning_profiler`, if any. Kept as a direct
            reference so its live `.summary()` can be read without a `trainer` handle.
        lightning_profiler_level: `off`, `simple`, `advanced`, or `pytorch` -- the
            level `lightning_profiler` was last set to.
        live_captures: Broker for cheap, non-persisted live views served at
            `SafePoint.BEFORE_OPTIMIZER_STEP` -- see `trainctl.fuse.filesystem`'s
            `_LIVE_FILES`.
        tunable_hparams: Current value/type of each hparam the model opted into via
            `trainctl_tunable_hparams`, refreshed at every safe point. Entries
            declared via the dict form also carry `min`/`max`/`label`.
    """

    def __init__(self, config: TrainctlConfig, backend: LightningBackend) -> None:
        self.config = config
        self.backend = backend
        self.run_id = next_id("run")
        self.started_at = time.time()
        self.state = StateStore(
            initial_snapshot(self.run_id, backend.name, backend.version)
        )
        self.command_queue = CommandQueue()
        self.gate = RunGate()
        self.artifacts: ArtifactStore | None = None
        self.torch_debug = TorchDebugManager(
            port=config.torch_debug_port, port_search=config.torch_debug_port_search
        )

        self._lock = threading.Lock()
        self._started = False
        self._is_rank_zero = True

        self._fuse_ops: Any = None
        self._fuse_thread: threading.Thread | None = None
        self._fuse_mountpoint: Path | None = None

        self._rest_server: Any = None
        self._rest_thread: threading.Thread | None = None
        self._rest_port: int | None = None

        self._model_class_name: str | None = None
        self._model_hparams: dict[str, Any] = {}
        self._model_summary: str = ""

        self._local_rank_info: RankInfo | None = None

        self.pipeline = PipelinePressureTracker(window=config.pipeline_pressure_window)
        self.current_step = CurrentStep()
        self.finite_guard = DataloaderFiniteGuard(
            enabled=config.dataloader_finite_check_enabled
        )
        self._exception_hold: ExceptionBreakpointInfo | None = None
        self._model_summary_json: dict[str, Any] = {}
        self.cuda_memory_history = cuda_memory.CudaMemoryHistoryState(
            enabled=False, max_entries=None, started_at=None
        )
        self.profiler_run: ProfilerRun | None = None
        self.lightning_profiler: Any | None = None
        self.lightning_profiler_level: str = "off"
        self.live_captures = LiveCaptureBroker()
        self.tunable_hparams: dict[str, hparam_registry.HparamValue] = {}
        self._tunable_hparam_specs: dict[str, hparam_registry.TunableHparamSpec] = {}

        self._hook_session: HookSession | None = None
        self._logging_session: LoggingSession | None = None

    @property
    def rest_url(self) -> str | None:
        """The reachable REST base URL, once started."""
        if self._rest_port is None:
            return None
        return f"http://{self.config.rest_host}:{self._rest_port}"

    @property
    def fuse_mountpoint(self) -> Path | None:
        """The active FUSE mountpoint, once started."""
        return self._fuse_mountpoint

    # ---- Lightning lifecycle ------------------------------------------------

    def setup_environment(self, pl_module: Any, stage: str) -> None:
        """Starts Trainctl's services. Idempotent; safe to call for every Lightning stage."""
        del stage
        with self._lock:
            if self._started:
                return
            self._started = True

        self._is_rank_zero = _is_rank_zero()
        self._gather_ranks(pl_module)
        self._detect_backward_interception(pl_module)

        self._guarded(
            "artifact directory", lambda: self._resolve_artifact_dir(pl_module)
        )

        if (
            self._is_rank_zero
            and self.config.torch_debug_enabled
            and self.config.rest_enabled
        ):
            self._guarded(
                "REST server + pytorch distributed debug server (paired ports)",
                self._start_rest_and_torch_debug,
            )
        else:
            if self.config.torch_debug_enabled:
                self._guarded(
                    "pytorch distributed debug server", self._start_torch_debug
                )
            if self._is_rank_zero and self.config.rest_enabled:
                self._guarded("REST server", self._start_rest)
        if self._is_rank_zero and self.config.fuse_enabled:
            self._guarded("FUSE mount", lambda: self._start_fuse(pl_module))

        self.state.update(status="running")

    def on_fit_start(self, pl_module: Any) -> None:
        """Publishes the first full-fidelity snapshot once the Trainer is attached."""
        self._model_class_name = type(pl_module).__name__
        self._model_hparams = dict(getattr(pl_module, "hparams", {}))
        self._model_summary = torchinfo_inspector.basic_summary_text(pl_module)
        self._model_summary_json = torchinfo_inspector.basic_summary_json(pl_module)
        self._tunable_hparam_specs = hparam_registry.collect_tunable_hparams(
            type(pl_module)
        )
        self._publish(pl_module, status="held" if self.gate.is_held() else "running")

    def teardown(self, pl_module: Any, stage: str) -> None:
        """Stops Trainctl's services. Idempotent; releases any dangling hold first."""
        del pl_module, stage
        with self._lock:
            if not self._started:
                return
            self._started = False
        self.state.update(status="stopping")
        self.gate.release()
        self._guarded("REST shutdown", self._stop_rest)
        self._guarded("FUSE shutdown", self._stop_fuse)
        self._guarded("pytorch distributed debug shutdown", self.torch_debug.stop)
        self.state.update(status="finished")

    # ---- lifecycle hooks --------------------------------------------------------

    def ensure_hook_session(self, trainer: Any, pl_module: Any) -> None:
        """Bootstraps this run's hooks tree and logging bridge for one stage `_run`.

        Idempotent -- a no-op if a session from this same stage is already open.
        Called only from the hooks adapter's own `setup(trainer, pl_module, stage)`,
        which Lightning always calls before any other callback method for that
        `_run` and always pairs with a matching `teardown` (barring an uncaught
        exception, handled separately by `finalize_hooks_and_logging`). A hook that
        fires with no stage active at all (e.g. a standalone `Trainer.save_checkpoint`
        call outside `fit`/`validate`/`test`/`predict`) finds no session and is a
        silent no-op in `dispatch_hook` -- there is no run for it to belong to.
        Reads rank/world size directly from `trainer`: `_local_rank_info` is not
        populated yet at this point (the adapter's `setup` runs before the composed
        model's own `setup`, which is what triggers `_gather_ranks`).
        """
        if self._hook_session is not None:
            return
        base_dir = _default_base_dir(pl_module, self.run_id)
        session_id = _new_hooks_session_id()
        logging_session = LoggingSession(session_id)
        if self.config.intercept_lightning_logging:
            logging_session.install()
        if self.config.run_log_enabled:
            rank = trainer.global_rank
            filename = (
                f"trainctl-rank-{rank}.log"
                if self.config.hooks_rank_policy == "all"
                else "trainctl.log"
            )
            logging_session.attach_run_file(base_dir / filename)
        log = logging_session.bind()
        shell = self.config.hooks_shell
        if shell is None:
            resolved = shutil.which("bash")
            if resolved is None:
                raise RuntimeError(
                    "trainctl: hooks_enabled requires bash on PATH or an explicit hooks_shell"
                )
            shell = Path(resolved)
        live_dir = bootstrap_hooks_tree(
            base_dir, self.config.hooks_source_dir, session_id=session_id, log=log
        )
        self._logging_session = logging_session
        self._hook_session = HookSession(
            live_dir,
            session_id=session_id,
            rank=trainer.global_rank,
            world_size=trainer.world_size,
            rank_policy=self.config.hooks_rank_policy,
            shell=shell,
            timeout_s=self.config.hooks_timeout_s,
            max_params_bytes=self.config.hooks_max_params_bytes,
            max_metadata_items=self.config.hooks_max_metadata_items,
            max_metadata_depth=self.config.hooks_max_metadata_depth,
            max_export_bytes=self.config.hooks_max_export_bytes,
            max_output_bytes=self.config.hooks_max_output_bytes,
            log=log,
        )

    def dispatch_hook(
        self,
        trainer: Any,
        hook: str,
        *,
        stage_override: str | None = None,
        arguments: dict[str, Any] | None = None,
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
    ) -> HandleResult | None:
        """Dispatches one Lightning callback occurrence through the live `HookSession`.

        A no-op if `ensure_hook_session` never installed a session (hooks disabled).
        """
        if self._hook_session is None:
            return None
        stage = stage_override if stage_override is not None else _stage_string(trainer)
        return self._hook_session.handle(
            hook,
            stage=stage,
            epoch=trainer.current_epoch,
            global_step=trainer.global_step,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            arguments=arguments,
        )

    def finalize_hooks_and_logging(self) -> None:
        """Closes the hooks session and logging bridge, if either is currently open.

        Idempotent. Called from the hooks adapter's own `teardown` on the normal
        path, and from its `on_exception` on the failure path, since Lightning skips
        the callback's own `teardown` after an uncaught exception.
        """
        if self._hook_session is not None:
            self._hook_session.close()
            self._hook_session = None
        if self._logging_session is not None:
            self._logging_session.close()
            self._logging_session = None

    # ---- safe points ----------------------------------------------------------

    def process_safe_point(
        self, pl_module: Any, safe_point: SafePoint, batch_idx: int | None = None
    ) -> None:
        """Publishes state, executes due commands, and blocks here if a hold is active.

        Called only from the Lightning training thread, at one of the safe points
        listed in `trainctl.runtime.events.SafePoint`.
        """
        trainer = pl_module.trainer
        self._run_due_commands(pl_module, trainer, safe_point, allowed=None)
        if safe_point is SafePoint.BEFORE_OPTIMIZER_STEP:
            self.live_captures.serve(pl_module)
        self._publish(
            pl_module,
            status="held" if self.gate.is_held() else "running",
            batch_idx=batch_idx,
        )
        self._refresh_tunable_hparams(pl_module, trainer)
        if self.gate.is_held():
            self._hold_loop(pl_module, trainer)

    def _refresh_tunable_hparams(self, pl_module: Any, trainer: Any) -> None:
        """Rebuilds `tunable_hparams` from each registered knob's current value.

        Reassigns the whole dict rather than mutating it in place, matching
        `cuda_memory_history`/`profiler_run`'s thread-safety idiom. A single
        unreadable knob is logged and skipped rather than blanking the rest.
        """
        fresh: dict[str, hparam_registry.HparamValue] = {}
        for name, spec in self._tunable_hparam_specs.items():
            try:
                parent, leaf = hparam_registry.resolve(pl_module, trainer, spec.path)
                described = hparam_registry.describe(
                    hparam_registry.get_field(parent, leaf)
                )
                fresh[name] = replace(
                    described, min=spec.min, max=spec.max, label=spec.label
                )
            except Exception as exc:  # noqa: BLE001 -- one broken knob must not blank the dict
                logger.warning("trainctl tunable hparam {!r} unreadable: {}", name, exc)
        self.tunable_hparams = fresh

    def _hold_loop(self, pl_module: Any, trainer: Any) -> None:
        while self.gate.is_held():
            self._run_due_commands(
                pl_module, trainer, safe_point=None, allowed=_ALLOWED_DURING_HOLD
            )
            self._publish(pl_module, status="held")
            if not self.gate.is_held():
                break
            self.gate.wait_for_control_event(timeout=0.1)
        self._publish(pl_module, status="running")

    def _run_due_commands(
        self,
        pl_module: Any,
        trainer: Any,
        safe_point: SafePoint | None,
        allowed: frozenset[str] | None,
    ) -> None:
        epoch = trainer.current_epoch
        step = trainer.global_step
        for command in self.command_queue.due(safe_point, epoch, step):
            if allowed is not None and command.kind not in allowed:
                continue
            self._execute(pl_module, trainer, command, safe_point)

    def _execute(
        self,
        pl_module: Any,
        trainer: Any,
        command: Command,
        safe_point: SafePoint | None,
    ) -> None:
        self.command_queue.mark_executing(command)
        handler = _HANDLERS.get(command.kind)
        if handler is None:
            self.command_queue.mark_result(
                command, succeeded=False, error=f"unknown command kind {command.kind!r}"
            )
            return
        try:
            result = handler(self, pl_module, trainer, command, safe_point)
        except Exception as exc:  # noqa: BLE001 -- command boundary: untrusted REST input must not crash training
            logger.warning(
                "trainctl command {} ({}) failed: {}", command.id, command.kind, exc
            )
            self.command_queue.mark_result(command, succeeded=False, error=str(exc))
            return
        self.command_queue.mark_result(command, succeeded=True, result=result)

    # ---- per-batch hooks ----------------------------------------------------------

    def begin_train_batch(self, pl_module: Any, batch: Any, batch_idx: int) -> None:
        """Records batch-start timing/state. Called from `on_train_batch_start`.

        Enters a cooperative hold directly (bypassing the command queue, like
        `_handle_hold` does from inside a command) if the dataloader finite guard is
        enabled and finds a non-finite tensor in `batch`.
        """
        trainer = pl_module.trainer
        now = time.perf_counter_ns()
        self.pipeline.batch_start(
            now, trainer.current_epoch, trainer.global_step, batch_idx
        )
        self.current_step.begin(
            batch, batch_idx, trainer.current_epoch, trainer.global_step, now
        )
        if self.finite_guard.enabled:
            report = self.finite_guard.check(batch)
            if report is not None:
                self._handle_bad_batch(pl_module, trainer, batch_idx, report)

    def before_backward(self, loss: Any) -> None:
        """Records the loss and pre-backward timestamp. Called from `on_before_backward`."""
        self.current_step.loss = loss
        self.pipeline.before_backward(time.perf_counter_ns())

    def after_backward(self) -> None:
        """Records the post-backward timestamp. Called from `on_after_backward`."""
        self.pipeline.after_backward(time.perf_counter_ns())

    def before_optimizer_step(self) -> None:
        """Records the pre-optimizer-step timestamp. Called from `on_before_optimizer_step`."""
        self.pipeline.before_optimizer(time.perf_counter_ns())

    def finish_train_batch(self, pl_module: Any) -> None:
        """Finalizes this batch's pipeline-pressure sample and clears `current_step`.

        Called from `on_train_batch_end`, after safe-point processing, in a `finally`
        block so the retained batch/loss/exception never survive into the next batch.
        """
        self.pipeline.batch_end(time.perf_counter_ns())
        self._advance_profiler(pl_module)
        self.current_step.clear()

    def _advance_profiler(self, pl_module: Any) -> None:
        """Steps the active `ProfilerRun`, if any, finalizing it once due.

        Runs inside the hot per-batch hook path, not a command handler, so this is a
        dedicated always-swallowing boundary rather than `_guarded` -- `_guarded`
        re-raises under `inspection_strict`, which would let an inspector failure
        crash live training.
        """
        run = self.profiler_run
        if run is None:
            return
        try:
            done = run.step()
        except Exception as exc:  # noqa: BLE001 -- inspector must never destabilize training
            logger.warning("trainctl: profiler step failed: {}", exc)
            self.profiler_run = None
            return
        if not done:
            return
        then_hold = run.then_hold
        try:
            files = run.finalize()
            self.artifacts.write_debug_capture(
                "profiler", files, metadata={"level": run.level}
            )
        except Exception as exc:  # noqa: BLE001 -- inspector must never destabilize training
            logger.warning("trainctl: profiler finalize failed: {}", exc)
        self.profiler_run = None
        if then_hold:
            trainer = pl_module.trainer
            self.gate.enter(
                next_id("profiler"),
                "train_batch_end",
                trainer.current_epoch,
                trainer.global_step,
            )
            self._publish(pl_module, status="held")
            self._hold_loop(pl_module, trainer)

    def _handle_bad_batch(
        self, pl_module: Any, trainer: Any, batch_idx: int, report: FiniteCheckReport
    ) -> None:
        rank = self._local_rank_info.global_rank if self._local_rank_info else 0
        self.artifacts.write_snapshot(
            "anomaly_input_nonfinite",
            {
                "reason": "nonfinite_dataloader_tensor",
                "rank": rank,
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "batch_idx": batch_idx,
                "tensors": [asdict(bad) for bad in report.bad_tensors],
            },
        )
        self.gate.enter(
            next_id("anomaly"),
            "train_batch_start",
            trainer.current_epoch,
            trainer.global_step,
        )
        self._publish(pl_module, status="held", batch_idx=batch_idx)
        self._hold_loop(pl_module, trainer)

    # ---- exception breakpoint ------------------------------------------------------

    def handle_backward_exception(
        self, pl_module: Any, loss: Any, exc: Exception
    ) -> None:
        """Freezes the failing batch/loss and holds for inspection before re-raising.

        Called from `TrainctlMixin.backward`'s `except` clause; no-ops if
        `break_on_backward_exception` is disabled. Diagnostic only: gradient state at
        this point is not a valid complete gradient set, and distributed collectives
        may already be inconsistent across ranks -- see `ExceptionBreakpointInfo`.
        """
        if not self.config.break_on_backward_exception:
            return
        trainer = pl_module.trainer
        self.current_step.loss = loss
        self.current_step.exception = exc
        rank = self._local_rank_info.global_rank if self._local_rank_info else 0
        self._exception_hold = ExceptionBreakpointInfo(
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            epoch=trainer.current_epoch,
            global_step=trainer.global_step,
            batch_idx=self.current_step.batch_idx,
            rank=rank,
            started_at=time.time(),
        )
        self._publish(pl_module, status="exception_held")
        self._exception_hold_loop(pl_module, trainer)
        self._publish(pl_module, status="running")

    def _exception_hold_loop(self, pl_module: Any, trainer: Any) -> None:
        while self._exception_hold is not None:
            self._run_due_commands(
                pl_module, trainer, safe_point=None, allowed=_ALLOWED_DURING_EXCEPTION
            )
            self._publish(pl_module, status="exception_held")
            if self._exception_hold is None:
                break
            self.gate.wait_for_control_event(timeout=0.1)

    def _detect_backward_interception(self, pl_module: Any) -> None:
        """Probes whether the active strategy is known to route backward through
        `LightningModule.backward()`, via an explicit allowlist (not runtime
        bypass-detection, which is unreliable).
        """
        trainer = getattr(pl_module, "trainer", None)
        strategy = getattr(trainer, "strategy", None) if trainer is not None else None
        if strategy is None:
            return
        strategy_name = type(strategy).__name__
        if strategy_name in _KNOWN_SAFE_BACKWARD_STRATEGIES:
            return
        reason = (
            f"strategy {strategy_name!r} is not in the known-safe allowlist for "
            "LightningModule.backward() interception"
        )
        self.state.update(
            backward_interception_supported=False, backward_interception_reason=reason
        )
        if self.config.break_on_backward_exception:
            logger.warning(
                "trainctl: break_on_backward_exception requested but {}", reason
            )

    # ---- state publication ------------------------------------------------------

    def _publish(
        self, pl_module: Any, status: str, batch_idx: int | None = None
    ) -> None:
        trainer = pl_module.trainer
        metrics: dict[str, ScalarMetric] = {}
        for key, value in trainer.callback_metrics.items():
            try:
                metrics[key] = ScalarMetric(
                    value=float(value),
                    epoch=trainer.current_epoch,
                    step=trainer.global_step,
                    updated_at=time.time(),
                )
            except (TypeError, ValueError):
                continue
        learning_rates = tuple(
            group["lr"]
            for optimizer in trainer.optimizers
            for group in optimizer.param_groups
        )
        stage = trainer.state.stage.value if trainer.state.stage is not None else None
        pending = sum(
            1 for command in self.command_queue.list_all() if command.status == "queued"
        )
        self.state.update(
            status=status,
            stage=stage,
            phase=_phase_of(stage),
            epoch=trainer.current_epoch,
            global_step=trainer.global_step,
            batch_idx=batch_idx,
            ranks=(self._current_rank_info(),),
            metrics=metrics,
            learning_rates=learning_rates,
            pending_commands=pending,
            hold_state="held" if self.gate.is_held() else "none",
            exception=self._exception_hold,
        )

    def _current_rank_info(self) -> RankInfo:
        """Refreshes `worker_pids` on the rank info gathered once at setup.

        DataLoader workers are spawned lazily and non-persistent ones churn per
        epoch, so this field alone is recomputed on every publish rather than
        gathered once like the rest of `RankInfo`.
        """
        import multiprocessing

        assert self._local_rank_info is not None, (
            "_gather_ranks must run before the first _publish"
        )
        worker_pids = tuple(
            sorted(
                child.pid
                for child in multiprocessing.active_children()
                if child.pid is not None
            )
        )
        return replace(self._local_rank_info, worker_pids=worker_pids)

    def _gather_ranks(self, pl_module: Any) -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            global_rank = dist.get_rank()
        else:
            world_size = 1
            global_rank = 0
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = str(getattr(pl_module, "device", "cpu"))
        rank_info = RankInfo(
            global_rank=global_rank,
            local_rank=local_rank,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            device=device,
        )
        self._local_rank_info = rank_info
        self.state.update(world_size=world_size, ranks=(rank_info,))

    # ---- service management ------------------------------------------------------

    def _guarded(self, description: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:
            if self.config.inspection_strict:
                raise
            logger.warning("trainctl: {} failed: {}", description, exc)

    def _start_torch_debug(self) -> None:
        self.torch_debug.start_after_distributed_init()
        self.torch_debug.wait_ready(timeout=10)
        self.state.update(torch_debug_url=self.torch_debug.frontend_url)

    def _start_rest_and_torch_debug(self) -> None:
        """Starts REST and the torch_debug frontend as one paired unit: both ports are
        offset from their configured base port by the same multiple of `_PORT_OFFSET_STEP`
        (e.g. offset 200 pairs REST 8290 with torch_debug 26199) -- a human-legible
        pairing, and a way to keep the two services from independently drifting to
        unrelated ports when several runs share a host and start at nearly the same time.

        Per offset attempt: probe the torch_debug candidate (it can't be held -- see
        `TorchDebugManager.start_after_distributed_init`), then bind-and-hold the REST
        candidate at the same offset (race-free once held), then re-probe the
        torch_debug candidate immediately before starting it, narrowing (not
        eliminating -- an upstream constraint) its check-then-bind race to that final
        gap. A collision at either check abandons the offset and retries the next one.
        """
        max_attempts = self.config.paired_port_search_attempts
        for attempt in range(max_attempts):
            offset = attempt * _PORT_OFFSET_STEP
            debug_candidate = self.config.torch_debug_port + offset
            if not self.torch_debug.port_is_free(debug_candidate):
                continue
            try:
                sock = _bind_socket(
                    self.config.rest_host, self.config.rest_port + offset
                )
            except OSError:
                continue
            if not self.torch_debug.port_is_free(debug_candidate):
                sock.close()
                continue
            self.torch_debug.start_after_distributed_init(port=debug_candidate)
            self.torch_debug.wait_ready(timeout=10)
            self.state.update(torch_debug_url=self.torch_debug.frontend_url)
            self._start_rest_on_socket(sock, self.config.rest_port + offset)
            return
        raise RuntimeError(
            f"no free paired REST/torch_debug port offset found in {max_attempts} "
            f"attempts (step {_PORT_OFFSET_STEP})"
        )

    def _resolve_artifact_dir(self, pl_module: Any) -> None:
        artifact_dir = (
            self.config.artifact_dir
            or _default_base_dir(pl_module, self.run_id) / "artifacts"
        )
        self.artifacts = ArtifactStore(artifact_dir)
        self.state.update(artifacts_path=str(artifact_dir))

    def _start_fuse(self, pl_module: Any) -> None:
        import mfusepy

        from trainctl.fuse.filesystem import TrainctlFS

        mountpoint = (
            self.config.fuse_mountpoint
            or _default_base_dir(pl_module, self.run_id) / "procfs"
        )
        mountpoint.mkdir(parents=True, exist_ok=True)
        ops = TrainctlFS(self)

        def _run() -> None:
            mfusepy.FUSE(ops, str(mountpoint), foreground=True, direct_io=True)

        thread = threading.Thread(target=_run, name="trainctl-fuse", daemon=True)
        thread.start()
        if not ops.wait_ready(timeout=10):
            raise RuntimeError("FUSE mount did not become ready in time")
        self._fuse_ops = ops
        self._fuse_thread = thread
        self._fuse_mountpoint = mountpoint

    def _stop_fuse(self) -> None:
        if self._fuse_thread is None:
            return
        import subprocess

        subprocess.run(["fusermount3", "-u", str(self._fuse_mountpoint)], check=False)  # noqa: S603, S607 -- own mount
        self._fuse_thread.join(timeout=5)
        self._fuse_ops = None
        self._fuse_thread = None
        self._fuse_mountpoint = None

    def _start_rest(self) -> None:
        sock, port = _bind_free_socket(
            self.config.rest_host, self.config.rest_port, self.config.rest_port_search
        )
        self._start_rest_on_socket(sock, port)

    def _start_rest_on_socket(self, sock: socket.socket, port: int) -> None:
        import uvicorn

        from trainctl.rest.app import build_app

        app = build_app(self)
        uv_config = uvicorn.Config(
            app, host=self.config.rest_host, port=port, log_level="warning"
        )
        server = uvicorn.Server(uv_config)
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [sock]},
            name="trainctl-rest",
            daemon=True,
        )
        thread.start()
        if not _wait_own_rest_ready(
            f"http://{self.config.rest_host}:{port}/health", self.run_id, timeout=10
        ):
            raise RuntimeError("REST server did not become ready in time")
        self._rest_server = server
        self._rest_thread = thread
        self._rest_port = port
        self.state.update(rest_host=self.config.rest_host, rest_port=port)

    def _stop_rest(self) -> None:
        if self._rest_server is None:
            return
        self._rest_server.should_exit = True
        if self._rest_thread is not None:
            self._rest_thread.join(timeout=5)
        self._rest_server = None
        self._rest_thread = None
        self._rest_port = None


def _is_rank_zero() -> bool:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _phase_of(stage: str | None) -> str:
    if stage is None:
        return "idle"
    if stage in ("sanity_check", "validate"):
        return "validate"
    if stage in ("train", "test", "predict"):
        return stage
    return "idle"


def _default_base_dir(pl_module: Any, run_id: str) -> Path:
    """Resolves the default directory for Trainctl's FUSE mount and artifact store.

    Colocates under the Lightning run's own `trainer.log_dir` (alongside checkpoints
    and logger output) whenever a Trainer is attached and reports one; falls back to a
    per-uid runtime/tmp directory, scoped by run id, when no Trainer is attached yet or
    no log directory is available.
    """
    trainer = getattr(pl_module, "trainer", None)
    log_dir = getattr(trainer, "log_dir", None) if trainer is not None else None
    if log_dir:
        return Path(log_dir)
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(xdg) / "trainctl" if xdg else Path(f"/tmp/trainctl-{os.getuid()}")  # noqa: S108 -- per-uid tmp root
    return root / run_id


def _new_hooks_session_id() -> str:
    """A collision-resistant session id: `commands.next_id` is process-local only."""
    return f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(4)}"


def _stage_string(trainer: Any) -> str | None:
    """`trainer.state.fn`'s value (`fit`/`validate`/`test`/`predict`), or `None`."""
    fn = trainer.state.fn
    return fn.value if fn is not None else None


# Shared offset step for `TrainctlRuntime._start_rest_and_torch_debug`'s paired port
# selection: REST and torch_debug candidates at attempt N are both `N * _PORT_OFFSET_STEP`
# above their configured base port (e.g. offset 200 pairs REST 8290 with debug 26199).
_PORT_OFFSET_STEP = 100


def _bind_socket(host: str, port: int) -> socket.socket:
    """Binds and holds a listening socket on exactly `port`. Raises `OSError` if taken.

    Unlike a probe-then-release check, the socket is kept open and handed straight to
    uvicorn (`Server.run(sockets=[...])`), so there is no gap between "found free" and
    "actually listening" for a second concurrent caller to race into.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        sock.close()
        raise
    sock.listen(2048)  # matches uvicorn.Config's own default backlog
    return sock


def _bind_free_socket(
    host: str, start: int, search_range: int
) -> tuple[socket.socket, int]:
    """Binds and holds a listening socket on the first free candidate port in
    `[start, start + search_range)`, tried sequentially. See `_bind_socket`.
    """
    for candidate in range(start, start + search_range):
        try:
            return _bind_socket(host, candidate), candidate
        except OSError:
            continue
    raise RuntimeError(f"no free REST port in [{start}, {start + search_range})")


def _wait_own_rest_ready(url: str, run_id: str, timeout: float) -> bool:
    """Polls `url` until it reports the expected `run_id`, or `timeout` elapses.

    Checking identity (not just reachability) guards against a narrow residual
    race: some other server -- another Trainctl run, most plausibly -- already
    answering on this exact host:port would otherwise look indistinguishable
    from our own server being ready.
    """
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(  # noqa: S310 -- own localhost server
                url, timeout=1
            ) as response:
                payload = json.loads(response.read())
        except Exception:  # noqa: BLE001 -- readiness probe boundary, not business logic
            time.sleep(0.2)
            continue
        if payload.get("run_id") == run_id:
            return True
        time.sleep(0.2)
    return False


# ---- command handlers -----------------------------------------------------------------


def _optimizer_summary_payload(trainer: Any) -> dict[str, Any]:
    return {
        str(index): {
            "type": type(optimizer).__name__,
            "param_groups": [{"lr": group["lr"]} for group in optimizer.param_groups],
        }
        for index, optimizer in enumerate(trainer.optimizers)
    }


def _handle_hold(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del pl_module
    runtime.gate.enter(
        command.id, str(safe_point), trainer.current_epoch, trainer.global_step
    )
    return {"hold": "entered", "reason": command.args.get("reason", "")}


def _handle_stop(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del runtime, pl_module, command, safe_point
    trainer.should_stop = True
    return {"stop": "requested"}


def _handle_checkpoint(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del pl_module, safe_point
    label = command.args.get("label", "manual")
    path = runtime.artifacts.checkpoint_path(label)
    trainer.save_checkpoint(str(path))
    return {"checkpoint_path": str(path)}


def _handle_snapshot(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del safe_point
    kind = command.args.get("kind", "runtime")
    if kind == "runtime":
        payload = asdict(runtime.state.read())
    elif kind == "model_summary":
        payload = {
            "class": type(pl_module).__name__,
            "hparams": dict(getattr(pl_module, "hparams", {})),
        }
    elif kind == "optimizer_summary":
        payload = _optimizer_summary_payload(trainer)
    elif kind == "scheduler_summary":
        configs = getattr(trainer, "lr_scheduler_configs", [])
        payload = {
            str(index): {"type": type(getattr(config, "scheduler", config)).__name__}
            for index, config in enumerate(configs)
        }
    else:
        raise ValueError(f"unknown snapshot kind {kind!r}")
    return runtime.artifacts.write_snapshot(kind, payload)


def _handle_set_learning_rate(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del pl_module, safe_point
    value = float(command.args["value"])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"learning rate must be finite and >= 0, got {value}")
    optimizer_index = int(command.args.get("optimizer", 0))
    optimizer = trainer.optimizers[optimizer_index]
    param_group = command.args.get("param_group", "*")
    if param_group != "*" and not runtime.config.optimizer_surgery_enabled:
        raise ValueError(
            "per-group learning-rate selection requires optimizer_surgery_enabled"
        )
    if param_group == "*":
        groups = list(enumerate(optimizer.param_groups))
    else:
        index = int(param_group)
        if not 0 <= index < len(optimizer.param_groups):
            raise ValueError(f"optimizer {optimizer_index} has no param_group {index}")
        groups = [(index, optimizer.param_groups[index])]
    changes: dict[str, Any] = {}
    for index, group in groups:
        changes[str(index)] = {"before": group["lr"], "after": value}
        group["lr"] = value
    result: dict[str, Any] = {"optimizer": optimizer_index, "groups": changes}
    if getattr(trainer, "lr_scheduler_configs", None):
        result["warning"] = (
            "optimizer has an active LR scheduler; this value may be overwritten "
            "on its next scheduler step"
        )
    return result


def _handle_reset_momentum(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del safe_point
    if not runtime.config.optimizer_surgery_enabled:
        raise ValueError("reset_momentum requires optimizer_surgery_enabled")
    optimizer_index = int(command.args.get("optimizer", 0))
    optimizer = trainer.optimizers[optimizer_index]
    adapter = find_adapter(optimizer)
    if adapter is None:
        raise ValueError(
            "optimizer_surgery_unsupported: no reset_momentum adapter for "
            f"{type(optimizer).__module__}.{type(optimizer).__name__}"
        )
    parameters = resolve_parameters(
        pl_module, optimizer, command.args.get("selector", "all")
    )
    result = adapter.reset_momentum(optimizer, parameters)
    return {"optimizer": optimizer_index, **result}


def _handle_set_dataloader_finite_check(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del pl_module, trainer, safe_point
    enabled = bool(command.args["enabled"])
    runtime.finite_guard.enabled = enabled
    return {"enabled": enabled}


def _handle_set_hparam(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del safe_point
    name = command.args["name"]
    spec = runtime._tunable_hparam_specs.get(name)
    if spec is None:
        raise ValueError(f"unknown or non-tunable hparam: {name!r}")
    value = command.args["value"]
    hparam_registry.validate_bounds(spec, value)
    parent, leaf = hparam_registry.resolve(pl_module, trainer, spec.path)
    result = hparam_registry.cast_and_assign(parent, leaf, value)
    return {"name": name, **asdict(result)}


def _handle_save_current_batch(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del pl_module, trainer, command, safe_point
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(runtime.current_step.batch, buffer)
    return runtime.artifacts.write_debug_capture(
        "current-batch", {"batch.pt": buffer.getvalue()}
    )


def _handle_save_current_loss(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del pl_module, trainer, command, safe_point
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(runtime.current_step.loss, buffer)
    return runtime.artifacts.write_debug_capture(
        "current-loss", {"loss.pt": buffer.getvalue()}
    )


def _handle_save_partial_gradients(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del trainer, command, safe_point
    summary = {
        name: {
            "grad_is_none": parameter.grad is None,
            "grad_norm": (
                float(parameter.grad.norm()) if parameter.grad is not None else None
            ),
            "grad_shape": (
                list(parameter.grad.shape) if parameter.grad is not None else None
            ),
        }
        for name, parameter in pl_module.named_parameters()
    }
    return runtime.artifacts.write_debug_capture(
        "partial-gradients", {"gradients.json": json.dumps(summary, indent=2)}
    )


def _handle_save_optimizer_summary(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del pl_module, command, safe_point
    payload = _optimizer_summary_payload(trainer)
    return runtime.artifacts.write_debug_capture(
        "optimizer-summary", {"optimizer.json": json.dumps(payload, indent=2)}
    )


def _handle_release_exception(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del pl_module, trainer, safe_point
    runtime._exception_hold = None
    return {"released": command.id}


def _handle_torchinfo_capture(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del trainer, command, safe_point
    if runtime.current_step.batch is None:
        raise ValueError("torchinfo_capture requires an in-flight batch")
    target = forward_input.resolve(pl_module, runtime.current_step.batch)
    files = torchinfo_inspector.rich_summary(pl_module, target)
    return runtime.artifacts.write_debug_capture("torchinfo-summary", files)


def _handle_profile(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del pl_module, trainer, safe_point
    if runtime.profiler_run is not None:
        raise ValueError("a profiler run is already armed")
    warmup_steps = int(command.args.get("warmup_steps", 0))
    active_steps = int(command.args.get("active_steps", 1))
    level = command.args.get("level", "basic")
    then_hold = bool(command.args.get("then_hold", False))
    run = ProfilerRun(command.id, warmup_steps, active_steps, level, then_hold)
    run.start()
    runtime.profiler_run = run
    return {
        "armed": True,
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "level": level,
        "then_hold": then_hold,
    }


_LIGHTNING_PROFILER_LEVELS = frozenset({"off", "simple", "advanced", "pytorch"})


def _handle_set_lightning_profiler(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    """Installs (or removes) one of Lightning's own profilers as `trainer.profiler`.

    Unlike `profile` (a bounded `torch.profiler` capture over N batches, finalized
    into a debug-capture artifact), Lightning's profilers wrap Lightning's own named
    hooks (`training_step`, `backward`, `optimizer_step`, dataloader-next, callbacks,
    ...) and accumulate continuously until swapped out or read. `trainer.profiler` is
    a plain attribute Lightning re-reads on every hook call, so swapping it here takes
    effect immediately, without restarting the Trainer.

    `command.args["level"]` selects `off`, `simple`, `advanced`, or `pytorch`.
    Selecting a level other than the current one -- or reselecting the current one --
    always installs a fresh profiler instance, discarding any previously accumulated
    durations (an implicit reset).
    """
    del pl_module, safe_point
    level = command.args.get("level", "off")
    if level not in _LIGHTNING_PROFILER_LEVELS:
        raise ValueError(f"unknown lightning profiler level {level!r}")
    profilers = runtime.backend.PL.profilers
    if level == "off":
        trainer.profiler = profilers.PassThroughProfiler()
        runtime.lightning_profiler = None
    else:
        profiler_cls = {
            "simple": profilers.SimpleProfiler,
            "advanced": profilers.AdvancedProfiler,
            "pytorch": profilers.PyTorchProfiler,
        }[level]
        profiler = profiler_cls()
        trainer.profiler = profiler
        runtime.lightning_profiler = profiler
    runtime.lightning_profiler_level = level
    return {"level": level}


def _handle_torchview_capture(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del trainer, command, safe_point
    if runtime.current_step.batch is None:
        raise ValueError("torchview_capture requires an in-flight batch")
    target = forward_input.resolve(pl_module, runtime.current_step.batch)
    files = graph_inspectors.torchview_capture(pl_module, target)
    return runtime.artifacts.write_debug_capture("torchview-graph", files)


def _handle_torchviz_capture(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    del trainer, safe_point
    show_attrs = bool(command.args.get("show_attrs", True))
    show_saved = bool(command.args.get("show_saved", True))
    if runtime._exception_hold is not None:
        # Diagnosing an actual backward failure: visualize the loss exactly as
        # autograd left it when backward() raised, not an unrelated fresh graph.
        if runtime.current_step.loss is None:
            raise ValueError("torchviz_capture requires an in-flight loss")
        files = graph_inspectors.torchviz_capture(
            pl_module, runtime.current_step.loss, show_attrs, show_saved
        )
    else:
        # A normal live capture: current_step.loss was already backwarded by
        # Lightning itself by the time any command can run, so its graph is gone
        # ("Trying to backward through the graph a second time") -- run a fresh,
        # not-yet-backwarded forward pass instead.
        if runtime.current_step.batch is None:
            raise ValueError("torchviz_capture requires an in-flight batch")
        target = forward_input.resolve(pl_module, runtime.current_step.batch)
        files = graph_inspectors.torchviz_capture_live(
            pl_module, target, show_attrs, show_saved
        )
    return runtime.artifacts.write_debug_capture("torchviz-graph", files)


def _handle_torchlens_capture(
    runtime: TrainctlRuntime,
    pl_module: Any,
    trainer: Any,
    command: Command,
    safe_point: SafePoint | None,
) -> dict:
    if runtime.current_step.batch is None:
        raise ValueError("torchlens_capture requires an in-flight batch")
    if not runtime.gate.is_held():
        # torchlens instruments the model with hooks; auto-engage a hold so a human
        # can inspect the result (or a failed capture's aftermath) before choosing to
        # resume, rather than requiring a separate hold command submitted first.
        runtime.gate.enter(
            command.id, str(safe_point), trainer.current_epoch, trainer.global_step
        )
    target = forward_input.resolve(pl_module, runtime.current_step.batch)
    files = torchlens_inspector.capture(pl_module, target)
    return runtime.artifacts.write_debug_capture("torchlens-trace", files)


_HANDLERS: dict[
    str, Callable[[TrainctlRuntime, Any, Any, Command, SafePoint | None], dict]
] = {
    "hold": _handle_hold,
    "stop": _handle_stop,
    "checkpoint": _handle_checkpoint,
    "snapshot": _handle_snapshot,
    "set_learning_rate": _handle_set_learning_rate,
    "reset_momentum": _handle_reset_momentum,
    "set_dataloader_finite_check": _handle_set_dataloader_finite_check,
    "set_hparam": _handle_set_hparam,
    "save_current_batch": _handle_save_current_batch,
    "save_current_loss": _handle_save_current_loss,
    "save_partial_gradients": _handle_save_partial_gradients,
    "save_optimizer_summary": _handle_save_optimizer_summary,
    "release_exception": _handle_release_exception,
    "torchinfo_capture": _handle_torchinfo_capture,
    "profile": _handle_profile,
    "torchview_capture": _handle_torchview_capture,
    "torchviz_capture": _handle_torchviz_capture,
    "torchlens_capture": _handle_torchlens_capture,
    "set_lightning_profiler": _handle_set_lightning_profiler,
}
