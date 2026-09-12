"""Typed commands, their `when` scheduling spec, and the thread-safe command queue.

REST never mutates the Trainer/model directly (see the module docstring in
`trainctl.runtime.runtime`); it only enqueues a `Command` here. The training thread
evaluates which commands are due at each safe point it passes through and executes
their handlers itself.
"""

import itertools
import threading
import time
from dataclasses import dataclass
from typing import Any

from trainctl.runtime.events import SafePoint

_id_counter = itertools.count(1)


def next_id(prefix: str) -> str:
    """Generates a short, monotonically increasing, process-local id."""
    return f"{prefix}_{next(_id_counter):06d}"


@dataclass(frozen=True)
class CommandWhen:
    """Normalized scheduling condition for a command.

    Attributes:
        safe_point: The safe point this condition is scored against, or None for
            `occurrence in ("immediate", "next_any")`, which match any safe point.
        occurrence: `immediate` (due as soon as any safe point, including inside an
            active hold, is reached), `next_any` (same as immediate but outside a hold),
            `next` (next occurrence of `safe_point`), `epoch` (next `safe_point` at or
            after `epoch`), or `step` (next safe point at or after `step`).
        epoch: Target epoch for `occurrence == "epoch"`.
        step: Target global step for `occurrence == "step"`.
    """

    safe_point: SafePoint | None
    occurrence: str
    epoch: int | None = None
    step: int | None = None

    def is_due(
        self, safe_point: SafePoint | None, epoch: int, global_step: int
    ) -> bool:
        """Decides whether this condition is satisfied at the given point in training."""
        if self.occurrence in ("immediate", "next_any"):
            return True
        if safe_point is None:
            return False
        if self.occurrence == "next":
            return safe_point == self.safe_point
        if self.occurrence == "epoch":
            return (
                safe_point == self.safe_point
                and self.epoch is not None
                and epoch >= self.epoch
            )
        if self.occurrence == "step":
            return self.step is not None and global_step >= self.step
        raise ValueError(f"unknown occurrence {self.occurrence!r}")


def parse_when(raw: object) -> CommandWhen:
    """Parses a REST-supplied `when` value into a `CommandWhen`.

    Accepts the shapes documented in the Trainctl REST API: the strings `"now"` and
    `"next_safe_point"`, or a mapping with a `safe_point` key plus either `occurrence`,
    `epoch`, or `global_step`.

    Args:
        raw: The decoded JSON value of the `when`/`execution.when` field.

    Returns:
        The normalized scheduling condition.

    Raises:
        ValueError: `raw` does not match a supported shape.
    """
    if raw in ("now", None):
        return CommandWhen(safe_point=None, occurrence="immediate")
    if raw == "next_safe_point":
        return CommandWhen(safe_point=None, occurrence="next_any")
    if isinstance(raw, dict):
        safe_point_raw = raw.get("safe_point")
        safe_point = SafePoint(safe_point_raw) if safe_point_raw else None
        if "epoch" in raw:
            return CommandWhen(
                safe_point=safe_point, occurrence="epoch", epoch=int(raw["epoch"])
            )
        if "global_step" in raw:
            return CommandWhen(
                safe_point=safe_point, occurrence="step", step=int(raw["global_step"])
            )
        occurrence = raw.get("occurrence", "next")
        return CommandWhen(safe_point=safe_point, occurrence=occurrence)
    raise ValueError(f"invalid 'when' spec: {raw!r}")


@dataclass
class Command:
    """A scheduled control-plane command and its execution outcome.

    Attributes:
        id: Unique command id.
        kind: One of `hold`, `stop`, `checkpoint`, `snapshot`, `set_learning_rate`,
            `reset_momentum`, `set_dataloader_finite_check`, `save_current_batch`,
            `save_current_loss`, `save_partial_gradients`, `save_optimizer_summary`,
            `release_exception`, `torchinfo_capture`, `profile`, `torchview_capture`,
            `torchviz_capture`, `torchlens_capture`. CUDA memory snapshots and
            allocator-history toggling are not commands -- see `GET
            /debug/cuda-memory/snapshot` and `GET /debug/cuda-memory/history`, which
            don't touch the Trainer/model and so don't need training-thread mediation.
        args: Handler-specific arguments.
        when: The scheduling condition that makes this command due.
        created_at: Unix timestamp of submission.
        status: `queued`, `executing`, `succeeded`, `failed`, or `cancelled`.
        result: Handler result payload, once succeeded.
        error: Failure message, once failed.
    """

    id: str
    kind: str
    args: dict[str, Any]
    when: CommandWhen
    created_at: float
    status: str = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None


class CommandQueue:
    """Thread-safe registry of submitted commands and their due-ness evaluation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._commands: dict[str, Command] = {}

    def submit(self, kind: str, args: dict[str, Any], when: CommandWhen) -> Command:
        """Registers a new queued command."""
        command = Command(
            id=next_id("cmd"), kind=kind, args=args, when=when, created_at=time.time()
        )
        with self._lock:
            self._commands[command.id] = command
        return command

    def get(self, command_id: str) -> Command | None:
        """Returns a command by id, or None."""
        with self._lock:
            return self._commands.get(command_id)

    def list_all(self) -> list[Command]:
        """Returns all known commands, in submission order."""
        with self._lock:
            return list(self._commands.values())

    def due(
        self, safe_point: SafePoint | None, epoch: int, global_step: int
    ) -> list[Command]:
        """Returns queued commands whose `when` condition is satisfied now."""
        with self._lock:
            return [
                command
                for command in self._commands.values()
                if command.status == "queued"
                and command.when.is_due(safe_point, epoch, global_step)
            ]

    def cancel(self, command_id: str) -> bool:
        """Cancels a still-queued command.

        Returns:
            True if cancelled, False if unknown or already past `queued`.
        """
        with self._lock:
            command = self._commands.get(command_id)
            if command is None or command.status != "queued":
                return False
            command.status = "cancelled"
            return True

    def mark_executing(self, command: Command) -> None:
        """Transitions a command to `executing`."""
        with self._lock:
            command.status = "executing"

    def mark_result(
        self,
        command: Command,
        *,
        succeeded: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Records a command's terminal outcome."""
        with self._lock:
            command.status = "succeeded" if succeeded else "failed"
            command.result = result
            command.error = error
