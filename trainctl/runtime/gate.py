"""Cooperative hold gate coordinating the Lightning training thread and REST.

The gate itself only tracks whether a hold is active and provides the events used to
wait for and signal release. Deciding *when* a hold should start is the command
scheduler's job (`runtime.commands`): a `hold` command becomes due at a safe point like
any other command, and its handler calls `RunGate.enter`.
"""

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveHold:
    """A hold currently blocking the training thread.

    Attributes:
        command_id: The `hold` command whose execution entered this hold.
        safe_point: The safe point the hold was entered at.
        epoch: Epoch at hold entry.
        step: Global step at hold entry.
        started_at: Unix timestamp of hold entry.
    """

    command_id: str
    safe_point: str
    epoch: int
    step: int
    started_at: float


class RunGate:
    """Tracks the active hold and the events used to wait for/signal release."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_hold: ActiveHold | None = None
        self._release_event = threading.Event()
        self._control_event = threading.Event()

    def enter(
        self, command_id: str, safe_point: str, epoch: int, step: int
    ) -> ActiveHold:
        """Marks the gate held. Must be called from the training thread."""
        hold = ActiveHold(
            command_id=command_id,
            safe_point=safe_point,
            epoch=epoch,
            step=step,
            started_at=time.time(),
        )
        with self._lock:
            self._active_hold = hold
            self._release_event.clear()
        return hold

    def is_held(self) -> bool:
        """Returns whether the gate currently blocks training progress."""
        with self._lock:
            return self._active_hold is not None

    def active_hold(self) -> ActiveHold | None:
        """Returns the active hold, or None."""
        with self._lock:
            return self._active_hold

    def release(self) -> bool:
        """Releases an active hold, if any. Safe to call from REST or training threads.

        Returns:
            True if a hold was released, False if the gate was not held.
        """
        with self._lock:
            if self._active_hold is None:
                return False
            self._active_hold = None
        self._release_event.set()
        self._control_event.set()
        return True

    def notify(self) -> None:
        """Wakes a training thread blocked in `wait_for_control_event` (e.g. new command)."""
        self._control_event.set()

    def wait_for_control_event(self, timeout: float) -> None:
        """Blocks the calling thread until notified or `timeout` elapses."""
        self._control_event.wait(timeout)
        self._control_event.clear()
