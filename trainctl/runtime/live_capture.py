"""Cross-thread hand-off for cheap, on-demand live views computed on the training thread.

Purpose-built for lazy FUSE files (see `trainctl.fuse.filesystem`): a reader thread
blocks in `LiveCaptureBroker.request` until the training thread calls `serve` from
inside a safe point, computes the callable there (where `current_step`/`pl_module` are
valid), and wakes the caller. Unlike `Command`, a live capture is never persisted to
`ArtifactStore` and never appears in `GET /commands` -- it is a transient read, not an
audited action.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class _PendingCapture:
    fn: Callable[[Any], Any]
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None


class LiveCaptureBroker:
    """Thread-safe hand-off of one-shot computations to the training thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[_PendingCapture] = []

    def request(self, fn: Callable[[Any], Any], timeout: float) -> Any:
        """Blocks until `fn(pl_module)` runs on the training thread, or `timeout` elapses.

        Args:
            fn: Called with `pl_module` from the training thread at the next safe point
                `serve` is invoked for. Any exception it raises is re-raised here, in
                the calling thread.
            timeout: Seconds to wait for a matching safe point.

        Returns:
            `fn`'s return value.

        Raises:
            TimeoutError: no matching safe point was reached within `timeout`.
        """
        pending = _PendingCapture(fn=fn)
        with self._lock:
            self._pending.append(pending)
        if not pending.event.wait(timeout):
            with self._lock:
                if pending in self._pending:
                    self._pending.remove(pending)
            raise TimeoutError(
                "no safe point reached within timeout; is training running?"
            )
        if pending.error is not None:
            raise pending.error
        return pending.result

    def serve(self, pl_module: Any) -> None:
        """Runs and wakes every pending request. Call only from the training thread."""
        with self._lock:
            pending, self._pending = self._pending, []
        for item in pending:
            try:
                item.result = item.fn(pl_module)
            except Exception as exc:  # noqa: BLE001 -- boundary: a live-view failure must not crash training
                logger.warning("trainctl live capture failed: {}", exc)
                item.error = exc
            finally:
                item.event.set()
