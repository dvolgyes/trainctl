"""On-demand CUDA memory snapshots and bounded allocator-history recording.

Orchestrates `torch.cuda.memory`'s own snapshot/history APIs -- no allocator
reimplementation. `_record_memory_history`'s enabled/disabled flag is process-global
torch state, not per-runtime, so callers (tests included) must restore it afterward.
"""

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CudaMemoryHistoryState:
    """Current allocator-history recording configuration.

    Attributes:
        enabled: Whether `torch.cuda.memory._record_memory_history` is active.
        max_entries: Bound passed to `_record_memory_history`, if any.
        started_at: Unix timestamp history recording was last enabled, if any.
    """

    enabled: bool
    max_entries: int | None
    started_at: float | None


def available() -> bool:
    """Returns whether a CUDA device is available in this process."""
    return torch.cuda.is_available()


def cheap_stats() -> dict[str, Any]:
    """Returns cheap, always-safe-to-read CUDA allocator counters.

    Returns `{"available": False}` when no CUDA device is present.
    """
    if not torch.cuda.is_available():
        return {"available": False}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "available": True,
        "memory_allocated": torch.cuda.memory_allocated(),
        "memory_reserved": torch.cuda.memory_reserved(),
        "max_memory_allocated": torch.cuda.max_memory_allocated(),
        "max_memory_reserved": torch.cuda.max_memory_reserved(),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
    }


def set_history(enabled: bool, max_entries: int | None) -> CudaMemoryHistoryState:
    """Enables or disables bounded CUDA allocator-history recording.

    Raises:
        RuntimeError: CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_memory history requires a CUDA device")
    if enabled:
        kwargs: dict[str, Any] = {"enabled": "all"}
        if max_entries is not None:
            kwargs["max_entries"] = max_entries
        torch.cuda.memory._record_memory_history(**kwargs)
        return CudaMemoryHistoryState(
            enabled=True, max_entries=max_entries, started_at=time.time()
        )
    torch.cuda.memory._record_memory_history(enabled=None)
    return CudaMemoryHistoryState(enabled=False, max_entries=None, started_at=None)


def dump_snapshot() -> bytes:
    """Dumps the current CUDA allocator snapshot and returns its pickle bytes.

    Raises:
        RuntimeError: CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("cuda_memory snapshot requires a CUDA device")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "snapshot.pickle"
        torch.cuda.memory._dump_snapshot(str(path))
        return path.read_bytes()
