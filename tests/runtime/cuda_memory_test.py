"""Behavioral tests for the CUDA memory inspector's pure functions.

`_record_memory_history` toggles process-global torch state, so every test that
enables it must disable it again in a `finally`, regardless of outcome. REST-route
behavior (GET /debug/cuda-memory/snapshot, GET /debug/cuda-memory/history) is
covered separately in tests/rest/cuda_memory_test.py -- these operations don't
touch the Trainer/model, so they execute directly from REST, not as commands.
"""

import pytest
import torch

from trainctl.runtime import cuda_memory

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def test_cheap_stats_reports_real_allocator_counters() -> None:
    stats = cuda_memory.cheap_stats()
    assert stats["available"] is True
    assert stats["free_bytes"] > 0
    assert stats["total_bytes"] > 0


def test_set_history_enable_and_disable_round_trip() -> None:
    try:
        state = cuda_memory.set_history(True, max_entries=1000)
        assert state.enabled is True
        assert state.max_entries == 1000
        assert state.started_at is not None
    finally:
        disabled = cuda_memory.set_history(False, max_entries=None)
        assert disabled.enabled is False


def test_dump_snapshot_returns_nonempty_pickle_bytes() -> None:
    try:
        cuda_memory.set_history(True, max_entries=1000)
        tensor = torch.randn(1000, 1000, device="cuda")
        _ = tensor @ tensor
        snapshot = cuda_memory.dump_snapshot()
        assert isinstance(snapshot, bytes)
        assert len(snapshot) > 0
    finally:
        cuda_memory.set_history(False, max_entries=None)
