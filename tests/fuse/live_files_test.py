"""Behavioral tests for lazy, safe-point-gated live FUSE files.

/model/summary-live.txt and /model/graph.dot block the reading thread until the
training thread reaches SafePoint.BEFORE_OPTIMIZER_STEP, compute a fresh value from
the in-flight batch there, and never persist an artifact -- unlike torchinfo_capture/
torchview_capture, submitted via POST /commands.
"""

import errno
import threading
import time

import mfusepy
import pytest
import torch

from trainctl.config import TrainctlConfig
from trainctl.fuse.filesystem import TrainctlFS
from trainctl.lightning_backend import load_backend
from trainctl.runtime import virtual_fs as fs_module
from trainctl.runtime.events import SafePoint
from trainctl.runtime.runtime import TrainctlRuntime


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _FakeState:
    stage = None


class _FakeTrainer:
    def __init__(self) -> None:
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.optimizers: list[object] = []
        self.state = _FakeState()


def _make_fs() -> tuple[TrainctlFS, TrainctlRuntime]:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
    return TrainctlFS(runtime), runtime


def _wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate(), "condition not met before timeout"


def _drive_next_safe_point(runtime: TrainctlRuntime, model: _Model) -> None:
    _wait_until(lambda: len(runtime.live_captures._pending) >= 1)
    runtime.process_safe_point(model, SafePoint.BEFORE_OPTIMIZER_STEP)


def test_model_summary_live_txt_blocks_then_serves_fresh_data() -> None:
    fs, runtime = _make_fs()
    model = _Model()
    model.trainer = _FakeTrainer()
    runtime._gather_ranks(model)
    runtime.current_step.begin(
        batch=torch.randn(3, 4), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )
    threading.Thread(
        target=_drive_next_safe_point, args=(runtime, model), daemon=True
    ).start()

    content = fs_module.generate(runtime, "/model/summary-live.txt")

    assert b"Linear" in content


def test_model_graph_dot_blocks_then_serves_dot_source() -> None:
    fs, runtime = _make_fs()
    model = _Model()
    model.trainer = _FakeTrainer()
    runtime._gather_ranks(model)
    runtime.current_step.begin(
        batch=torch.randn(3, 4), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )
    threading.Thread(
        target=_drive_next_safe_point, args=(runtime, model), daemon=True
    ).start()

    content = fs_module.generate(runtime, "/model/graph.dot")

    assert b"digraph" in content


def test_getattr_reports_zero_size_without_computing(monkeypatch: pytest.MonkeyPatch) -> None:
    fs, _runtime = _make_fs()

    def _boom(rt: object) -> bytes:
        raise AssertionError("getattr must not compute a live file")

    monkeypatch.setitem(fs_module._LIVE_FILES, "/model/summary-live.txt", _boom)

    attrs = fs.getattr("/model/summary-live.txt")

    assert attrs["st_size"] == 0


def test_open_translates_timeout_to_fuse_errno(monkeypatch: pytest.MonkeyPatch) -> None:
    fs, _runtime = _make_fs()

    def _raise_timeout(rt: object) -> bytes:
        raise TimeoutError("no safe point reached")

    monkeypatch.setitem(fs_module._LIVE_FILES, "/model/summary-live.txt", _raise_timeout)

    with pytest.raises(mfusepy.FuseOSError) as excinfo:
        fs.open("/model/summary-live.txt", 0)
    assert excinfo.value.errno == errno.ETIMEDOUT


def test_open_translates_other_failures_to_eio(monkeypatch: pytest.MonkeyPatch) -> None:
    fs, _runtime = _make_fs()

    def _raise_value_error(rt: object) -> bytes:
        raise ValueError("requires an in-flight batch")

    monkeypatch.setitem(
        fs_module._LIVE_FILES, "/model/summary-live.txt", _raise_value_error
    )

    with pytest.raises(mfusepy.FuseOSError) as excinfo:
        fs.open("/model/summary-live.txt", 0)
    assert excinfo.value.errno == errno.EIO


def test_model_dir_lists_live_files() -> None:
    fs, _runtime = _make_fs()
    names = fs.readdir("/model", 0)
    assert "summary-live.txt" in names
    assert "graph.dot" in names
