"""Behavioral tests for torchview/torchviz graph capture and their command handlers.

torchview/torchviz are pip packages present in `dev`, but both also require the
system Graphviz `dot` binary to actually render -- `shutil.which("dot") is None` is a
second, independent skip layer alongside the python-import check.
"""

import shutil
import sys
from pathlib import Path

import pytest
import torch

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.runtime import graph_inspectors
from trainctl.runtime.artifacts import ArtifactStore
from trainctl.runtime.commands import parse_when
from trainctl.runtime.runtime import TrainctlRuntime

pytestmark = pytest.mark.skipif(
    shutil.which("dot") is None, reason="Graphviz 'dot' binary not installed"
)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_runtime(tmp_path: Path, **overrides: object) -> TrainctlRuntime:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False, **overrides
    )
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return runtime


class _FakeState:
    stage = None


class _FakeTrainer:
    def __init__(self) -> None:
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.optimizers: list[object] = []
        self.state = _FakeState()


def test_torchview_available_true_when_installed() -> None:
    assert graph_inspectors.torchview_available() is True


def test_torchview_available_false_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torchview", None)
    assert graph_inspectors.torchview_available() is False


def test_torchviz_available_false_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torchviz", None)
    assert graph_inspectors.torchviz_available() is False


def test_torchview_capture_renders_svg_and_dot() -> None:
    model = _Model()
    files = graph_inspectors.torchview_capture(model, torch.randn(3, 4))
    assert files["graph.svg"].startswith(b"<?xml") or b"<svg" in files["graph.svg"]
    assert "digraph" in files["graph.dot"]


def test_torchviz_capture_renders_svg_and_dot() -> None:
    model = _Model()
    x = torch.randn(3, 4, requires_grad=True)
    loss = model(x).sum()
    files = graph_inspectors.torchviz_capture(model, loss)
    assert b"<svg" in files["graph.svg"] or files["graph.svg"].startswith(b"<?xml")
    assert "digraph" in files["graph.dot"]


def test_torchview_capture_command_writes_artifact(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()
    runtime.current_step.begin(
        batch=torch.randn(3, 4), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )

    cmd = runtime.command_queue.submit("torchview_capture", {}, parse_when("now"))
    runtime._execute(model, trainer, cmd, safe_point=None)

    assert cmd.status == "succeeded"
    assert (Path(cmd.result["path"]) / "graph.svg").exists()


def test_torchview_capture_fails_without_in_flight_batch(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()

    cmd = runtime.command_queue.submit("torchview_capture", {}, parse_when("now"))
    runtime._execute(model, trainer, cmd, safe_point=None)

    assert cmd.status == "failed"
    assert "in-flight batch" in cmd.error


def test_torchviz_capture_command_writes_artifact(tmp_path: Path) -> None:
    """A normal (non-exception-hold) capture runs a fresh forward pass rather than
    reusing `current_step.loss`, which is always already-backwarded by Lightning
    itself by the time any command can execute."""
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()
    runtime.current_step.begin(
        batch=torch.randn(3, 4), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )

    cmd = runtime.command_queue.submit("torchviz_capture", {}, parse_when("now"))
    runtime._execute(model, trainer, cmd, safe_point=None)

    assert cmd.status == "succeeded"
    assert (Path(cmd.result["path"]) / "graph.svg").exists()


def test_torchviz_capture_fails_without_in_flight_batch(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()

    cmd = runtime.command_queue.submit("torchviz_capture", {}, parse_when("now"))
    runtime._execute(model, trainer, cmd, safe_point=None)

    assert cmd.status == "failed"
    assert "in-flight batch" in cmd.error


def test_torchviz_capture_uses_current_step_loss_during_exception_hold(
    tmp_path: Path,
) -> None:
    """During an active exception hold, the actual failed loss's graph is what's
    worth inspecting -- not a fresh, unrelated forward pass."""
    from trainctl.runtime.state import ExceptionBreakpointInfo

    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()
    x = torch.randn(3, 4, requires_grad=True)
    runtime.current_step.loss = model(x).sum()
    runtime._exception_hold = ExceptionBreakpointInfo(
        exception_type="ValueError",
        exception_message="boom",
        epoch=0,
        global_step=0,
        batch_idx=0,
        rank=0,
        started_at=0.0,
    )

    cmd = runtime.command_queue.submit("torchviz_capture", {}, parse_when("now"))
    runtime._execute(model, trainer, cmd, safe_point=None)

    assert cmd.status == "succeeded"
    assert (Path(cmd.result["path"]) / "graph.svg").exists()


def test_torchviz_capture_fails_without_in_flight_loss_during_exception_hold(
    tmp_path: Path,
) -> None:
    from trainctl.runtime.state import ExceptionBreakpointInfo

    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()
    runtime._exception_hold = ExceptionBreakpointInfo(
        exception_type="ValueError",
        exception_message="boom",
        epoch=0,
        global_step=0,
        batch_idx=0,
        rank=0,
        started_at=0.0,
    )

    cmd = runtime.command_queue.submit("torchviz_capture", {}, parse_when("now"))
    runtime._execute(model, trainer, cmd, safe_point=None)

    assert cmd.status == "failed"
    assert "in-flight loss" in cmd.error
