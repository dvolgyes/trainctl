"""Behavioral tests for the torchinfo inspector and its command handler."""

import sys
from pathlib import Path

import pytest
import torch

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.runtime import torchinfo_inspector
from trainctl.runtime.artifacts import ArtifactStore
from trainctl.runtime.commands import parse_when
from trainctl.runtime.runtime import TrainctlRuntime


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


def test_basic_summary_json_counts_params_without_torchinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torchinfo", None)
    model = _Model()
    payload = torchinfo_inspector.basic_summary_json(model)
    assert payload["total_params"] == 10
    assert payload["trainable_params"] == 10
    assert payload["non_trainable_params"] == 0
    assert payload["torchinfo_used"] is False


def test_basic_summary_text_falls_back_to_str_without_torchinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torchinfo", None)
    model = _Model()
    assert torchinfo_inspector.basic_summary_text(model) == str(model)


def test_basic_summary_text_uses_torchinfo_when_available() -> None:
    model = _Model()
    text = torchinfo_inspector.basic_summary_text(model)
    assert "Total params" in text


def test_rich_summary_raises_without_torchinfo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torchinfo", None)
    model = _Model()
    with pytest.raises(ImportError):
        torchinfo_inspector.rich_summary(model, torch.randn(3, 4))


def test_rich_summary_does_not_perturb_batchnorm_running_stats() -> None:
    class BNModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bn(x)

    model = BNModel()
    model.train()
    running_mean_before = model.bn.running_mean.clone()

    torchinfo_inspector.rich_summary(model, torch.randn(8, 4) * 10 + 5)

    assert model.training is True
    assert torch.equal(running_mean_before, model.bn.running_mean)


def test_torchinfo_capture_command_writes_artifact(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()
    runtime.current_step.begin(
        batch=torch.randn(3, 4), batch_idx=0, epoch=0, global_step=0, started_ns=0
    )

    cmd = runtime.command_queue.submit("torchinfo_capture", {}, parse_when("now"))
    runtime._execute(model, trainer, cmd, safe_point=None)

    assert cmd.status == "succeeded"
    assert (Path(cmd.result["path"]) / "summary.txt").exists()
    assert (Path(cmd.result["path"]) / "summary.json").exists()


def test_torchinfo_capture_fails_without_in_flight_batch(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    model = _Model()

    cmd = runtime.command_queue.submit("torchinfo_capture", {}, parse_when("now"))
    runtime._execute(model, trainer, cmd, safe_point=None)

    assert cmd.status == "failed"
    assert "in-flight batch" in cmd.error
