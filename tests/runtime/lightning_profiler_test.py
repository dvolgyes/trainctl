"""Behavioral tests for the `set_lightning_profiler` command handler."""

from pathlib import Path

import pytest

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.runtime.artifacts import ArtifactStore
from trainctl.runtime.commands import parse_when
from trainctl.runtime.runtime import TrainctlRuntime


def _make_runtime(tmp_path: Path) -> TrainctlRuntime:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
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


class _FakePLModule:
    def __init__(self, trainer: _FakeTrainer) -> None:
        self.trainer = trainer


def _set_level(runtime: TrainctlRuntime, trainer: _FakeTrainer, level: str):
    pl_module = _FakePLModule(trainer)
    cmd = runtime.command_queue.submit(
        "set_lightning_profiler", {"level": level}, parse_when("now")
    )
    runtime._execute(pl_module, trainer, cmd, safe_point=None)
    return cmd


@pytest.mark.parametrize(
    ("level", "class_name"),
    [
        ("simple", "SimpleProfiler"),
        ("advanced", "AdvancedProfiler"),
        ("pytorch", "PyTorchProfiler"),
    ],
)
def test_set_lightning_profiler_installs_the_requested_class(
    tmp_path: Path, level: str, class_name: str
) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()

    cmd = _set_level(runtime, trainer, level)

    assert cmd.status == "succeeded"
    assert cmd.result == {"level": level}
    assert type(trainer.profiler).__name__ == class_name
    assert trainer.profiler is runtime.lightning_profiler
    assert runtime.lightning_profiler_level == level


def test_set_lightning_profiler_off_restores_pass_through(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    _set_level(runtime, trainer, "simple")

    cmd = _set_level(runtime, trainer, "off")

    assert cmd.status == "succeeded"
    assert type(trainer.profiler).__name__ == "PassThroughProfiler"
    assert runtime.lightning_profiler is None
    assert runtime.lightning_profiler_level == "off"


def test_set_lightning_profiler_rejects_unknown_level(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()

    cmd = _set_level(runtime, trainer, "nonsense")

    assert cmd.status == "failed"
    assert "level" in cmd.error


def test_set_lightning_profiler_reselecting_same_level_resets_accumulated_state(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    trainer = _FakeTrainer()
    _set_level(runtime, trainer, "simple")
    runtime.lightning_profiler.start("some_action")
    runtime.lightning_profiler.stop("some_action")
    assert "some_action" in runtime.lightning_profiler.summary()

    _set_level(runtime, trainer, "simple")

    assert "some_action" not in runtime.lightning_profiler.summary()
