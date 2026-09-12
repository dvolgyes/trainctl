"""Behavioral tests for the dataloader finite-value guard and its hold integration."""

import threading
import time
from collections import namedtuple
from pathlib import Path

import torch

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.runtime.artifacts import ArtifactStore
from trainctl.runtime.finite_guard import DataloaderFiniteGuard, iter_batch_tensors
from trainctl.runtime.runtime import TrainctlRuntime

_Pair = namedtuple("_Pair", ["inputs", "targets"])


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


class _FakePLModule:
    def __init__(self, trainer: _FakeTrainer) -> None:
        self.trainer = trainer


def test_iter_batch_tensors_traverses_nested_containers() -> None:
    batch = {
        "inputs": torch.zeros(2, 2),
        "meta": [torch.ones(3), (torch.ones(1),)],
    }
    refs = iter_batch_tensors(batch)
    assert {ref.path for ref in refs} == {
        "batch['inputs']",
        "batch['meta'][0]",
        "batch['meta'][1][0]",
    }


def test_iter_batch_tensors_traverses_namedtuples() -> None:
    batch = _Pair(inputs=torch.zeros(2), targets=torch.ones(2))
    refs = iter_batch_tensors(batch)
    assert {ref.path for ref in refs} == {"batch[0]", "batch[1]"}


def test_iter_batch_tensors_skips_integer_and_boolean_tensors() -> None:
    batch = {
        "labels": torch.zeros(2, dtype=torch.long),
        "mask": torch.ones(2, dtype=torch.bool),
    }
    assert iter_batch_tensors(batch) == []


def test_iter_batch_tensors_skips_unknown_container_types() -> None:
    class _Opaque:
        tensor = torch.zeros(2)

    assert iter_batch_tensors(_Opaque()) == []


def test_check_disabled_never_traverses_batch() -> None:
    guard = DataloaderFiniteGuard(enabled=False)
    report = guard.check({"x": torch.tensor([float("nan")])})
    assert report is None
    assert guard.checked_batches == 0


def test_check_finds_nan_posinf_neginf_counts() -> None:
    guard = DataloaderFiniteGuard(enabled=True)
    batch = {"x": torch.tensor([float("nan"), float("inf"), float("-inf"), 1.0])}

    report = guard.check(batch)

    assert report is not None
    assert len(report.bad_tensors) == 1
    bad = report.bad_tensors[0]
    assert bad.nan_count == 1
    assert bad.posinf_count == 1
    assert bad.neginf_count == 1
    assert guard.checked_batches == 1
    assert guard.failures == 1


def test_check_returns_none_for_finite_batch() -> None:
    guard = DataloaderFiniteGuard(enabled=True)
    report = guard.check({"x": torch.ones(4)})
    assert report is None
    assert guard.checked_batches == 1
    assert guard.failures == 0


def test_begin_train_batch_enters_hold_on_bad_batch(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, dataloader_finite_check_enabled=True)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)
    runtime._gather_ranks(pl_module)

    thread = threading.Thread(
        target=runtime.begin_train_batch,
        args=(pl_module, {"x": torch.tensor([float("nan")])}, 0),
    )
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not runtime.gate.is_held() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runtime.gate.is_held()

        snapshots = sorted((runtime.artifacts.root / "snapshots").iterdir())
        assert len(snapshots) == 1
        assert (snapshots[0] / "anomaly_input_nonfinite.json").exists()
    finally:
        runtime.gate.release()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_begin_train_batch_does_not_hold_on_finite_batch(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, dataloader_finite_check_enabled=True)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    runtime.begin_train_batch(pl_module, batch={"x": torch.ones(4)}, batch_idx=0)

    assert not runtime.gate.is_held()


def test_begin_train_batch_skips_check_entirely_when_guard_disabled(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path, dataloader_finite_check_enabled=False)
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    runtime.begin_train_batch(
        pl_module, batch={"x": torch.tensor([float("nan")])}, batch_idx=0
    )

    assert not runtime.gate.is_held()
    assert runtime.finite_guard.checked_batches == 0
