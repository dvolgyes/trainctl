"""Behavioral tests for CurrentStep retention and TrainctlRuntime's per-batch wiring."""

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.runtime.current_step import CurrentStep
from trainctl.runtime.runtime import TrainctlRuntime


def _make_runtime() -> TrainctlRuntime:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    return TrainctlRuntime(config, load_backend("lightning.pytorch"))


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


def test_begin_sets_fields_and_discards_previous_batch() -> None:
    step = CurrentStep()
    step.begin(batch="first", batch_idx=0, epoch=0, global_step=0, started_ns=100)
    step.loss = "first-loss"
    step.exception = ValueError("boom")

    step.begin(batch="second", batch_idx=1, epoch=0, global_step=1, started_ns=200)

    assert step.batch == "second"
    assert step.batch_idx == 1
    assert step.loss is None
    assert step.exception is None


def test_clear_drops_live_references_but_keeps_scalar_markers() -> None:
    step = CurrentStep()
    step.begin(batch="b", batch_idx=3, epoch=1, global_step=7, started_ns=100)
    step.loss = "loss"
    step.exception = RuntimeError("failed")

    step.clear()

    assert step.batch is None
    assert step.loss is None
    assert step.exception is None
    assert step.epoch == 1
    assert step.batch_idx == 3


def test_begin_train_batch_populates_current_step() -> None:
    runtime = _make_runtime()
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    runtime.begin_train_batch(pl_module, batch={"x": 1}, batch_idx=2)

    assert runtime.current_step.batch == {"x": 1}
    assert runtime.current_step.batch_idx == 2


def test_finish_train_batch_clears_current_step() -> None:
    runtime = _make_runtime()
    trainer = _FakeTrainer()
    pl_module = _FakePLModule(trainer)

    runtime.begin_train_batch(pl_module, batch={"x": 1}, batch_idx=0)
    runtime.current_step.loss = "loss"
    runtime.finish_train_batch(pl_module)

    assert runtime.current_step.batch is None
    assert runtime.current_step.loss is None
