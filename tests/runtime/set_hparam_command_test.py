"""Behavioral tests for the `set_hparam` command, end-to-end through `process_safe_point`.

Exercises the full path a REST-submitted mutation takes: `POST /commands` enqueues,
the training thread executes it at a safe point via `_handle_set_hparam`, and
`process_safe_point` refreshes `runtime.tunable_hparams` (read back at
`GET /files/model/tunable-hparams.json`) -- not just `hparam_registry`'s functions in
isolation.
"""

from pathlib import Path

import torch

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.runtime.artifacts import ArtifactStore
from trainctl.runtime.commands import parse_when
from trainctl.runtime.events import SafePoint
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


class _Model(torch.nn.Module):
    trainctl_tunable_hparams = ["augmentation_strength"]

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
        self.augmentation_strength = 0.5


def _ready_model(runtime: TrainctlRuntime) -> _Model:
    model = _Model()
    model.trainer = _FakeTrainer()
    runtime._gather_ranks(model)
    runtime.on_fit_start(model)
    return model


def test_registered_hparam_updates_after_the_next_safe_point(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    model = _ready_model(runtime)

    command = runtime.command_queue.submit(
        "set_hparam",
        {"name": "augmentation_strength", "value": 0.9},
        parse_when("now"),
    )
    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)

    assert runtime.command_queue.get(command.id).status == "succeeded"
    assert model.augmentation_strength == 0.9
    assert runtime.tunable_hparams["augmentation_strength"].value == 0.9
    assert runtime.tunable_hparams["augmentation_strength"].type == "float"


def test_unregistered_hparam_name_is_rejected(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    model = _ready_model(runtime)

    command = runtime.command_queue.submit(
        "set_hparam", {"name": "not_registered", "value": 1}, parse_when("now")
    )
    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)

    result = runtime.command_queue.get(command.id)
    assert result.status == "failed"
    assert "not_registered" in result.error
    assert model.augmentation_strength == 0.5


def test_tunable_hparams_reflects_direct_attribute_changes_after_safe_point(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    model = _ready_model(runtime)

    model.augmentation_strength = 0.3
    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)

    assert runtime.tunable_hparams["augmentation_strength"].value == 0.3


def test_unregistered_model_has_no_tunable_hparams(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    model = torch.nn.Linear(2, 2)
    model.trainer = _FakeTrainer()
    runtime._gather_ranks(model)
    runtime.on_fit_start(model)

    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)

    assert runtime.tunable_hparams == {}


class _BoundedSubModel(_Model):
    trainctl_tunable_hparams = {
        "learning_rate": {"min": 0.0, "max": 1.0, "label": "Learning rate"},
    }

    def __init__(self) -> None:
        super().__init__()
        self.learning_rate = 0.1


def _ready_bounded_model(runtime: TrainctlRuntime) -> _BoundedSubModel:
    model = _BoundedSubModel()
    model.trainer = _FakeTrainer()
    runtime._gather_ranks(model)
    runtime.on_fit_start(model)
    return model


def test_mro_merged_hparams_include_ancestor_list_and_subclass_dict_forms(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    model = _ready_bounded_model(runtime)

    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)

    assert set(runtime.tunable_hparams) == {"augmentation_strength", "learning_rate"}
    lr = runtime.tunable_hparams["learning_rate"]
    assert lr.value == 0.1
    assert lr.min == 0.0
    assert lr.max == 1.0
    assert lr.label == "Learning rate"


def test_in_bounds_set_hparam_succeeds(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    model = _ready_bounded_model(runtime)

    command = runtime.command_queue.submit(
        "set_hparam", {"name": "learning_rate", "value": 0.5}, parse_when("now")
    )
    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)

    assert runtime.command_queue.get(command.id).status == "succeeded"
    assert model.learning_rate == 0.5


def test_out_of_bounds_set_hparam_is_rejected_with_unchanged_value(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    model = _ready_bounded_model(runtime)

    command = runtime.command_queue.submit(
        "set_hparam", {"name": "learning_rate", "value": 5.0}, parse_when("now")
    )
    runtime.process_safe_point(model, SafePoint.TRAIN_BATCH_END)

    result = runtime.command_queue.get(command.id)
    assert result.status == "failed"
    assert "learning_rate" in result.error
    assert "1.0" in result.error
    assert model.learning_rate == 0.1
