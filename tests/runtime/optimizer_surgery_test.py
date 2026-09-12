"""Behavioral tests for optimizer surgery: LR changes and momentum reset."""

import pytest
import torch

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.runtime.commands import Command, parse_when
from trainctl.runtime.optimizer_surgery import (
    AdamSurgeryAdapter,
    SGDSurgeryAdapter,
    find_adapter,
    resolve_parameters,
)
from trainctl.runtime.runtime import TrainctlRuntime


def _make_runtime(**overrides: object) -> TrainctlRuntime:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False, **overrides
    )
    return TrainctlRuntime(config, load_backend("lightning.pytorch"))


class _FakeState:
    stage = None


class _FakeTrainer:
    def __init__(self, optimizers: list | None = None) -> None:
        self.callback_metrics: dict[str, float] = {}
        self.current_epoch = 0
        self.global_step = 0
        self.optimizers = optimizers or []
        self.state = _FakeState()
        self.lr_scheduler_configs: list = []


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(2, 2)
        self.b = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


class _FakePLModule:
    def __init__(self, trainer: _FakeTrainer, model: torch.nn.Module) -> None:
        self.trainer = trainer
        self._model = model

    def named_parameters(self):
        return self._model.named_parameters()


def _step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad()
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()


def _submit_and_run(
    runtime: TrainctlRuntime,
    pl_module: _FakePLModule,
    trainer: _FakeTrainer,
    kind: str,
    args: dict,
) -> Command:
    command = runtime.command_queue.submit(kind, args, parse_when("now"))
    runtime._execute(pl_module, trainer, command, safe_point=None)
    return command


# ---- adapter unit tests ----


def test_find_adapter_returns_none_for_unsupported_optimizer() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adagrad(model.parameters())
    assert find_adapter(optimizer) is None


def test_sgd_adapter_drops_momentum_buffer() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    _step(model, optimizer)
    assert all("momentum_buffer" in optimizer.state[p] for p in model.parameters())

    result = SGDSurgeryAdapter().reset_momentum(optimizer, list(model.parameters()))

    assert result["semantic"] == "momentum_buffer_dropped"
    assert all("momentum_buffer" not in optimizer.state[p] for p in model.parameters())


def test_adam_adapter_zeroes_exp_avg_but_not_exp_avg_sq_or_step() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    _step(model, optimizer)
    before_sq = {
        id(p): optimizer.state[p]["exp_avg_sq"].clone() for p in model.parameters()
    }
    before_step = {id(p): optimizer.state[p]["step"].clone() for p in model.parameters()}

    result = AdamSurgeryAdapter().reset_momentum(optimizer, list(model.parameters()))

    assert result["semantic"] == "exp_avg_zeroed"
    for p in model.parameters():
        assert torch.equal(
            optimizer.state[p]["exp_avg"], torch.zeros_like(optimizer.state[p]["exp_avg"])
        )
        assert torch.equal(optimizer.state[p]["exp_avg_sq"], before_sq[id(p)])
        assert torch.equal(optimizer.state[p]["step"], before_step[id(p)])


def test_resolve_parameters_all_returns_every_param() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resolved = resolve_parameters(model, optimizer, "all")
    assert len(resolved) == len(list(model.parameters()))


def test_resolve_parameters_by_name() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resolved = resolve_parameters(model, optimizer, {"parameter": "a.weight"})
    expected = dict(model.named_parameters())["a.weight"]
    assert len(resolved) == 1
    assert resolved[0] is expected


def test_resolve_parameters_by_prefix() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resolved = resolve_parameters(model, optimizer, {"parameter_prefix": "a."})
    resolved_ids = {id(p) for p in resolved}
    names = {name for name, p in model.named_parameters() if id(p) in resolved_ids}
    assert names == {"a.weight", "a.bias"}


def test_resolve_parameters_by_param_group() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(
        [{"params": model.a.parameters()}, {"params": model.b.parameters()}], lr=0.1
    )
    resolved = resolve_parameters(model, optimizer, {"param_group": 1})
    expected_ids = {id(p) for p in model.b.parameters()}
    assert {id(p) for p in resolved} == expected_ids


def test_resolve_parameters_rejects_unknown_shape() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="invalid momentum-reset selector"):
        resolve_parameters(model, optimizer, {"bogus": 1})


def test_resolve_parameters_rejects_unknown_name() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="no optimizer parameter named"):
        resolve_parameters(model, optimizer, {"parameter": "does-not-exist"})


# ---- command handler tests ----


def test_set_learning_rate_updates_all_groups_by_default() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime, pl_module, trainer, "set_learning_rate", {"value": 0.5}
    )

    assert command.status == "succeeded"
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_set_learning_rate_targets_single_param_group() -> None:
    runtime = _make_runtime()
    model = _Model()
    optimizer = torch.optim.SGD(
        [{"params": model.a.parameters()}, {"params": model.b.parameters()}], lr=0.1
    )
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime,
        pl_module,
        trainer,
        "set_learning_rate",
        {"value": 0.9, "param_group": 1},
    )

    assert command.status == "succeeded"
    assert optimizer.param_groups[0]["lr"] == 0.1
    assert optimizer.param_groups[1]["lr"] == 0.9


def test_set_learning_rate_rejects_non_finite_value() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime, pl_module, trainer, "set_learning_rate", {"value": float("nan")}
    )

    assert command.status == "failed"
    assert optimizer.param_groups[0]["lr"] == 0.1


def test_set_learning_rate_rejects_negative_value() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime, pl_module, trainer, "set_learning_rate", {"value": -1.0}
    )

    assert command.status == "failed"


def test_set_learning_rate_warns_when_scheduler_active() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _FakeTrainer(optimizers=[optimizer])
    trainer.lr_scheduler_configs = [object()]
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime, pl_module, trainer, "set_learning_rate", {"value": 0.5}
    )

    assert command.status == "succeeded"
    assert "warning" in command.result


def test_set_learning_rate_param_group_requires_surgery_enabled() -> None:
    runtime = _make_runtime(optimizer_surgery_enabled=False)
    model = _Model()
    optimizer = torch.optim.SGD(
        [{"params": model.a.parameters()}, {"params": model.b.parameters()}], lr=0.1
    )
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime,
        pl_module,
        trainer,
        "set_learning_rate",
        {"value": 0.5, "param_group": 0},
    )

    assert command.status == "failed"


def test_set_learning_rate_all_groups_still_works_when_surgery_disabled() -> None:
    runtime = _make_runtime(optimizer_surgery_enabled=False)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime, pl_module, trainer, "set_learning_rate", {"value": 0.5}
    )

    assert command.status == "succeeded"


def test_reset_momentum_zeroes_adam_exp_avg_via_command() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    _step(model, optimizer)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(runtime, pl_module, trainer, "reset_momentum", {})

    assert command.status == "succeeded"
    assert command.result["semantic"] == "exp_avg_zeroed"
    for p in model.parameters():
        assert torch.equal(
            optimizer.state[p]["exp_avg"], torch.zeros_like(optimizer.state[p]["exp_avg"])
        )


def test_reset_momentum_rejects_unsupported_optimizer_without_mutating_state() -> None:
    runtime = _make_runtime()
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
    _step(model, optimizer)
    before = {id(p): optimizer.state[p]["sum"].clone() for p in model.parameters()}
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(runtime, pl_module, trainer, "reset_momentum", {})

    assert command.status == "failed"
    for p in model.parameters():
        assert torch.equal(optimizer.state[p]["sum"], before[id(p)])


def test_reset_momentum_rejected_when_surgery_disabled() -> None:
    runtime = _make_runtime(optimizer_surgery_enabled=False)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    _step(model, optimizer)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(runtime, pl_module, trainer, "reset_momentum", {})

    assert command.status == "failed"


def test_reset_momentum_with_selector_only_resets_targeted_group() -> None:
    runtime = _make_runtime()
    model = _Model()
    optimizer = torch.optim.SGD(
        [{"params": model.a.parameters()}, {"params": model.b.parameters()}],
        lr=0.1,
        momentum=0.9,
    )
    _step(model, optimizer)
    trainer = _FakeTrainer(optimizers=[optimizer])
    pl_module = _FakePLModule(trainer, model)

    command = _submit_and_run(
        runtime,
        pl_module,
        trainer,
        "reset_momentum",
        {"selector": {"param_group": 0}},
    )

    assert command.status == "succeeded"
    for p in model.a.parameters():
        assert "momentum_buffer" not in optimizer.state[p]
    for p in model.b.parameters():
        assert "momentum_buffer" in optimizer.state[p]
