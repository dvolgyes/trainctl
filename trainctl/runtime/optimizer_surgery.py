"""Typed, conservative optimizer mutations: parameter selection and momentum reset.

No arbitrary optimizer-dict mutation and no arbitrary Python object paths -- only the
operations named here, against a small registry of known optimizer types. An unknown
optimizer type is rejected rather than guessed at.
"""

from typing import Any, Protocol

import torch


class OptimizerSurgeryAdapter(Protocol):
    """One optimizer family's momentum-reset semantics."""

    def supports(self, optimizer: torch.optim.Optimizer) -> bool:
        """Returns whether this adapter knows how to reset `optimizer`'s momentum."""
        ...

    def reset_momentum(
        self, optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter]
    ) -> dict[str, Any]:
        """Resets first-order momentum for `parameters` in `optimizer`'s state."""
        ...


class SGDSurgeryAdapter:
    """`torch.optim.SGD`: drops the momentum buffer, letting SGD recreate it as needed."""

    def supports(self, optimizer: torch.optim.Optimizer) -> bool:
        return isinstance(optimizer, torch.optim.SGD)

    def reset_momentum(
        self, optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter]
    ) -> dict[str, Any]:
        reset_count = 0
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            if state is not None and state.pop("momentum_buffer", None) is not None:
                reset_count += 1
        return {"semantic": "momentum_buffer_dropped", "parameters_reset": reset_count}


class AdamSurgeryAdapter:
    """`torch.optim.Adam`/`AdamW`: zeroes the first moment (`exp_avg`) in place only.

    `exp_avg_sq`, `max_exp_avg_sq`, and `step` are left untouched -- reset_momentum has
    one stable meaning: reset first-order momentum while retaining variance/step
    history. A separate `reset_variance`/`reset_optimizer_state` operation would be
    needed to touch those.
    """

    _TYPES = (torch.optim.Adam, torch.optim.AdamW)

    def supports(self, optimizer: torch.optim.Optimizer) -> bool:
        return isinstance(optimizer, self._TYPES)

    def reset_momentum(
        self, optimizer: torch.optim.Optimizer, parameters: list[torch.nn.Parameter]
    ) -> dict[str, Any]:
        reset_count = 0
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            exp_avg = state.get("exp_avg") if state is not None else None
            if exp_avg is not None:
                exp_avg.zero_()
                reset_count += 1
        return {"semantic": "exp_avg_zeroed", "parameters_reset": reset_count}


_ADAPTERS: tuple[OptimizerSurgeryAdapter, ...] = (SGDSurgeryAdapter(), AdamSurgeryAdapter())


def find_adapter(optimizer: torch.optim.Optimizer) -> OptimizerSurgeryAdapter | None:
    """Returns the first registered adapter supporting `optimizer`'s type, or `None`."""
    for adapter in _ADAPTERS:
        if adapter.supports(optimizer):
            return adapter
    return None


def resolve_parameters(
    pl_module: Any, optimizer: torch.optim.Optimizer, selector: Any
) -> list[torch.nn.Parameter]:
    """Resolves a REST-supplied selector to concrete parameters inside `optimizer`.

    Args:
        pl_module: The composed `LightningModule`, used to map parameters to names.
        optimizer: The target optimizer; only its own `param_groups` are eligible.
        selector: `"all"`, `{"parameter": name}`, `{"parameter_prefix": prefix}`, or
            `{"param_group": index}`.

    Raises:
        ValueError: `selector` has an unknown shape, or matches nothing.
    """
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    if selector == "all":
        return all_params
    if isinstance(selector, dict):
        if "param_group" in selector:
            index = int(selector["param_group"])
            if not 0 <= index < len(optimizer.param_groups):
                raise ValueError(f"optimizer has no param_group {index}")
            return list(optimizer.param_groups[index]["params"])
        names_by_id = {id(p): name for name, p in pl_module.named_parameters()}
        if "parameter" in selector:
            name = selector["parameter"]
            matches = [p for p in all_params if names_by_id.get(id(p)) == name]
            if not matches:
                raise ValueError(f"no optimizer parameter named {name!r}")
            return matches
        if "parameter_prefix" in selector:
            prefix = selector["parameter_prefix"]
            matches = [
                p for p in all_params if names_by_id.get(id(p), "").startswith(prefix)
            ]
            if not matches:
                raise ValueError(f"no optimizer parameter matching prefix {prefix!r}")
            return matches
    raise ValueError(f"invalid momentum-reset selector: {selector!r}")
