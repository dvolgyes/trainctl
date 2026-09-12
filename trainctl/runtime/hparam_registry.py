"""Generic named tunable-hyperparameter registry: resolve, describe, and type-preserving cast.

A LightningModule opts knobs in via a class attribute `trainctl_tunable_hparams`,
in either form:

  * `list[str]` -- bare dotted paths, no bounds/label; type is auto-detected from
    the current value at read time.
  * `dict[str, dict]` -- path -> metadata dict with optional `min`/`max`/`label`.
    The dict key IS the resolution path (same role the bare string plays in the
    list form); a display name is `label`, not a second key scheme.

A path segment may be an attribute or a dict key (`get_field`/`set_field`
resolve either), so a path can walk into a nested dict-of-dicts, not just plain
attribute chains. Declarations are merged across `type(pl_module).__mro__` via
`collect_tunable_hparams`, so a mixin and a subclass can each contribute their
own fields without one silently shadowing the other's whole declaration; a
more-derived class re-declaring the same path replaces that path's spec
wholesale. Everything not listed stays exactly as read-only as
`TrainctlRuntime._model_hparams` already makes it.
"""

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import torch

_MAX_INLINE_TENSOR_ELEMENTS = 4096


@dataclass(frozen=True)
class HparamValue:
    """A tunable hparam's current value, its type, and (for tensors) dtype/device/shape.

    Attributes:
        value: The current value, or `None` for a tensor above
            `_MAX_INLINE_TENSOR_ELEMENTS` elements.
        type: One of `"bool"`, `"int"`, `"float"`, `"str"`, `"tensor"`.
        dtype: `str(tensor.dtype)`, tensors only.
        device: `str(tensor.device)`, tensors only.
        shape: `list(tensor.shape)`, tensors only.
        min: The field's declared lower bound, if any -- spec metadata, not
            runtime-introspected.
        max: The field's declared upper bound, if any.
        label: The field's declared display name, if any.
    """

    value: Any
    type: str
    dtype: str | None = None
    device: str | None = None
    shape: list[int] | None = None
    min: int | float | None = None
    max: int | float | None = None
    label: str | None = None


@dataclass(frozen=True)
class TunableHparamSpec:
    """One collected+normalized `trainctl_tunable_hparams` entry.

    Attributes:
        path: Dotted resolution path -- identical to this spec's key in
            `collect_tunable_hparams`'s return dict.
        min: Optional inclusive lower bound `validate_bounds` enforces. `None` is
            unbounded below; always `None` for a bare `list[str]`-form entry.
        max: Optional inclusive upper bound, same semantics.
        label: Optional human-readable display name; `None` falls back to `path`.
    """

    path: str
    min: int | float | None = None
    max: int | float | None = None
    label: str | None = None


def collect_tunable_hparams(model_cls: type) -> dict[str, TunableHparamSpec]:
    """Merges `trainctl_tunable_hparams` across `model_cls.__mro__` into one spec dict.

    Walks the MRO ancestors-first (`reversed(model_cls.__mro__)`), reading only
    each class's own directly-declared attribute (`base.__dict__`, never plain
    `getattr`, so a value already visible through inheritance isn't reprocessed
    at every MRO level it's visible from). A more-derived class's entry for a
    given path fully replaces a less-derived one's -- whole-spec replacement,
    matching ordinary attribute-shadowing semantics, not a per-key metadata
    merge. Each class may use either accepted form, independently of the others.

    Raises:
        TypeError: a class's `trainctl_tunable_hparams` is a bare `str` (missing
            the enclosing list brackets -- iterating it would otherwise silently
            register one bogus single-character path per letter), or a dict
            form's per-path metadata value is not itself a `dict`.
    """
    result: dict[str, TunableHparamSpec] = {}
    for base in reversed(model_cls.__mro__):
        declared = base.__dict__.get("trainctl_tunable_hparams")
        if declared is None:
            continue
        if isinstance(declared, str):
            raise TypeError(
                f"{base.__name__}.trainctl_tunable_hparams must be a list[str] or "
                f"dict[str, dict], got a bare str {declared!r} "
                "(missing the enclosing list brackets?)"
            )
        if isinstance(declared, dict):
            for path, meta in declared.items():
                if not isinstance(meta, dict):
                    raise TypeError(
                        f"{base.__name__}.trainctl_tunable_hparams[{path!r}] must "
                        f"be a dict, got {type(meta).__name__}"
                    )
                result[path] = TunableHparamSpec(
                    path=path,
                    min=meta.get("min"),
                    max=meta.get("max"),
                    label=meta.get("label"),
                )
        else:
            for path in declared:
                result[path] = TunableHparamSpec(path=path)
    return result


def get_field(obj: Any, key: str) -> Any:
    """Reads `obj.<key>`, falling back to `obj[key]` when `obj` lacks that
    attribute but is a `Mapping` (e.g. a dict segment of a config tree).
    Attribute access is tried first, so a plain attribute path resolves exactly
    as it always has.

    Raises:
        AttributeError: `key` is neither an attribute of `obj` nor, for a
            `Mapping` `obj`, one of its keys.
    """
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, Mapping) and key in obj:
        return obj[key]
    raise AttributeError(f"{key!r} is not an attribute or dict key of {obj!r}")


def set_field(obj: Any, key: str, value: Any) -> None:
    """Writes `obj.<key> = value`, falling back to `obj[key] = value` for a
    `MutableMapping` `obj` lacking that attribute. Mirrors `get_field`'s
    attribute-first precedence.

    Raises:
        AttributeError: `key` is neither a settable attribute of `obj` nor, for a
            `MutableMapping` `obj`, an assignable key.
    """
    if hasattr(obj, key):
        setattr(obj, key, value)
    elif isinstance(obj, MutableMapping):
        obj[key] = value
    else:
        raise AttributeError(f"{key!r} is not an attribute or dict key of {obj!r}")


def resolve(pl_module: Any, trainer: Any, path: str) -> tuple[Any, str]:
    """Walks a dotted path to its parent object and leaf name.

    A `"trainer."` prefix roots the walk at `trainer`; otherwise it roots at
    `pl_module` (e.g. `"hparams.augmentation_strength"` reads
    `pl_module.hparams.augmentation_strength`). Each intermediate segment may be
    an attribute or a dict key (see `get_field`), so a path may walk through a
    nested dict-of-dicts as well as plain attribute chains.
    """
    if path.startswith("trainer."):
        obj: Any = trainer
        parts = path.removeprefix("trainer.").split(".")
    else:
        obj = pl_module
        parts = path.split(".")
    for part in parts[:-1]:
        obj = get_field(obj, part)
    return obj, parts[-1]


def describe(value: Any) -> HparamValue:
    """Reports a tunable hparam's current value, type, and (for tensors) dtype/device/shape.

    Raises:
        ValueError: `value`'s type is not one of bool/int/float/str/`torch.Tensor`.
    """
    if isinstance(value, torch.Tensor):
        inline_value = (
            value.detach().cpu().tolist()
            if value.numel() <= _MAX_INLINE_TENSOR_ELEMENTS
            else None
        )
        return HparamValue(
            value=inline_value,
            type="tensor",
            dtype=str(value.dtype),
            device=str(value.device),
            shape=list(value.shape),
        )
    if isinstance(value, bool):  # must precede the int check: bool is an int subclass
        return HparamValue(value=value, type="bool")
    if isinstance(value, int):
        return HparamValue(value=value, type="int")
    if isinstance(value, float):
        return HparamValue(value=value, type="float")
    if isinstance(value, str):
        return HparamValue(value=value, type="str")
    raise ValueError(f"unsupported tunable hparam type: {type(value).__name__}")


def cast_and_assign(parent: Any, leaf: str, new_value: Any) -> HparamValue:
    """Casts `new_value` to `parent.<leaf>`'s current type and assigns it in place.

    A tensor knob is matched to the current tensor's dtype *and* device via
    `torch.as_tensor`, then copied in place with `copy_` to preserve the tensor's
    identity (any optimizer/autograd references to it stay valid) -- this holds
    regardless of whether `leaf` is a plain attribute or a dict key (see
    `get_field`/`set_field`), since `copy_` mutates the tensor object itself.

    Raises:
        ValueError: the current value's type is unsupported, or a replacement
            tensor's shape does not match the current tensor's shape.
    """
    current = get_field(parent, leaf)
    if isinstance(current, torch.Tensor):
        tensor = torch.as_tensor(new_value, dtype=current.dtype, device=current.device)
        if tuple(tensor.shape) != tuple(current.shape):
            raise ValueError(
                f"shape mismatch: {leaf!r} has shape {tuple(current.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        current.copy_(tensor)
        return describe(current)
    if isinstance(current, bool):  # must precede the int check: bool is an int subclass
        set_field(parent, leaf, bool(new_value))
    elif isinstance(current, int):
        set_field(parent, leaf, int(new_value))
    elif isinstance(current, float):
        set_field(parent, leaf, float(new_value))
    elif isinstance(current, str):
        set_field(parent, leaf, str(new_value))
    else:
        raise ValueError(f"unsupported tunable hparam type: {type(current).__name__}")
    return describe(get_field(parent, leaf))


def validate_bounds(spec: TunableHparamSpec, value: Any) -> None:
    """Raises ValueError if `value` violates `spec.min`/`spec.max`.

    A no-op when neither bound is declared (every bare `list[str]`-form entry,
    and any dict-form entry that simply omits `min`/`max`).

    A tensor-shaped `value` (a `torch.Tensor`, or a nested list/tuple -- the JSON
    encoding of one) is checked element-wise via `torch.as_tensor`: every
    element must satisfy both bounds, not just an aggregate. A rank-0 tensor
    (the common case for a single-value tunable tensor) degenerates to the same
    comparison as a plain scalar automatically. A plain scalar `value` is
    compared directly.
    """
    if spec.min is None and spec.max is None:
        return
    if isinstance(value, torch.Tensor | list | tuple):
        tensor = torch.as_tensor(value)
        if spec.min is not None and bool((tensor < spec.min).any()):
            raise ValueError(
                f"{spec.path!r}: all elements must be >= {spec.min}, "
                f"got min {tensor.min().item()!r}"
            )
        if spec.max is not None and bool((tensor > spec.max).any()):
            raise ValueError(
                f"{spec.path!r}: all elements must be <= {spec.max}, "
                f"got max {tensor.max().item()!r}"
            )
        return
    if spec.min is not None and value < spec.min:
        raise ValueError(f"{spec.path!r} must be >= {spec.min}, got {value!r}")
    if spec.max is not None and value > spec.max:
        raise ValueError(f"{spec.path!r} must be <= {spec.max}, got {value!r}")
