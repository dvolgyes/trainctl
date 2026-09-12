"""Bounded, non-evaluating description of Lightning callback arguments.

Builds the light hook's `params.json` payload (schema/identity fields plus a bounded
description of the callback's own arguments). A tensor argument is described --
shape, dtype, device -- never evaluated: no `.item()`, `.tolist()`, or `.cpu()` on
this path. Increment I3 reuses `describe_arguments` for the heavy manifest's
non-tensor structure description, alongside the separate export of the tensors
themselves.
"""

import json
import os
import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy
import torch

_SCHEMA_VERSION = 1
_MAX_STRING_LEN = 200

_HEAVY_SCHEMA_VERSION = 1

_NUMPY_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}
_BFLOAT16_EXPORT_DTYPE = "float32"


@dataclass(frozen=True)
class OccurrenceContext:
    """Identity shared by every script invoked for one Lightning callback firing.

    Attributes:
        session_id: This run's hooks session id (see `manager.bootstrap_hooks_tree`).
        hook: The exact `Callback` method name, e.g. `on_train_batch_start`.
        stage: `trainer.state.stage` as a string, or `None` if unavailable.
        epoch: `trainer.current_epoch`, or `None` if unavailable.
        global_step: `trainer.global_step`, or `None` if unavailable.
        batch_idx: The batch index Lightning passed, or `None` for non-batch hooks.
        dataloader_idx: The dataloader index Lightning passed, or `None` otherwise.
        rank: `trainer.global_rank`.
        world_size: `trainer.world_size`.
    """

    session_id: str
    hook: str
    stage: str | None
    epoch: int | None
    global_step: int | None
    batch_idx: int | None
    dataloader_idx: int | None
    rank: int
    world_size: int


@dataclass
class _TraversalState:
    """Mutable bookkeeping threaded through one `describe_arguments` call."""

    max_items: int
    max_depth: int
    seen: set[int] = field(default_factory=set)
    truncated: bool = False
    reasons: set[str] = field(default_factory=set)


def new_id() -> str:
    """Returns a short, collision-resistant identifier for one occurrence or invocation."""
    return secrets.token_hex(8)


def describe_arguments(
    arguments: Mapping[str, Any], *, max_items: int, max_depth: int
) -> tuple[dict[str, Any], bool, str | None]:
    """Describes a callback's named arguments as a bounded, JSON-safe structure.

    Args:
        arguments: The callback's own arguments, by name (e.g. `{"batch": ..., "batch_idx": ...}`).
        max_items: Maximum entries kept per traversed `Mapping`/`list`/`tuple`.
        max_depth: Maximum recursion depth below `arguments` itself.

    Returns:
        A tuple of `(description, truncated, reason)`. `reason` is `None` unless
        `truncated` is True, in which case it names every bound that was hit.
    """
    state = _TraversalState(max_items=max_items, max_depth=max_depth)
    described = {
        str(name): _describe(value, state, depth=0) for name, value in arguments.items()
    }
    reason = "; ".join(sorted(state.reasons)) if state.truncated else None
    return described, state.truncated, reason


def _describe(value: Any, state: _TraversalState, *, depth: int) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": value.requires_grad,
        }
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        if len(value) <= _MAX_STRING_LEN:
            return value
        state.truncated = True
        state.reasons.add("string_length")
        return value[:_MAX_STRING_LEN] + "...(truncated)"
    if depth >= state.max_depth:
        state.truncated = True
        state.reasons.add("max_depth")
        return {"kind": "truncated", "reason": "max_depth"}
    object_id = id(value)
    if object_id in state.seen:
        return {"kind": "cycle"}
    if isinstance(value, Mapping):
        return _describe_items(
            value.items(), state, depth=depth, object_id=object_id, is_mapping=True
        )
    if isinstance(value, list | tuple):
        return _describe_items(
            enumerate(value), state, depth=depth, object_id=object_id, is_mapping=False
        )
    return {"kind": "object", "type": type(value).__qualname__}


def _describe_items(
    items: Any, state: _TraversalState, *, depth: int, object_id: int, is_mapping: bool
) -> Any:
    state.seen.add(object_id)
    kept: list[tuple[Any, Any]] = []
    omitted = 0
    for key, value in items:
        if len(kept) >= state.max_items:
            omitted += 1
            continue
        kept.append((key, value))
    described = [(key, _describe(value, state, depth=depth + 1)) for key, value in kept]
    state.seen.discard(object_id)
    if omitted:
        state.truncated = True
        state.reasons.add("max_items")
    if is_mapping:
        result: dict[str, Any] = {str(key): value for key, value in described}
        if omitted:
            result["__truncated__"] = f"{omitted} more item(s) omitted"
        return result
    result_list: list[Any] = [value for _key, value in described]
    if omitted:
        result_list.append(
            {"kind": "truncated", "reason": f"{omitted} more item(s) omitted"}
        )
    return result_list


def build_light_params(
    *,
    context: OccurrenceContext,
    occurrence_id: str,
    invocation_id: str,
    arguments: Mapping[str, Any],
    max_params_bytes: int,
    max_metadata_items: int,
    max_metadata_depth: int,
) -> tuple[dict[str, Any], bool, str | None]:
    """Assembles the light hook's `params.json` document.

    Enforces `max_params_bytes` on the encoded result: if the described arguments
    would push the document over budget, `arguments` is dropped entirely (not
    partially re-truncated) and the truncation reason records why.

    Returns:
        A tuple of `(params, truncated, reason)`, mirroring `describe_arguments`.
    """
    described, truncated, reason = describe_arguments(
        arguments, max_items=max_metadata_items, max_depth=max_metadata_depth
    )
    params = _assemble(
        context, occurrence_id, invocation_id, described, truncated, reason
    )
    encoded = json.dumps(params).encode("utf-8")
    if len(encoded) <= max_params_bytes:
        return params, truncated, reason
    reason = "max_params_bytes" if reason is None else f"{reason}; max_params_bytes"
    params = _assemble(context, occurrence_id, invocation_id, None, True, reason)
    return params, True, reason


def _assemble(
    context: OccurrenceContext,
    occurrence_id: str,
    invocation_id: str,
    arguments: dict[str, Any] | None,
    truncated: bool,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "session_id": context.session_id,
        "occurrence_id": occurrence_id,
        "invocation_id": invocation_id,
        "hook": context.hook,
        "modality": "light",
        "timestamp": time.time(),
        "pid": _pid(),
        "rank": context.rank,
        "world_size": context.world_size,
        "stage": context.stage,
        "epoch": context.epoch,
        "global_step": context.global_step,
        "batch_idx": context.batch_idx,
        "dataloader_idx": context.dataloader_idx,
        "arguments": arguments,
        "truncated": truncated,
        "truncated_reason": reason,
    }


def _pid() -> int:
    return os.getpid()


def write_json_file(path: Path, document: dict[str, Any]) -> Path:
    """Writes `document` as JSON to `path` and returns `path`."""
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


@dataclass(frozen=True)
class TensorRecord:
    """One tensor exported for a heavy hook invocation.

    Attributes:
        argument_path: Slash-joined path locating this tensor within the callback's
            named tensor arguments, e.g. `batch/0/image` -- never derived from a
            dictionary key as a filename.
        file: The generated `.npy` filename (`tensor-000001.npy`, ...).
        shape: The tensor's shape.
        dtype: The numpy dtype the array was actually written as.
        original_dtype: The tensor's original torch dtype, as a string, only when it
            differs from `dtype` (currently only `bfloat16` -> `float32`); else `None`.
        device: The tensor's original device, as a string.
    """

    argument_path: str
    file: str
    shape: list[int]
    dtype: str
    original_dtype: str | None
    device: str


@dataclass(frozen=True)
class HeavyCaptureFailure:
    """Why an entire heavy capture was refused before any tensor was copied off-device.

    Attributes:
        reason: A human-readable description naming the offending argument path or
            the exceeded limit.
    """

    reason: str


@dataclass(frozen=True)
class _TensorLeaf:
    path: str
    tensor: torch.Tensor


def _resolve_export_dtype(tensor: torch.Tensor) -> str | None:
    """Returns the numpy dtype name this tensor would be saved as, or `None` if unsupported."""
    if tensor.dtype is torch.bfloat16:
        return _BFLOAT16_EXPORT_DTYPE
    return _NUMPY_DTYPE_NAMES.get(tensor.dtype)


def _is_supported_layout(tensor: torch.Tensor) -> bool:
    if tensor.layout is not torch.strided:
        return False
    if tensor.is_quantized:
        return False
    if tensor.device.type == "meta":
        return False
    return not getattr(tensor, "is_nested", False)


def collect_tensor_leaves(
    arguments: Mapping[str, Any],
    tensor_arg_names: Sequence[str],
    *,
    max_items: int,
    max_depth: int,
) -> tuple[list[_TensorLeaf], bool, str | None]:
    """Finds every tensor nested within this callback's tensor-carrying arguments.

    Only traverses `arguments[name]` for each `name in tensor_arg_names` -- the
    catalogue's static, per-callback list of arguments that may carry tensors.
    Bounded and cycle-safe like `describe_arguments`; hitting a bound stops that
    branch (fewer tensors found), it does not fail the capture.
    """
    state = _TraversalState(max_items=max_items, max_depth=max_depth)
    leaves: list[_TensorLeaf] = []
    for name in tensor_arg_names:
        if name in arguments:
            _collect_leaves(arguments[name], name, state, leaves, depth=0)
    reason = "; ".join(sorted(state.reasons)) if state.truncated else None
    return leaves, state.truncated, reason


def _collect_leaves(
    value: Any, path: str, state: _TraversalState, leaves: list[_TensorLeaf], *, depth: int
) -> None:
    if isinstance(value, torch.Tensor):
        leaves.append(_TensorLeaf(path=path, tensor=value))
        return
    if not isinstance(value, Mapping | list | tuple):
        return
    if depth >= state.max_depth:
        state.truncated = True
        state.reasons.add("max_depth")
        return
    object_id = id(value)
    if object_id in state.seen:
        return
    items = value.items() if isinstance(value, Mapping) else enumerate(value)
    state.seen.add(object_id)
    for index, (key, item) in enumerate(items):
        if index >= state.max_items:
            state.truncated = True
            state.reasons.add("max_items")
            break
        _collect_leaves(item, f"{path}/{key}", state, leaves, depth=depth + 1)
    state.seen.discard(object_id)


def prepare_heavy_export(
    arguments: Mapping[str, Any],
    tensor_arg_names: Sequence[str],
    *,
    max_items: int,
    max_depth: int,
    max_export_bytes: int,
) -> tuple[list[_TensorLeaf], bool, str | None] | HeavyCaptureFailure:
    """Validates every tensor a heavy hook would export, without copying any of them.

    "All or failed": every found tensor must have a supported layout/dtype and the
    predicted total converted size must fit `max_export_bytes`, checked before any
    host allocation. A callback with no tensor arguments (or none present this
    firing) validates successfully with an empty leaf list -- a valid empty export.
    """
    leaves, truncated, reason = collect_tensor_leaves(
        arguments, tensor_arg_names, max_items=max_items, max_depth=max_depth
    )
    predicted_bytes = 0
    for leaf in leaves:
        if not _is_supported_layout(leaf.tensor):
            return HeavyCaptureFailure(
                f"unsupported tensor layout/device at {leaf.path}: "
                f"layout={leaf.tensor.layout}, device={leaf.tensor.device}"
            )
        export_dtype = _resolve_export_dtype(leaf.tensor)
        if export_dtype is None:
            return HeavyCaptureFailure(f"unsupported dtype at {leaf.path}: {leaf.tensor.dtype}")
        predicted_bytes += leaf.tensor.numel() * numpy.dtype(export_dtype).itemsize
    if predicted_bytes > max_export_bytes:
        return HeavyCaptureFailure(
            f"predicted export size {predicted_bytes} bytes exceeds "
            f"hooks_max_export_bytes ({max_export_bytes})"
        )
    return leaves, truncated, reason


def export_tensors(leaves: list[_TensorLeaf], invocation_dir: Path) -> list[TensorRecord]:
    """Detaches, moves to CPU, converts if needed, and writes each tensor as `.npy`.

    Only called after `prepare_heavy_export` has validated every leaf, so this never
    encounters an unsupported layout/dtype -- writes are all-or-nothing by
    construction, not by rollback. `numpy.save(..., allow_pickle=False)`: no Python
    object, ever, is pickled into these files.
    """
    records: list[TensorRecord] = []
    for index, leaf in enumerate(leaves, start=1):
        export_dtype = _resolve_export_dtype(leaf.tensor)
        if export_dtype is None:
            raise AssertionError(f"unvalidated tensor reached export: {leaf.path}")
        cpu_tensor = leaf.tensor.detach().cpu()
        if leaf.tensor.dtype is torch.bfloat16:
            cpu_tensor = cpu_tensor.to(torch.float32)
        array = cpu_tensor.numpy()
        filename = f"tensor-{index:06d}.npy"
        numpy.save(invocation_dir / filename, array, allow_pickle=False)
        records.append(
            TensorRecord(
                argument_path=leaf.path,
                file=filename,
                shape=list(leaf.tensor.shape),
                dtype=export_dtype,
                original_dtype=str(leaf.tensor.dtype) if leaf.tensor.dtype is torch.bfloat16 else None,
                device=str(leaf.tensor.device),
            )
        )
    return records


def build_heavy_manifest(
    *,
    context: OccurrenceContext,
    occurrence_id: str,
    invocation_id: str,
    arguments: Mapping[str, Any],
    tensor_arg_names: Sequence[str],
    tensor_records: list[TensorRecord],
    max_metadata_items: int,
    max_metadata_depth: int,
    truncated: bool,
    truncated_reason: str | None,
) -> dict[str, Any]:
    """Assembles the heavy hook's `manifest.json` document.

    `arguments` describes only the tensor-carrying arguments' non-tensor structure
    (primitives, containers) plus each tensor's shape/dtype/device -- the same
    bounded, non-evaluating description the light path uses. `tensor_records`
    separately names the actual exported `.npy` files; a script correlates the two
    by `argument_path`.
    """
    restricted = {name: arguments[name] for name in tensor_arg_names if name in arguments}
    described, meta_truncated, meta_reason = describe_arguments(
        restricted, max_items=max_metadata_items, max_depth=max_metadata_depth
    )
    all_truncated = truncated or meta_truncated
    reasons = [r for r in (truncated_reason, meta_reason) if r]
    return {
        "schema_version": _HEAVY_SCHEMA_VERSION,
        "session_id": context.session_id,
        "occurrence_id": occurrence_id,
        "invocation_id": invocation_id,
        "hook": context.hook,
        "modality": "heavy",
        "timestamp": time.time(),
        "pid": _pid(),
        "rank": context.rank,
        "world_size": context.world_size,
        "stage": context.stage,
        "epoch": context.epoch,
        "global_step": context.global_step,
        "batch_idx": context.batch_idx,
        "dataloader_idx": context.dataloader_idx,
        "arguments": described,
        "tensor_records": [
            {
                "argument_path": record.argument_path,
                "file": record.file,
                "shape": record.shape,
                "dtype": record.dtype,
                "original_dtype": record.original_dtype,
                "device": record.device,
            }
            for record in tensor_records
        ],
        "truncated": all_truncated,
        "truncated_reason": "; ".join(reasons) if reasons else None,
    }
