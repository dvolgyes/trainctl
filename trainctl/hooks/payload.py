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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_SCHEMA_VERSION = 1
_MAX_STRING_LEN = 200


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
