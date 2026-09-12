"""Opt-in check of every floating/complex tensor in a training batch for NaN/Inf.

Answers "did the batch delivered to the model already contain non-finite values?" --
not a replacement for PyTorch/Lightning autograd anomaly detection, which diagnoses a
failing *backward*. This checks *inputs*, before the training step runs, so a bad
batch can be handled with the normal cooperative hold rather than the exception-hold
path in `trainctl.runtime.runtime.handle_backward_exception`.

Disabled by default: checking every batch can force a host/device synchronization and
is not free, even in the simple per-tensor form implemented here.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BatchTensorRef:
    """One floating/complex tensor found while traversing a batch.

    Attributes:
        path: Human-readable location within the batch, e.g. `batch[1]["targets"]`.
        tensor: The live tensor. Never copied by the traversal itself.
    """

    path: str
    tensor: torch.Tensor


@dataclass(frozen=True)
class BadTensorInfo:
    """A tensor that failed the finite check.

    Attributes:
        path: Same location convention as `BatchTensorRef.path`.
        shape: Tensor shape.
        dtype: Tensor dtype, as a string.
        device: Tensor device, as a string.
        nan_count: Number of NaN elements.
        posinf_count: Number of `+inf` elements.
        neginf_count: Number of `-inf` elements.
    """

    path: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    nan_count: int
    posinf_count: int
    neginf_count: int


@dataclass(frozen=True)
class FiniteCheckReport:
    """The result of a failed `DataloaderFiniteGuard.check`.

    Attributes:
        bad_tensors: Every tensor in the batch that failed the finite check.
    """

    bad_tensors: tuple[BadTensorInfo, ...]


def iter_batch_tensors(batch: Any, path: str = "batch") -> list[BatchTensorRef]:
    """Traverses `batch`, yielding every reachable floating/complex tensor.

    Supports `Tensor`, `Mapping`, `list`, `tuple` (including namedtuples). Integer and
    boolean tensors are skipped -- non-finite values aren't meaningful for them.
    Unknown container types are skipped rather than traversed.
    """
    refs: list[BatchTensorRef] = []
    if isinstance(batch, torch.Tensor):
        if batch.is_floating_point() or batch.is_complex():
            refs.append(BatchTensorRef(path=path, tensor=batch))
        return refs
    if isinstance(batch, Mapping):
        for key, value in batch.items():
            refs.extend(iter_batch_tensors(value, f"{path}[{key!r}]"))
        return refs
    if isinstance(batch, tuple | list):
        for index, value in enumerate(batch):
            refs.extend(iter_batch_tensors(value, f"{path}[{index}]"))
        return refs
    return refs


class DataloaderFiniteGuard:
    """Checks training batches for non-finite tensors, when enabled.

    Attributes:
        enabled: Whether `check` actually traverses batches. Mutated only from the
            training thread, via the `set_dataloader_finite_check` command.
        checked_batches: Number of batches checked since this guard was created.
        failures: Number of batches that failed the check.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.checked_batches = 0
        self.failures = 0

    def check(self, batch: Any) -> FiniteCheckReport | None:
        """Returns a `FiniteCheckReport` if `batch` contains non-finite values, else `None`.

        No-ops (returns `None` without traversing `batch`) when `enabled` is False.
        """
        if not self.enabled:
            return None
        self.checked_batches += 1
        bad: list[BadTensorInfo] = []
        for ref in iter_batch_tensors(batch):
            if bool(torch.isfinite(ref.tensor).all()):
                continue
            bad.append(
                BadTensorInfo(
                    path=ref.path,
                    shape=tuple(ref.tensor.shape),
                    dtype=str(ref.tensor.dtype),
                    device=str(ref.tensor.device),
                    nan_count=int(torch.isnan(ref.tensor).sum()),
                    posinf_count=int(torch.isposinf(ref.tensor).sum()),
                    neginf_count=int(torch.isneginf(ref.tensor).sum()),
                )
            )
        if not bad:
            return None
        self.failures += 1
        return FiniteCheckReport(bad_tensors=tuple(bad))
