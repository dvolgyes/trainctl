"""Derives a LightningModule's `forward()` input from an arbitrary training batch.

`forward`'s signature is model-specific and need not match the batch shape
`training_step` receives -- e.g. an `(images, labels)` pair for ordinary supervised
learning, which is not what `forward` itself accepts. Every live forward pass Trainctl
runs on a model's behalf (torchinfo/torchview/torchviz/torchlens captures, and the
live model-summary/graph FUSE/REST views) resolves the raw batch through here first,
so the mismatch is fixed once rather than at each call site.
"""

from typing import Any


def resolve(pl_module: Any, batch: Any) -> Any:
    """Returns what to pass to `pl_module.forward`, given a raw training batch.

    Calls `pl_module.trainctl_forward_input(batch)` when the model defines it --
    needed whenever `forward`'s signature doesn't match the raw batch shape (a dict
    batch, more than one input tensor, extra collate-added fields, ...). Otherwise
    falls back to `batch[0]` for a list/tuple batch of two or more elements (the
    common `(input, target)` supervised-learning convention), else the batch as-is.
    """
    hook = getattr(pl_module, "trainctl_forward_input", None)
    if hook is not None:
        return hook(batch)
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0]
    return batch
