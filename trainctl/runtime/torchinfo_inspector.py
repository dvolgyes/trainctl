"""Optional torchinfo-backed model summaries.

`torchinfo` is an optional dependency (`trainctl[torchinfo]`); every entry point here
degrades gracefully when it isn't installed. `torchinfo.summary` runs the model in eval
mode internally and restores the original mode afterward, so `rich_summary`'s forward
pass does not perturb BatchNorm running statistics or leak into the live autograd graph.
"""

from typing import Any


def available() -> bool:
    """Returns whether `torchinfo` can be imported."""
    try:
        import torchinfo  # noqa: F401
    except ImportError:
        return False
    return True


def basic_summary_text(pl_module: Any) -> str:
    """A static, no-input module/param-tree summary; falls back to `str(pl_module)`.

    Never raises: this backs the always-computed `/model/summary.txt`, so an
    unavailable or failing torchinfo must not change existing behavior.
    """
    try:
        import torchinfo
    except ImportError:
        return str(pl_module)
    try:
        return str(torchinfo.summary(pl_module, verbose=0))
    except Exception:  # noqa: BLE001 -- best-effort cosmetic summary, never fatal
        return str(pl_module)


def basic_summary_json(pl_module: Any) -> dict[str, Any]:
    """Always-available parameter counts, independent of torchinfo's presence."""
    total = sum(p.numel() for p in pl_module.parameters())
    trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
        "torchinfo_used": available(),
    }


def rich_summary(pl_module: Any, batch: Any) -> dict[str, str]:
    """A forward-traced summary using the live in-flight `batch` as input.

    Args:
        pl_module: The composed `LightningModule`.
        batch: The batch to trace shapes with, e.g. `runtime.current_step.batch`.

    Returns:
        `{"summary.txt": <text>, "summary.json": <json>}`.

    Raises:
        ImportError: torchinfo is not installed.
    """
    import json

    import torchinfo

    stats = torchinfo.summary(pl_module, input_data=batch, verbose=0)
    payload = {
        "total_params": stats.total_params,
        "trainable_params": stats.trainable_params,
        "total_mult_adds": stats.total_mult_adds,
    }
    return {"summary.txt": str(stats), "summary.json": json.dumps(payload, indent=2)}
