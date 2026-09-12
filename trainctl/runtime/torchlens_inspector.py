"""Single controlled TorchLens forward-trace capture.

TorchLens is an optional dependency (`trainctl[torchlens]`). This wraps only its
single-shot `trace()` entry point (metadata-only, `layers_to_save="none"` -- no raw
activation tensors are retained) plus the resulting `Trace`'s own `summary()`/`draw()`;
no tracing logic is reimplemented here. Continuous "arm next batch" capture is not
supported -- see the implementation plan for why.
"""

import tempfile
from pathlib import Path
from typing import Any


def available() -> bool:
    """Returns whether `torchlens` can be imported."""
    try:
        import torchlens  # noqa: F401
    except ImportError:
        return False
    return True


def capture(model: Any, batch: Any) -> dict[str, str | bytes]:
    """Traces one forward pass and returns a summary and a rendered graph.

    Args:
        model: The composed `LightningModule`.
        batch: Positional input for `model.forward()`, e.g.
            `runtime.current_step.batch`. Must already match `forward`'s own
            signature -- a batch shaped for `training_step` (e.g. an `(x, y)` pair)
            may need unpacking by the caller first.

    Returns:
        `{"summary.txt": <overview text>, "graph.svg": <rendered graph bytes>}`.

    Raises:
        ImportError: torchlens is not installed.
    """
    import torchlens as tl

    # tl.trace() monkeypatches torch functions at the process level (not per-model)
    # to intercept every tensor-producing op, and by its own documented default
    # leaves them wrapped after returning -- corrupting every later real forward
    # pass in the process, not just this model's. unwrap_when_done=True is what
    # actually restores the original torch callables; a per-model copy wouldn't
    # help here since the wrapping isn't model-scoped.
    was_training = model.training
    model.eval()
    try:
        trace = tl.trace(model, batch, layers_to_save="none", unwrap_when_done=True)
    finally:
        model.train(was_training)
    summary_text = trace.summary(level="overview")
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "graph"
        trace.draw(vis_outpath=str(out_path), vis_fileformat="svg", vis_save_only=True)
        graph_bytes = out_path.with_suffix(".svg").read_bytes()
    return {"summary.txt": summary_text, "graph.svg": graph_bytes}
