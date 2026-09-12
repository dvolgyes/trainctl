"""Architecture and autograd graph capture via torchview/torchviz.

Both are optional dependencies (`trainctl[graphs]`) and both render through the
Graphviz `dot` binary, not just the pip package -- `available()` checks both layers.
Rendering writes to a real file path (`graphviz.Digraph.render`), so both captures use
the tempdir-then-read-bytes technique to hand the result to `ArtifactStore` unchanged.

`torchview_dot` is the exception: `graphviz.Digraph.source` is built in pure Python and
never invokes the `dot` binary, so it needs only the `torchview` package. It backs the
live FUSE graph file (see `trainctl.fuse.filesystem`), which favors that over rendering.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch


def _python_available(module_name: str) -> bool:
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


def torchview_available() -> bool:
    """Returns whether torchview and the `dot` binary are both present."""
    return _python_available("torchview") and shutil.which("dot") is not None


def torchviz_available() -> bool:
    """Returns whether torchviz and the `dot` binary are both present."""
    return _python_available("torchviz") and shutil.which("dot") is not None


def _render(visual_graph: Any) -> dict[str, str | bytes]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        visual_graph.render(
            filename="graph", directory=tmp_dir, format="svg", cleanup=True
        )
        svg_bytes = (Path(tmp_dir) / "graph.svg").read_bytes()
    return {"graph.svg": svg_bytes, "graph.dot": visual_graph.source}


def torchview_capture(model: Any, batch: Any) -> dict[str, str | bytes]:
    """Renders the model's architecture graph traced against `batch`.

    Raises:
        ImportError: torchview or the `dot` binary is unavailable.
    """
    import torchview

    if shutil.which("dot") is None:
        raise ImportError("torchview_capture requires the Graphviz 'dot' binary")
    # torchview defaults to cuda whenever it's available and calls model.to(device)
    # on the *actual* model (nn.Module.to is in-place) -- pin it to the model's real
    # device so a live capture can't silently relocate the model mid-training.
    graph = torchview.draw_graph(
        model, input_data=batch, device=next(model.parameters()).device
    )
    return _render(graph.visual_graph)


def torchview_dot(model: Any, batch: Any) -> str:
    """Returns the DOT source for the model's architecture graph traced against `batch`.

    Never renders, so unlike `torchview_capture` this needs only the `torchview`
    package, not the `dot` binary.

    Raises:
        ImportError: torchview is not installed.
    """
    import torchview

    return torchview.draw_graph(
        model, input_data=batch, device=next(model.parameters()).device
    ).visual_graph.source


def torchviz_capture(
    model: Any, output: Any, show_attrs: bool = True, show_saved: bool = True
) -> dict[str, str | bytes]:
    """Renders the autograd graph rooted at `output`.

    `output` must not already have been backwarded with `retain_graph=False` (the
    default) -- its saved tensors would already be freed. Use `torchviz_capture_live`
    for a live diagnostic capture during ordinary training, where the most recent
    `current_step.loss` is always already-backwarded by Lightning itself; this
    function stays useful as-is for the backward-exception breakpoint, where
    `current_step.loss` is the loss whose `backward()` call raised and is exactly
    the graph state worth inspecting.

    Raises:
        ImportError: torchviz or the `dot` binary is unavailable.
    """
    import torchviz

    if shutil.which("dot") is None:
        raise ImportError("torchviz_capture requires the Graphviz 'dot' binary")
    dot = torchviz.make_dot(
        output,
        params=dict(model.named_parameters()),
        show_attrs=show_attrs,
        show_saved=show_saved,
    )
    return _render(dot)


def torchviz_capture_live(
    model: Any, forward_input: Any, show_attrs: bool = True, show_saved: bool = True
) -> dict[str, str | bytes]:
    """Runs a fresh forward pass and renders its (still-intact) autograd graph.

    Unlike calling `torchviz_capture` on an already-backwarded loss -- which fails
    with "Trying to backward through the graph a second time" once Lightning's own
    backward (`retain_graph=False`) has freed its saved tensors -- this computes a
    new output whose graph has never been backwarded. Runs in eval mode (restored
    afterward) so it doesn't perturb BatchNorm running stats mid-training, matching
    `torchinfo_inspector.rich_summary`'s own guarantee.

    Raises:
        ImportError: torchviz or the `dot` binary is unavailable.
    """
    if shutil.which("dot") is None:
        raise ImportError("torchviz_capture requires the Graphviz 'dot' binary")
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            output = model(forward_input)
    finally:
        model.train(was_training)
    return torchviz_capture(model, output, show_attrs, show_saved)
