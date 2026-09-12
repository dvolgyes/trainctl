"""Transport-agnostic content for Trainctl's read-only virtual filesystem.

Owns the actual "what does this virtual path contain" policy -- both
`trainctl.fuse.filesystem.TrainctlFS` (a thin mfusepy adapter) and
`trainctl.rest.app` (a thin `GET /files/{path}` adapter) read through the plain
functions here (`generate`, `is_virtual_dir`, `list_dir`) rather than each
re-deriving the same `TrainctlRuntime` state independently.
"""

import json
from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from trainctl.runtime import (
    cuda_memory,
    forward_input,
    graph_inspectors,
    inspectors_registry,
    torchinfo_inspector,
)

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime
    from trainctl.runtime.state import RankInfo


def _text(value: str) -> bytes:
    return (value if value.endswith("\n") else value + "\n").encode()


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, default=str) + "\n").encode()


def _bool(value: bool) -> str:
    return "true" if value else "false"


README = """Trainctl live filesystem
========================

This is a read-only live view of a PyTorch Lightning run.

state/
    Current values pushed by TrainctlMixin hooks, including
    state/cuda-memory.json (allocator-history config + cheap counters) and
    state/profiler.json (bounded-profiler phase, idle unless one is armed) and
    state/lightning-profiler.json (which of Lightning's own profilers, if any,
    is currently installed as trainer.profiler -- see model/lightning-profiler-
    summary.txt, and POST /commands kind=set_lightning_profiler to change it).
    These reads do not inspect CUDA tensors or synchronously query the Trainer.

meta/
    Process IDs, selected ports and runtime metadata.

model/
    Static and cheap model info: summary.txt/summary.json (torchinfo when
    installed, else a plain str(model) fallback), inspectors.json
    (torchinfo/torchview/torchviz/torchlens availability), and
    tunable-hparams.json (current value/type/dtype/device of every hparam a
    LightningModule opted into live editing via `trainctl_tunable_hparams` --
    see trainctl.runtime.hparam_registry). summary-live.txt and graph.dot are
    lazy: reading them blocks until the training thread reaches its next
    optimizer step, then returns a fresh torchinfo summary / torchview DOT
    source traced against the in-flight batch -- nothing is persisted, so
    incidental traversal (ls, find, indexers) never litters GET /artifacts, and
    graph.dot skips rendering entirely (no Graphviz `dot` binary needed).
    Rendered SVGs (torchview_capture, torchviz_capture) and TorchLens traces
    (torchlens_capture) stay deliberate, persisted POST /commands captures,
    listed under GET /artifacts -- there is no live "latest" pointer for those.

debug/pytorch/
    Status of the (experimental, best-effort) PyTorch distributed debug HTTP
    server. See debug/pytorch/README.md.

CUDA memory snapshots/history recording are plain REST reads (GET
/debug/cuda-memory/snapshot, GET /debug/cuda-memory/history) since they don't
touch the Trainer/model; torchinfo/torchview/torchviz/torchlens captures and
bounded profiling go through POST /commands since they need the in-flight
batch/loss or training-thread-affine state; live-tunable hparams are edited
through POST /commands (kind=set_hparam) for the same reason.

This entire tree is also reachable read-only over REST at GET /files/<path>
(e.g. GET /files/model/summary.txt), for callers that can't mount FUSE (a
browser, for instance). Checkpoints, snapshots and debug captures produced by
REST are real files on the regular filesystem, one level up from this mount
(see meta/artifacts-path), and are reachable over REST at
/artifact-files/<relative path> (the relative path is given as `files_url` in
GET /artifacts/{id}'s response).

Kernel file-content caching is disabled with FUSE direct I/O. Trainctl may
cache internally where it has explicit consistency semantics.

Use REST for pause/continue/checkpoint/snapshot/control operations.
"""

DEBUG_README = """Trainctl PyTorch distributed debug projection
==============================================

status reports whether torch.distributed.debug's experimental HTTP frontend is
running for this run. Proxying its cheap/expensive/profiler endpoints through
this filesystem is not implemented yet; use its HTTP frontend directly via the
URL in status until then.
"""

_STATIC_FILES: dict[str, Callable[["TrainctlRuntime"], bytes]] = {
    "/README.md": lambda rt: _text(README),
    "/meta/run-id": lambda rt: _text(rt.run_id),
    "/meta/started-at": lambda rt: _text(str(rt.started_at)),
    "/meta/versions.json": lambda rt: _json(
        {"lightning_backend": rt.backend.name, "lightning_version": rt.backend.version}
    ),
    "/meta/lightning/backend": lambda rt: _text(rt.backend.name),
    "/meta/lightning/version": lambda rt: _text(rt.backend.version),
    "/meta/rest/enabled": lambda rt: _text(_bool(rt.config.rest_enabled)),
    "/meta/rest/host": lambda rt: _text(rt.config.rest_host if rt.rest_url else ""),
    "/meta/rest/port": lambda rt: _text(str(rt._rest_port) if rt._rest_port else ""),
    "/meta/rest/url": lambda rt: _text(rt.rest_url or ""),
    "/meta/pytorch-debug/enabled": lambda rt: _text(
        _bool(rt.config.torch_debug_enabled)
    ),
    "/meta/pytorch-debug/port": lambda rt: _text(
        str(rt.torch_debug._selected_port or "")
    ),
    "/meta/pytorch-debug/url": lambda rt: _text(rt.torch_debug.frontend_url or ""),
    "/meta/artifacts-path": lambda rt: _text(
        str(rt.artifacts.root) if rt.artifacts else ""
    ),
    "/meta/optimizer-surgery-enabled": lambda rt: _text(
        _bool(rt.config.optimizer_surgery_enabled)
    ),
    "/state/snapshot.json": lambda rt: _json(asdict(rt.state.read())),
    "/state/status": lambda rt: _text(rt.state.read().status),
    "/state/stage": lambda rt: _text(rt.state.read().stage or ""),
    "/state/phase": lambda rt: _text(rt.state.read().phase),
    "/state/epoch": lambda rt: _text(str(rt.state.read().epoch)),
    "/state/global-step": lambda rt: _text(str(rt.state.read().global_step)),
    "/state/batch-idx": lambda rt: _text(
        str(rt.state.read().batch_idx) if rt.state.read().batch_idx is not None else ""
    ),
    "/state/generation": lambda rt: _text(str(rt.state.read().generation)),
    "/state/metrics/index.json": lambda rt: _json(
        {key: asdict(metric) for key, metric in rt.state.read().metrics.items()}
    ),
    "/state/control/hold-state": lambda rt: _text(rt.state.read().hold_state),
    "/state/control/pending-command-count": lambda rt: _text(
        str(rt.state.read().pending_commands)
    ),
    "/state/breakpoint.json": lambda rt: _json(
        asdict(rt.state.read().exception) if rt.state.read().exception else None
    ),
    "/state/dataloader/finite-check.json": lambda rt: _json(
        {
            "enabled": rt.finite_guard.enabled,
            "checked_batches": rt.finite_guard.checked_batches,
            "failures": rt.finite_guard.failures,
        }
    ),
    "/state/pipeline/config.json": lambda rt: _json(
        {
            "enabled": rt.config.pipeline_pressure_enabled,
            "window": rt.config.pipeline_pressure_window,
        }
    ),
    "/state/pipeline/latest.json": lambda rt: _json(
        asdict(rt.pipeline.latest()) if rt.pipeline.latest() else None
    ),
    "/state/pipeline/summary.json": lambda rt: _json(rt.pipeline.summary()),
    "/model/class": lambda rt: _text(rt._model_class_name or ""),
    "/model/hparams.json": lambda rt: _json(rt._model_hparams),
    "/model/summary.txt": lambda rt: _text(rt._model_summary or ""),
    "/model/summary.json": lambda rt: _json(rt._model_summary_json),
    "/model/inspectors.json": lambda rt: _json(
        inspectors_registry.availability_snapshot()
    ),
    "/model/tunable-hparams.json": lambda rt: _json(
        {name: asdict(value) for name, value in rt.tunable_hparams.items()}
    ),
    "/state/cuda-memory.json": lambda rt: _json(
        {**asdict(rt.cuda_memory_history), **cuda_memory.cheap_stats()}
    ),
    "/state/profiler.json": lambda rt: _json(
        rt.profiler_run.phase_state() if rt.profiler_run else {"state": "idle"}
    ),
    "/state/lightning-profiler.json": lambda rt: _json(
        {"level": rt.lightning_profiler_level}
    ),
    "/model/lightning-profiler-summary.txt": lambda rt: _text(
        rt.lightning_profiler.summary()
        if rt.lightning_profiler
        else "lightning profiler not enabled"
    ),
    "/debug/README.md": lambda rt: _text(DEBUG_README),
    "/debug/pytorch/README.md": lambda rt: _text(DEBUG_README),
    "/debug/pytorch/status": lambda rt: _json(
        {"enabled": rt.torch_debug._started, "url": rt.torch_debug.frontend_url}
    ),
}

_LIVE_CAPTURE_TIMEOUT_S = 30.0


def _require_in_flight_batch(rt: "TrainctlRuntime", what: str) -> Any:
    if rt.current_step.batch is None:
        raise ValueError(f"{what} requires an in-flight batch")
    return rt.current_step.batch


def _live_model_summary(rt: "TrainctlRuntime") -> bytes:
    def compute(pl_module: Any) -> str:
        batch = _require_in_flight_batch(rt, "model/summary-live.txt")
        target = forward_input.resolve(pl_module, batch)
        return torchinfo_inspector.rich_summary(pl_module, target)["summary.txt"]

    return _text(rt.live_captures.request(compute, timeout=_LIVE_CAPTURE_TIMEOUT_S))


def _live_model_graph(rt: "TrainctlRuntime") -> bytes:
    def compute(pl_module: Any) -> str:
        batch = _require_in_flight_batch(rt, "model/graph.dot")
        target = forward_input.resolve(pl_module, batch)
        return graph_inspectors.torchview_dot(pl_module, target)

    return _text(rt.live_captures.request(compute, timeout=_LIVE_CAPTURE_TIMEOUT_S))


_LIVE_FILES: dict[str, Callable[["TrainctlRuntime"], bytes]] = {
    "/model/summary-live.txt": _live_model_summary,
    "/model/graph.dot": _live_model_graph,
}

_RANK_FIELDS = ("pid", "hostname", "global-rank", "local-rank", "device")
_WORKER_FIELDS = ("pid",)


def _index_at(parts: list[str], pos: int, count: int) -> int | None:
    """Parses `parts[pos]` as an integer index below `count`, or returns None."""
    if pos >= len(parts):
        return None
    try:
        index = int(parts[pos])
    except ValueError:
        return None
    return index if 0 <= index < count else None


def _rank_field_values(rank: "RankInfo") -> dict[str, str]:
    return {
        "pid": str(rank.pid),
        "hostname": rank.hostname,
        "global-rank": str(rank.global_rank),
        "local-rank": str(rank.local_rank),
        "device": rank.device,
    }


def _resolve_rank(
    parts: list[str], ranks: tuple["RankInfo", ...]
) -> "RankInfo | None":
    index = _index_at(parts, 3, len(ranks))
    return ranks[index] if index is not None else None


def rank_file(runtime: "TrainctlRuntime", path: str) -> bytes | None:
    parts = path.split("/")
    if (
        len(parts) != 5
        or parts[1] != "meta"
        or parts[2] != "ranks"
        or parts[4] not in _RANK_FIELDS
    ):
        return None
    rank = _resolve_rank(parts, runtime.state.read().ranks)
    if rank is None:
        return None
    return _text(_rank_field_values(rank)[parts[4]])


def worker_file(runtime: "TrainctlRuntime", path: str) -> bytes | None:
    parts = path.split("/")
    if (
        len(parts) != 7
        or parts[1] != "meta"
        or parts[2] != "ranks"
        or parts[4] != "workers"
        or parts[6] not in _WORKER_FIELDS
    ):
        return None
    rank = _resolve_rank(parts, runtime.state.read().ranks)
    if rank is None:
        return None
    widx = _index_at(parts, 5, len(rank.worker_pids))
    if widx is None:
        return None
    return _text(str(rank.worker_pids[widx]))


def is_live_file(path: str) -> bool:
    """Returns whether `path` is one of the lazy, safe-point-blocking live files."""
    return path in _LIVE_FILES


def generate(runtime: "TrainctlRuntime", path: str) -> bytes | None:
    """Returns `path`'s current content, or `None` if `path` names no virtual file.

    Raises:
        TimeoutError: `path` is a live file and no safe point was reached in time.
    """
    generator = _STATIC_FILES.get(path)
    if generator is not None:
        return generator(runtime)
    live_generator = _LIVE_FILES.get(path)
    if live_generator is not None:
        return live_generator(runtime)
    rank_content = rank_file(runtime, path)
    if rank_content is not None:
        return rank_content
    return worker_file(runtime, path)


def is_virtual_dir(runtime: "TrainctlRuntime", path: str) -> bool:
    """Returns whether `path` names a virtual directory (not a file) in this tree."""
    if path == "/":
        return True
    if path == "/meta/ranks":
        return True
    prefix = path.rstrip("/") + "/"
    if any(
        candidate.startswith(prefix) for candidate in (*_STATIC_FILES, *_LIVE_FILES)
    ):
        return True
    ranks = runtime.state.read().ranks
    parts = path.split("/")
    if len(parts) == 4 and parts[1] == "meta" and parts[2] == "ranks":
        return _index_at(parts, 3, len(ranks)) is not None
    if (
        len(parts) == 5
        and parts[1] == "meta"
        and parts[2] == "ranks"
        and parts[4] == "workers"
    ):
        return _index_at(parts, 3, len(ranks)) is not None
    if (
        len(parts) == 6
        and parts[1] == "meta"
        and parts[2] == "ranks"
        and parts[4] == "workers"
    ):
        rank = _resolve_rank(parts, ranks)
        if rank is None:
            return False
        return _index_at(parts, 5, len(rank.worker_pids)) is not None
    return False


def list_dir(runtime: "TrainctlRuntime", path: str) -> list[str]:
    """Returns `path`'s child names, sorted, without `.`/`..` (a FUSE-only convention)."""
    if path == "/meta/ranks":
        count = len(runtime.state.read().ranks)
        return [f"{i:03d}" for i in range(count)]

    parts = path.split("/")
    ranks = runtime.state.read().ranks
    if len(parts) == 4 and parts[1] == "meta" and parts[2] == "ranks":
        return [*_RANK_FIELDS, "workers"]
    if (
        len(parts) == 5
        and parts[1] == "meta"
        and parts[2] == "ranks"
        and parts[4] == "workers"
    ):
        rank = _resolve_rank(parts, ranks)
        count = len(rank.worker_pids) if rank is not None else 0
        return [f"{i:03d}" for i in range(count)]
    if (
        len(parts) == 6
        and parts[1] == "meta"
        and parts[2] == "ranks"
        and parts[4] == "workers"
    ):
        return [*_WORKER_FIELDS]

    prefix = path if path.endswith("/") else path + "/"
    names: set[str] = set()
    for candidate in (*_STATIC_FILES, *_LIVE_FILES, "/meta/ranks"):
        if not candidate.startswith(prefix):
            continue
        remainder = candidate[len(prefix) :]
        name = remainder.split("/", 1)[0]
        if name:
            names.add(name)
    return sorted(names)
