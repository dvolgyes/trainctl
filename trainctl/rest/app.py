"""FastAPI app exposing Trainctl's REST control/debug plane on global rank 0.

Every mutating route only enqueues a `Command` or calls `RunGate.release` (safe from any
thread); the Lightning training thread executes commands from
`TrainctlRuntime._run_due_commands`. REST must never touch the Trainer/model directly —
see the module docstring in `trainctl.runtime.runtime`.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

from trainctl.rest.schemas import (
    CheckpointIn,
    CommandIn,
    DebugCaptureIn,
    HoldIn,
    SnapshotIn,
)
from trainctl.runtime import cuda_memory, inspectors_registry, virtual_fs
from trainctl.runtime.commands import Command, parse_when

_STATIC_DIR = Path(__file__).parent / "static"
_MEDIA_TYPES = {".json": "application/json"}

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime

_ROUTE_GROUPS: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...] = (
    (
        "Live state",
        (
            ("GET", "/health", "Liveness check"),
            ("GET", "/run", "Full current run snapshot"),
            ("GET", "/metrics", "Scalar metrics logged so far"),
            ("GET", "/optimizers", "Current learning rates"),
        ),
    ),
    (
        "Commands &amp; holds",
        (
            ("GET", "/commands", "List all submitted commands"),
            ("GET", "/commands/{id}", "One command's status/result"),
            (
                "POST",
                "/commands",
                "Submit a command (set_learning_rate, checkpoint, snapshot, stop, ...)",
            ),
            ("GET", "/holds", "List holds, and the currently active one if any"),
            (
                "POST",
                "/holds",
                "Arm a hold; waits briefly to confirm engagement, else 202 pollable",
            ),
            ("POST", "/holds/{id}/cancel", "Cancel a pending (not yet active) hold"),
            ("POST", "/holds/{id}/release", "Release a specific active hold"),
            ("POST", "/control/pause", "Convenience: hold now"),
            ("POST", "/control/continue", "Release the currently active hold"),
            ("POST", "/control/stop", "Stop training"),
        ),
    ),
    (
        "Checkpoints &amp; snapshots",
        (
            ("POST", "/checkpoints", "Write a checkpoint"),
            ("POST", "/snapshots", "Write a snapshot"),
            (
                "GET",
                "/artifacts",
                "List checkpoints, snapshots and debug captures on disk",
            ),
            ("GET", "/artifacts/{id}", "One snapshot/debug capture's manifest"),
            (
                "POST",
                "/debug/captures",
                "Persist a debug capture taken by an external debugger tool",
            ),
        ),
    ),
    (
        "PyTorch distributed debug",
        (
            (
                "GET",
                "/debug/pytorch/status",
                "Whether torch.distributed.debug's HTTP frontend is running, and its URL",
            ),
        ),
    ),
    (
        "CUDA memory",
        (
            (
                "GET",
                "/debug/cuda-memory/snapshot",
                "Dump a CUDA allocator snapshot now, written as a debug capture",
            ),
            (
                "GET",
                "/debug/cuda-memory/history",
                "Enable/disable bounded CUDA allocator-history recording",
            ),
        ),
    ),
    (
        "Telemetry",
        (
            (
                "GET",
                "/telemetry/pipeline",
                "Rolling per-batch phase timing (input gap, forward/backward/optimizer)",
            ),
        ),
    ),
    (
        "Inspectors",
        (
            (
                "GET",
                "/inspectors",
                "Optional-package inspector availability (torchinfo/torchview/torchviz/torchlens)",
            ),
        ),
    ),
    (
        "Files",
        (
            (
                "GET",
                "/files/{path}",
                "Read the same virtual filesystem the FUSE mount projects (model/, state/, meta/, debug/)",
            ),
            (
                "GET",
                "/artifact-files/{path}",
                "Raw checkpoint/snapshot/debug-capture file content (see files_url on GET /artifacts/{id})",
            ),
        ),
    ),
)


def build_app(runtime: "TrainctlRuntime") -> FastAPI:
    """Builds the REST app bound to one `TrainctlRuntime` instance."""
    app = FastAPI(title="trainctl")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    if runtime.artifacts is not None:
        app.mount(
            "/artifact-files",
            StaticFiles(directory=str(runtime.artifacts.root)),
            name="artifact-files",
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "run_id": runtime.run_id}

    @app.get("/run")
    def get_run() -> dict[str, Any]:
        return asdict(runtime.state.read())

    @app.get("/metrics")
    def get_metrics() -> dict[str, Any]:
        return {
            key: asdict(metric) for key, metric in runtime.state.read().metrics.items()
        }

    @app.get("/optimizers")
    def get_optimizers() -> dict[str, Any]:
        return {"learning_rates": list(runtime.state.read().learning_rates)}

    @app.get("/commands")
    def list_commands() -> list[dict[str, Any]]:
        return [_command_dict(command) for command in runtime.command_queue.list_all()]

    @app.get("/commands/{command_id}")
    def get_command(command_id: str) -> dict[str, Any]:
        command = runtime.command_queue.get(command_id)
        if command is None:
            raise HTTPException(status_code=404, detail="unknown command id")
        return _command_dict(command)

    @app.post("/commands")
    def submit_command(body: CommandIn) -> dict[str, Any]:
        when = _parse_when_or_400(body.execution.when)
        command = runtime.command_queue.submit(body.kind, body.args, when)
        runtime.gate.notify()
        return _command_dict(command)

    @app.get("/holds")
    def list_holds() -> dict[str, Any]:
        holds = [
            _command_dict(command)
            for command in runtime.command_queue.list_all()
            if command.kind == "hold"
        ]
        active = runtime.gate.active_hold()
        return {"holds": holds, "active": asdict(active) if active else None}

    @app.post("/holds", response_model=None)
    def arm_hold(body: HoldIn) -> dict[str, Any] | JSONResponse:
        when = _parse_when_or_400(body.when)
        command = runtime.command_queue.submit("hold", {"reason": body.reason}, when)
        runtime.gate.notify()
        command = _wait_for_command(runtime, command.id, timeout_s=body.timeout_s)
        if command.status == "queued":
            return JSONResponse(status_code=202, content=_command_dict(command))
        return _command_dict(command)

    @app.post("/holds/{command_id}/cancel")
    def cancel_hold(command_id: str) -> dict[str, Any]:
        if not runtime.command_queue.cancel(command_id):
            raise HTTPException(
                status_code=404,
                detail="hold not cancellable (unknown id or already resolved)",
            )
        return {"cancelled": command_id}

    @app.post("/holds/{command_id}/release")
    def release_hold(command_id: str) -> dict[str, Any]:
        active = runtime.gate.active_hold()
        if active is None or active.command_id != command_id:
            raise HTTPException(status_code=404, detail="no active hold with that id")
        runtime.gate.release()
        return {"released": command_id}

    @app.post("/control/pause")
    def control_pause() -> dict[str, Any]:
        command = runtime.command_queue.submit(
            "hold", {"reason": "control/pause"}, parse_when("now")
        )
        runtime.gate.notify()
        return _command_dict(command)

    @app.post("/control/continue")
    def control_continue() -> dict[str, Any]:
        return {"released": runtime.gate.release()}

    @app.post("/control/stop")
    def control_stop() -> dict[str, Any]:
        command = runtime.command_queue.submit("stop", {}, parse_when("now"))
        runtime.gate.notify()
        return _command_dict(command)

    @app.post("/checkpoints")
    def create_checkpoint(body: CheckpointIn) -> dict[str, Any]:
        when = _parse_when_or_400(body.when)
        command = runtime.command_queue.submit(
            "checkpoint", {"label": body.label}, when
        )
        runtime.gate.notify()
        return _command_dict(command)

    @app.post("/snapshots")
    def create_snapshot(body: SnapshotIn) -> dict[str, Any]:
        when = _parse_when_or_400(body.when)
        command = runtime.command_queue.submit("snapshot", {"kind": body.kind}, when)
        runtime.gate.notify()
        return _command_dict(command)

    @app.get("/artifacts")
    def list_artifacts() -> dict[str, Any]:
        root = runtime.artifacts.root
        return {
            "checkpoints": sorted(
                entry.name for entry in (root / "checkpoints").glob("*.ckpt")
            ),
            "snapshots": sorted(entry.name for entry in (root / "snapshots").iterdir()),
            "debug": sorted(entry.name for entry in (root / "debug").iterdir()),
        }

    @app.get("/artifacts/{artifact_id}")
    def get_artifact(artifact_id: str) -> dict[str, Any]:
        for subdir in ("snapshots", "debug"):
            manifest_path = (
                runtime.artifacts.root / subdir / artifact_id / "manifest.json"
            )
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                manifest["files_url"] = f"/artifact-files/{subdir}/{artifact_id}"
                return manifest
        raise HTTPException(status_code=404, detail="unknown artifact id")

    @app.post("/debug/captures")
    def create_debug_capture(body: DebugCaptureIn) -> dict[str, Any]:
        try:
            return runtime.artifacts.write_debug_capture(
                body.kind, body.files, body.metadata
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/debug/pytorch/status")
    def debug_pytorch_status() -> dict[str, Any]:
        return {
            "enabled": runtime.torch_debug._started,
            "url": runtime.torch_debug.frontend_url,
        }

    @app.get("/debug/cuda-memory/snapshot")
    def cuda_memory_snapshot() -> dict[str, Any]:
        try:
            snapshot_bytes = cuda_memory.dump_snapshot()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        stats = cuda_memory.cheap_stats()
        return runtime.artifacts.write_debug_capture(
            "cuda-memory",
            {
                "snapshot.pickle": snapshot_bytes,
                "summary.json": json.dumps(stats, indent=2),
            },
        )

    @app.get("/debug/cuda-memory/history")
    def cuda_memory_history(
        enabled: bool, max_entries: int | None = None
    ) -> dict[str, Any]:
        try:
            state = cuda_memory.set_history(enabled, max_entries)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime.cuda_memory_history = state
        return asdict(state)

    @app.get("/telemetry/pipeline")
    def telemetry_pipeline() -> dict[str, Any]:
        latest = runtime.pipeline.latest()
        return {
            "latest": asdict(latest) if latest else None,
            "summary": runtime.pipeline.summary(),
        }

    @app.get("/inspectors")
    def get_inspectors() -> dict[str, Any]:
        return inspectors_registry.availability_snapshot()

    @app.get("/files/{path:path}")
    def get_file(path: str) -> Response:
        virtual_path = "/" + path
        if virtual_fs.is_virtual_dir(runtime, virtual_path):
            return JSONResponse(
                {
                    "path": virtual_path,
                    "entries": virtual_fs.list_dir(runtime, virtual_path),
                }
            )
        try:
            content = virtual_fs.generate(runtime, virtual_path)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            logger.warning("trainctl GET /files{} failed: {}", virtual_path, exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if content is None:
            raise HTTPException(status_code=404, detail="unknown virtual path")
        media_type = _MEDIA_TYPES.get(Path(virtual_path).suffix, "text/plain")
        return Response(content=content, media_type=media_type)

    return app


def _command_dict(command: Command) -> dict[str, Any]:
    return {
        "id": command.id,
        "kind": command.kind,
        "args": command.args,
        "status": command.status,
        "result": command.result,
        "error": command.error,
        "created_at": command.created_at,
    }


def _wait_for_command(
    runtime: "TrainctlRuntime",
    command_id: str,
    timeout_s: float,
    poll_interval_s: float = 0.05,
) -> Command:
    """Polls until `command_id` leaves `queued`, or `timeout_s` elapses.

    A hold command moves straight from `queued` to `succeeded`/`failed` synchronously
    on the training thread the instant `RunGate.enter` returns (see
    `TrainctlRuntime._execute`), so this doubles as "wait for the hold to actually
    engage." A command still `queued` when this returns hasn't reached its safe point
    yet -- e.g. a `train_epoch_end` hold armed mid-epoch -- and stays pollable via
    `GET /commands/{id}` or `GET /holds`.
    """
    deadline = time.monotonic() + timeout_s
    command = runtime.command_queue.get(command_id)
    while (
        command is not None
        and command.status == "queued"
        and time.monotonic() < deadline
    ):
        time.sleep(poll_interval_s)
        command = runtime.command_queue.get(command_id)
    assert command is not None, (
        f"command {command_id!r} vanished from the queue while waiting"
    )
    return command


def _parse_when_or_400(raw: Any) -> Any:
    try:
        return parse_when(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
