"""Disk artifact layout for checkpoints, snapshots, and debug captures.

Artifacts are written by the training thread (via command handlers) and read back
read-only through the FUSE `artifacts/` tree. This module owns only the on-disk layout
and manifest format, not the FUSE projection.
"""

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from trainctl.runtime.commands import next_id


class ArtifactStore:
    """Materializes checkpoints, snapshots, and debug captures under `root`.

    Attributes:
        root: Base artifact directory for this run.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        (self.root / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.root / "snapshots").mkdir(parents=True, exist_ok=True)
        (self.root / "debug").mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, label: str) -> Path:
        """Returns a fresh checkpoint file path under `checkpoints/`."""
        return self.root / "checkpoints" / f"ckpt-{next_id('c')}-{label}.ckpt"

    def write_snapshot(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Writes a snapshot's manifest and payload; returns the manifest.

        Args:
            kind: Snapshot kind, e.g. `runtime`, `model_summary`, `optimizer_summary`.
            payload: JSON-serializable snapshot content.

        Returns:
            The written manifest (id, kind, created_at, and payload path).
        """
        snap_id = next_id("snap")
        snap_dir = self.root / "snapshots" / snap_id
        snap_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"id": snap_id, "kind": kind, "created_at": time.time()}
        (snap_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (snap_dir / f"{kind}.json").write_text(
            json.dumps(payload, indent=2, default=_json_default)
        )
        manifest["path"] = str(snap_dir)
        return manifest

    def write_debug_capture(
        self,
        kind: str,
        files: dict[str, str | bytes],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Writes a debug capture's manifest plus one or more named files; returns the manifest.

        Args:
            kind: Debug capture kind, e.g. `pyspy-stack`, `gdb-snapshot`, `batch-tensor`.
            files: Mapping of bare filename (no path separators) to content. A `str`
                value is written as text; a `bytes` value (e.g. a `torch.save` buffer)
                is written as-is.
            metadata: Optional JSON-serializable context to record alongside the capture.

        Returns:
            The written manifest (id, kind, created_at, metadata, filenames, and path).

        Raises:
            ValueError: a filename in `files` is not a bare name (REST-supplied input boundary:
                rejects path separators/`..` so a capture can never write outside its own directory).
        """
        for filename in files:
            if "/" in filename or filename in (".", ".."):
                raise ValueError(
                    f"invalid capture filename {filename!r}: must be a bare filename"
                )
        dbg_id = next_id("dbg")
        dbg_dir = self.root / "debug" / dbg_id
        dbg_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "id": dbg_id,
            "kind": kind,
            "created_at": time.time(),
            "metadata": metadata or {},
            "files": sorted(files),
        }
        (dbg_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for filename, content in files.items():
            if isinstance(content, bytes):
                (dbg_dir / filename).write_bytes(content)
            else:
                (dbg_dir / filename).write_text(content)
        manifest["path"] = str(dbg_dir)
        return manifest


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)
