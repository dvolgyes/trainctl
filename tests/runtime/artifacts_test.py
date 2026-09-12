"""Behavioral tests for ArtifactStore.write_debug_capture (multi-file + metadata)."""

import json
import threading
from pathlib import Path

import pytest

from trainctl.runtime.artifacts import ArtifactStore


def test_write_debug_capture_writes_files_and_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.write_debug_capture(
        "pyspy-stack", {"result.json": '{"a": 1}'}, {"pid": 123}
    )

    dbg_dir = tmp_path / "debug" / manifest["id"]
    assert (dbg_dir / "result.json").read_text() == '{"a": 1}'

    written_manifest = json.loads((dbg_dir / "manifest.json").read_text())
    assert written_manifest["kind"] == "pyspy-stack"
    assert written_manifest["metadata"] == {"pid": 123}
    assert written_manifest["files"] == ["result.json"]


def test_write_debug_capture_writes_multiple_named_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    manifest = store.write_debug_capture(
        "gdb-snapshot", {"result.json": "{}", "raw.txt": "hello"}
    )

    dbg_dir = tmp_path / "debug" / manifest["id"]
    assert (dbg_dir / "raw.txt").read_text() == "hello"
    assert manifest["files"] == ["raw.txt", "result.json"]
    assert manifest["metadata"] == {}


@pytest.mark.parametrize("bad_filename", ["../escape.txt", "sub/dir.txt", ".", ".."])
def test_write_debug_capture_rejects_path_traversal(
    tmp_path: Path, bad_filename: str
) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="invalid capture filename"):
        store.write_debug_capture("bad", {bad_filename: "x"})


def test_write_debug_capture_ids_dont_collide_under_concurrency(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ids: list[str] = []
    lock = threading.Lock()

    def _write() -> None:
        manifest = store.write_debug_capture("kind", {"f.txt": "x"})
        with lock:
            ids.append(manifest["id"])

    threads = [threading.Thread(target=_write) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ids) == 20
    assert len(set(ids)) == 20
