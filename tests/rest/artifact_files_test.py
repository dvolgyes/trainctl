"""Behavioral tests for the /artifact-files StaticFiles mount and GET /artifacts/{id}'s files_url.

Checkpoints/snapshots/debug captures are real files under `runtime.artifacts.root`;
the mount serves them directly rather than through a hand-rolled download route.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.rest.app import build_app
from trainctl.runtime.artifacts import ArtifactStore
from trainctl.runtime.runtime import TrainctlRuntime


def _client(tmp_path: Path) -> tuple[TestClient, TrainctlRuntime]:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return TestClient(build_app(runtime)), runtime


def test_get_artifact_includes_files_url(tmp_path: Path) -> None:
    client, runtime = _client(tmp_path)
    manifest = runtime.artifacts.write_debug_capture("demo", {"summary.txt": "hello"})

    response = client.get(f"/artifacts/{manifest['id']}")

    assert response.status_code == 200
    assert response.json()["files_url"] == f"/artifact-files/debug/{manifest['id']}"


def test_artifact_files_mount_serves_debug_capture_bytes(tmp_path: Path) -> None:
    client, runtime = _client(tmp_path)
    manifest = runtime.artifacts.write_debug_capture(
        "demo", {"summary.txt": "hello world"}
    )

    response = client.get(f"/artifact-files/debug/{manifest['id']}/summary.txt")

    assert response.status_code == 200
    assert response.text == "hello world"


def test_artifact_files_mount_serves_snapshot_payload(tmp_path: Path) -> None:
    client, runtime = _client(tmp_path)
    manifest = runtime.artifacts.write_snapshot("runtime", {"foo": "bar"})

    response = client.get(f"/artifact-files/snapshots/{manifest['id']}/runtime.json")

    assert response.status_code == 200
    assert response.json() == {"foo": "bar"}


def test_artifact_files_mount_404_for_unknown_file(tmp_path: Path) -> None:
    client, _runtime = _client(tmp_path)

    response = client.get("/artifact-files/debug/does-not-exist/summary.txt")

    assert response.status_code == 404
