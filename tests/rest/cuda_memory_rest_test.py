"""Behavioral tests for GET /debug/cuda-memory/snapshot and GET /debug/cuda-memory/history.

These operations don't touch the Trainer/model (CUDA memory is process-global
state), so they execute directly from the REST route -- not through
POST /commands -- and are plain GETs since neither uploads a file nor needs a
complex request body.
"""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.rest.app import build_app
from trainctl.runtime import cuda_memory
from trainctl.runtime.artifacts import ArtifactStore
from trainctl.runtime.runtime import TrainctlRuntime

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _client(tmp_path: Path) -> tuple[TestClient, TrainctlRuntime]:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
    runtime.artifacts = ArtifactStore(tmp_path)
    return TestClient(build_app(runtime)), runtime


def test_get_cuda_memory_snapshot_writes_artifact(tmp_path: Path) -> None:
    client, _runtime = _client(tmp_path)

    response = client.get("/debug/cuda-memory/snapshot")

    assert response.status_code == 200
    manifest = response.json()
    assert (Path(manifest["path"]) / "snapshot.pickle").exists()
    assert (Path(manifest["path"]) / "summary.json").exists()


def test_get_cuda_memory_history_enables_and_updates_runtime_state(
    tmp_path: Path,
) -> None:
    client, runtime = _client(tmp_path)
    try:
        response = client.get(
            "/debug/cuda-memory/history",
            params={"enabled": "true", "max_entries": 500},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["enabled"] is True
        assert body["max_entries"] == 500
        assert runtime.cuda_memory_history.enabled is True
        assert runtime.cuda_memory_history.max_entries == 500
    finally:
        cuda_memory.set_history(False, max_entries=None)


def test_get_cuda_memory_history_disable_needs_no_max_entries(tmp_path: Path) -> None:
    client, _runtime = _client(tmp_path)

    response = client.get("/debug/cuda-memory/history", params={"enabled": "false"})

    assert response.status_code == 200
    assert response.json()["enabled"] is False
