"""Behavioral tests for GET /inspectors."""

from fastapi.testclient import TestClient

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.rest.app import build_app
from trainctl.runtime.runtime import TrainctlRuntime


def _client() -> TestClient:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
    return TestClient(build_app(runtime))


def test_inspectors_reports_availability_for_all_four() -> None:
    response = _client().get("/inspectors")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"torchinfo", "torchview", "torchviz", "torchlens"}
    for entry in body.values():
        assert isinstance(entry["available"], bool)
