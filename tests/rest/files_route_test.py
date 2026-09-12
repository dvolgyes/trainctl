"""Behavioral tests for GET /files/{path} -- the REST mirror of the FUSE virtual tree.

`get_file` reuses `virtual_fs.generate`/`is_virtual_dir`/`list_dir`, the same functions
`TrainctlFS` calls -- these tests check the REST-specific wiring (status codes, JSON
listing shape, error-boundary translation), not the virtual tree's content, which
`tests/fuse/*` already covers.
"""

import pytest
from fastapi.testclient import TestClient

from trainctl.config import TrainctlConfig
from trainctl.lightning_backend import load_backend
from trainctl.rest.app import build_app
from trainctl.runtime import virtual_fs
from trainctl.runtime.runtime import TrainctlRuntime


def _client() -> tuple[TestClient, TrainctlRuntime]:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False
    )
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
    return TestClient(build_app(runtime)), runtime


def test_get_file_matches_virtual_fs_generate_content() -> None:
    client, runtime = _client()
    runtime._model_class_name = "DemoModule"

    response = client.get("/files/model/class")

    assert response.status_code == 200
    assert response.text == virtual_fs.generate(runtime, "/model/class").decode()


def test_get_file_json_content_type_for_json_path() -> None:
    client, runtime = _client()

    response = client.get("/files/model/hparams.json")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")


def test_get_file_on_virtual_directory_returns_json_listing() -> None:
    client, _runtime = _client()

    response = client.get("/files/model")

    assert response.status_code == 200
    body = response.json()
    assert body["path"] == "/model"
    assert "class" in body["entries"]


def test_get_file_root_directory_returns_json_listing() -> None:
    client, _runtime = _client()

    response = client.get("/files/")

    assert response.status_code == 200
    assert response.json()["path"] == "/"


def test_get_file_unknown_path_is_404() -> None:
    client, _runtime = _client()

    response = client.get("/files/does/not/exist")

    assert response.status_code == 404


def test_get_file_live_view_timeout_maps_to_504(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _runtime = _client()

    def _raise_timeout(rt: object) -> bytes:
        raise TimeoutError("no safe point reached")

    monkeypatch.setitem(virtual_fs._LIVE_FILES, "/model/summary-live.txt", _raise_timeout)

    response = client.get("/files/model/summary-live.txt")

    assert response.status_code == 504


def test_get_file_other_failure_maps_to_500(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _runtime = _client()

    def _raise_value_error(rt: object) -> bytes:
        raise ValueError("requires an in-flight batch")

    monkeypatch.setitem(
        virtual_fs._LIVE_FILES, "/model/summary-live.txt", _raise_value_error
    )

    response = client.get("/files/model/summary-live.txt")

    assert response.status_code == 500
