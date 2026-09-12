"""Behavioral tests for trainctl.hooks.session's HookSession."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from loguru import logger

from trainctl.hooks.session import HookSession

_SHELL = Path(shutil.which("bash"))


@pytest.fixture()
def log():
    return logger


class _FakeLog:
    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def warning(self, message, *args) -> None:
        self.warnings.append(message.format(*args))

    def error(self, message, *args) -> None:
        self.errors.append(message.format(*args))

    def info(self, *_args, **_kwargs) -> None:
        pass


def _make_script(path: Path, body: str, *, executable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/usr/bin/env bash\n{body}\n")
    path.chmod(0o755 if executable else 0o644)


def _make_session(
    live_dir: Path, *, rank: int, rank_policy: str, log=None
) -> HookSession:
    return HookSession(
        live_dir,
        session_id="s-1",
        rank=rank,
        world_size=2,
        rank_policy=rank_policy,
        shell=_SHELL,
        timeout_s=5.0,
        max_params_bytes=256 * 1024,
        max_metadata_items=1000,
        max_metadata_depth=8,
        max_export_bytes=1024**3,
        max_output_bytes=64 * 1024,
        log=log if log is not None else logger,
    )


def test_rank_zero_policy_dispatches_on_rank_zero(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", "exit 0")
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result is not None
    assert result.light.phase == "completed"


def test_rank_zero_policy_skips_non_zero_rank_and_allocates_nothing(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    marker = tmp_path / "marker"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", f'touch "{marker}"')
    session = _make_session(live_dir, rank=1, rank_policy="rank_zero")

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result is None
    assert not marker.exists()


def test_all_policy_dispatches_on_non_zero_rank(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", "exit 0")
    session = _make_session(live_dir, rank=1, rank_policy="all")

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result is not None
    assert result.light.phase == "completed"


@pytest.mark.parametrize("rank_policy", ["rank_zero", "all"])
def test_disabled_script_returns_none_on_rank_zero(tmp_path, rank_policy) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(
        live_dir / "light" / "on_train_batch_start.sh", "exit 0", executable=False
    )
    session = _make_session(live_dir, rank=0, rank_policy=rank_policy)

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result is not None
    assert result.light is None
    assert result.heavy is None


def test_handle_threads_identity_and_arguments_into_params(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=3,
        global_step=42,
        batch_idx=7,
        dataloader_idx=1,
        arguments={"note": "hello"},
    )

    assert result is not None
    assert result.light.phase == "completed"
    seen = result.light.stdout_tail
    assert '"session_id": "s-1"' in seen
    assert '"hook": "on_train_batch_start"' in seen
    assert '"stage": "fit"' in seen
    assert '"epoch": 3' in seen
    assert '"global_step": 42' in seen
    assert '"batch_idx": 7' in seen
    assert '"dataloader_idx": 1' in seen
    assert '"hello"' in seen


def test_epoch_scan_warns_once_ever_for_unknown_path_across_handle_calls(
    tmp_path,
) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", "exit 0")
    _make_script(live_dir / "light" / "not_a_callback.sh", "exit 0", executable=False)
    fake_log = _FakeLog()
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero", log=fake_log)

    session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)
    session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=1)
    session.handle("on_train_batch_start", stage="fit", epoch=1, global_step=2)

    assert len(fake_log.warnings) == 1
    assert "not_a_callback.sh" in fake_log.warnings[0]


def test_close_removes_tmp_root_and_is_idempotent(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", "exit 0")
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")
    session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    tmp_root = live_dir.parent / ".hooks-tmp"
    assert tmp_root.exists()

    session.close()
    assert not tmp_root.exists()

    session.close()  # idempotent, must not raise
    assert not tmp_root.exists()


def test_public_identity_attributes_match_constructor_args(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    session = _make_session(live_dir, rank=1, rank_policy="all")

    assert session.live_dir == live_dir
    assert session.session_id == "s-1"
    assert session.rank == 1
    assert session.world_size == 2


def test_handle_dispatches_both_light_and_heavy_when_both_enabled(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", 'cat "$1"')
    _make_script(live_dir / "heavy" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=0,
        global_step=0,
        arguments={"batch": {"x": torch.arange(4.0)}},
    )

    assert result.light is not None
    assert result.light.phase == "completed"
    assert result.heavy is not None
    assert result.heavy.phase == "completed"


def test_light_and_heavy_share_occurrence_id_but_not_invocation_id(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", 'cat "$1"')
    _make_script(live_dir / "heavy" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=0,
        global_step=0,
        arguments={"batch": {"x": torch.arange(4.0)}},
    )

    light_doc = json.loads(result.light.stdout_tail)
    heavy_doc = json.loads(result.heavy.stdout_tail)
    assert light_doc["occurrence_id"] == heavy_doc["occurrence_id"]
    assert light_doc["invocation_id"] != heavy_doc["invocation_id"]


def test_only_heavy_enabled_light_is_none(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "heavy" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=0,
        global_step=0,
        arguments={"batch": {"x": torch.arange(4.0)}},
    )

    assert result.light is None
    assert result.heavy is not None
    assert result.heavy.phase == "completed"


def test_only_light_enabled_heavy_is_none(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle("on_train_batch_start", stage="fit", epoch=0, global_step=0)

    assert result.heavy is None
    assert result.light is not None
    assert result.light.phase == "completed"


def test_heavy_export_restricted_to_catalogue_tensor_args(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "heavy" / "on_train_batch_start.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=0,
        global_step=0,
        arguments={
            "batch": {"x": torch.arange(4.0)},
            "unrelated": {"y": torch.arange(4.0)},
        },
    )

    manifest = json.loads(result.heavy.stdout_tail)
    paths = {record["argument_path"] for record in manifest["tensor_records"]}
    assert "batch/x" in paths
    assert not any(path.startswith("unrelated") for path in paths)


def test_heavy_dispatches_with_empty_manifest_for_tensorless_hook(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    _make_script(live_dir / "heavy" / "on_train_epoch_end.sh", 'cat "$1"')
    session = _make_session(live_dir, rank=0, rank_policy="rank_zero")

    result = session.handle("on_train_epoch_end", stage="fit", epoch=0, global_step=0)

    assert result.heavy is not None
    assert result.heavy.phase == "completed"
    manifest = json.loads(result.heavy.stdout_tail)
    assert manifest["tensor_records"] == []


def test_rank_gating_skips_both_modalities_together(tmp_path) -> None:
    live_dir = tmp_path / "run" / "hooks"
    heavy_marker = tmp_path / "heavy-marker"
    _make_script(live_dir / "light" / "on_train_batch_start.sh", "exit 0")
    _make_script(
        live_dir / "heavy" / "on_train_batch_start.sh", f'touch "{heavy_marker}"'
    )
    session = _make_session(live_dir, rank=1, rank_policy="rank_zero")

    result = session.handle(
        "on_train_batch_start",
        stage="fit",
        epoch=0,
        global_step=0,
        arguments={"batch": {"x": torch.arange(4.0)}},
    )

    assert result is None
    assert not heavy_marker.exists()
