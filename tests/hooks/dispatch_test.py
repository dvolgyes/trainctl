"""Behavioral tests for trainctl.hooks.dispatch's dispatch_light."""

import json
import shutil
from pathlib import Path

from trainctl.hooks.dispatch import dispatch_light
from trainctl.hooks.payload import OccurrenceContext

_SHELL = Path(shutil.which("bash"))


class _FakeLog:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []

    def error(self, message, *args) -> None:
        self.errors.append(message.format(*args))

    def info(self, message, *args) -> None:
        self.infos.append(message.format(*args))

    def warning(self, message, *args) -> None:
        pass


def _context(**overrides) -> OccurrenceContext:
    fields = {
        "session_id": "sess-1",
        "hook": "on_train_batch_start",
        "stage": "fit",
        "epoch": 3,
        "global_step": 42,
        "batch_idx": 7,
        "dataloader_idx": None,
        "rank": 0,
        "world_size": 1,
    }
    fields.update(overrides)
    return OccurrenceContext(**fields)


def _make_light_script(live_dir, hook: str, body: str) -> None:
    script = live_dir / "light" / f"{hook}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"#!/usr/bin/env bash\n{body}\n")
    script.chmod(0o755)


def _dispatch(live_dir, tmp_root, context, *, arguments=None, log=None):
    return dispatch_light(
        live_dir,
        context,
        arguments or {},
        shell=_SHELL,
        timeout_s=5.0,
        max_params_bytes=256 * 1024,
        max_metadata_items=1000,
        max_metadata_depth=8,
        max_output_bytes=64 * 1024,
        tmp_root=tmp_root,
        log=log or _FakeLog(),
    )


def test_absent_script_returns_none_and_allocates_nothing(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    live_dir.mkdir()
    tmp_root = tmp_path / "tmp-root"
    result = _dispatch(live_dir, tmp_root, _context())
    assert result is None
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []


def test_non_executable_script_returns_none_and_allocates_nothing(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(live_dir, "on_train_batch_start", "exit 0")
    (live_dir / "light" / "on_train_batch_start.sh").chmod(0o644)
    tmp_root = tmp_path / "tmp-root"
    result = _dispatch(live_dir, tmp_root, _context())
    assert result is None
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []


def test_successful_script_returns_result_and_logs_nothing_at_error(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(live_dir, "on_train_batch_start", "exit 0")
    tmp_root = tmp_path / "tmp-root"
    log = _FakeLog()
    result = _dispatch(live_dir, tmp_root, _context(), log=log)
    assert result is not None
    assert result.phase == "completed"
    assert result.exit_code == 0
    assert log.errors == []


def test_failing_script_returns_result_and_logs_exactly_one_error(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(live_dir, "on_train_batch_start", "exit 3")
    tmp_root = tmp_path / "tmp-root"
    log = _FakeLog()
    context = _context()
    result = _dispatch(live_dir, tmp_root, context, log=log)
    assert result is not None
    assert result.exit_code == 3
    assert len(log.errors) == 1
    message = log.errors[0]
    assert context.hook in message
    assert context.session_id in message
    assert "exit_code=3" in message


def test_invocation_dir_exists_during_run_and_is_removed_after(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    marker = tmp_path / "marker.json"
    _make_light_script(live_dir, "on_train_batch_start", f'cp "$1" "{marker}"')
    tmp_root = tmp_path / "tmp-root"
    result = _dispatch(live_dir, tmp_root, _context())
    assert result is not None
    assert result.phase == "completed"
    assert result.exit_code == 0

    # The script's own copy proves params.json existed, mid-run, as valid JSON.
    marker_content = json.loads(marker.read_text())
    assert marker_content["hook"] == "on_train_batch_start"

    # The per-invocation temp dir is cleaned up afterward; nothing lingers under tmp_root.
    assert not tmp_root.exists() or list(tmp_root.iterdir()) == []


def test_metacharacter_laden_argument_is_inert(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(live_dir, "on_train_batch_start", 'cat "$1"')
    tmp_root = tmp_path / "tmp-root"
    canary = tmp_path / "should-not-run"
    payload = f"$({{ touch {canary}; }}); `id`; ${{IFS}}"
    result = _dispatch(live_dir, tmp_root, _context(), arguments={"note": payload})
    assert result is not None
    assert result.phase == "completed"
    assert not canary.exists()
    dumped = json.loads(result.stdout_tail)
    assert dumped["arguments"]["note"] == payload


def test_context_fields_round_trip_into_params_json(tmp_path) -> None:
    live_dir = tmp_path / "hooks"
    _make_light_script(live_dir, "on_train_batch_start", 'cat "$1"')
    tmp_root = tmp_path / "tmp-root"
    context = _context(stage="validate", epoch=9, session_id="sess-xyz")
    result = _dispatch(live_dir, tmp_root, context)
    assert result is not None
    dumped = json.loads(result.stdout_tail)
    assert dumped["hook"] == "on_train_batch_start"
    assert dumped["session_id"] == "sess-xyz"
    assert dumped["stage"] == "validate"
    assert dumped["epoch"] == 9
