"""Behavioral tests for trainctl.hooks.discovery (T2)."""

import os
import stat

import pytest
from loguru import logger

from trainctl.hooks.discovery import EpochScanState, resolve_enabled_script


@pytest.fixture()
def log():
    return logger


def _make_script(path, *, executable: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nexit 0\n")
    mode = 0o755 if executable else 0o644
    path.chmod(mode)


def test_absent_script_is_not_enabled(tmp_path) -> None:
    assert resolve_enabled_script(tmp_path, "light", "on_train_batch_start") is None


def test_non_executable_script_is_not_enabled(tmp_path) -> None:
    script = tmp_path / "light" / "on_train_batch_start.sh"
    _make_script(script, executable=False)
    assert resolve_enabled_script(tmp_path, "light", "on_train_batch_start") is None


def test_chmod_plus_x_enables_immediately(tmp_path) -> None:
    script = tmp_path / "light" / "on_train_batch_start.sh"
    _make_script(script, executable=False)
    assert resolve_enabled_script(tmp_path, "light", "on_train_batch_start") is None
    script.chmod(0o755)
    assert resolve_enabled_script(tmp_path, "light", "on_train_batch_start") == script


def test_chmod_minus_x_disables_immediately(tmp_path) -> None:
    script = tmp_path / "light" / "on_train_batch_start.sh"
    _make_script(script, executable=True)
    assert resolve_enabled_script(tmp_path, "light", "on_train_batch_start") == script
    script.chmod(0o644)
    assert resolve_enabled_script(tmp_path, "light", "on_train_batch_start") is None


def test_delete_then_recreate_is_freshly_validated(tmp_path) -> None:
    script = tmp_path / "light" / "on_train_batch_start.sh"
    _make_script(script, executable=True)
    script.unlink()
    assert resolve_enabled_script(tmp_path, "light", "on_train_batch_start") is None
    _make_script(script, executable=True)
    assert resolve_enabled_script(tmp_path, "light", "on_train_batch_start") == script


def test_symlink_is_never_enabled_even_if_target_is_executable(tmp_path) -> None:
    real = tmp_path / "real.sh"
    _make_script(real, executable=True)
    link = tmp_path / "light" / "on_train_batch_start.sh"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)
    assert stat.S_ISLNK(os.lstat(link).st_mode)
    assert resolve_enabled_script(tmp_path, "light", "on_train_batch_start") is None


def test_scan_if_due_warns_once_per_unknown_path(tmp_path, log) -> None:
    (tmp_path / "light").mkdir()
    (tmp_path / "heavy").mkdir()
    _make_script(tmp_path / "light" / "not_a_callback.sh", executable=False)
    warnings: list[str] = []

    class _FakeLog:
        def warning(self, message, *args) -> None:
            warnings.append(message.format(*args))

    state = EpochScanState()
    state.scan_if_due(tmp_path, ("fit", 0), _FakeLog())
    assert len(warnings) == 1
    assert "not_a_callback.sh" in warnings[0]

    # Same epoch token: no re-scan at all.
    state.scan_if_due(tmp_path, ("fit", 0), _FakeLog())
    assert len(warnings) == 1

    # New epoch token re-scans, but the same path never warns twice.
    state.scan_if_due(tmp_path, ("fit", 1), _FakeLog())
    assert len(warnings) == 1


def test_scan_if_due_does_not_warn_about_known_callback_names(tmp_path, log) -> None:
    (tmp_path / "light").mkdir()
    (tmp_path / "heavy").mkdir()
    _make_script(tmp_path / "light" / "on_train_batch_start.sh", executable=False)
    warnings: list[str] = []

    class _FakeLog:
        def warning(self, message, *args) -> None:
            warnings.append(message.format(*args))

    state = EpochScanState()
    state.scan_if_due(tmp_path, ("fit", 0), _FakeLog())
    assert warnings == []


def test_scan_if_due_skips_unreadable_modality_dir_without_raising(tmp_path, log) -> None:
    state = EpochScanState()
    state.scan_if_due(tmp_path / "does-not-exist", ("fit", 0), log)
