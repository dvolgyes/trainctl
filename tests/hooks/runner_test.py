"""Behavioral tests for trainctl.hooks.runner (T5)."""

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest
from loguru import logger

from trainctl.hooks.runner import run_script

_SHELL = Path(shutil.which("bash"))


@pytest.fixture()
def log():
    return logger


@pytest.fixture()
def shell() -> Path:
    return _SHELL


def _write_script(directory: Path, content: str, name: str = "script.sh") -> Path:
    path = directory / name
    path.write_text(content)
    path.chmod(0o755)
    return path


def _assert_process_dead(pid: int, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail(f"pid {pid} is still alive after {timeout_s}s")


def test_success_reports_completed_with_exit_code_zero(tmp_path, shell, log) -> None:
    script = _write_script(tmp_path, "#!/usr/bin/env bash\nexit 0\n")
    result = run_script(
        script,
        [],
        cwd=tmp_path,
        shell=shell,
        timeout_s=5.0,
        max_output_bytes=1024,
        log=log,
    )
    assert result.phase == "completed"
    assert result.exit_code == 0
    assert result.signal_number is None
    assert result.error is None


def test_nonzero_exit_is_reported_as_data_not_raised(tmp_path, shell, log) -> None:
    script = _write_script(tmp_path, "#!/usr/bin/env bash\nexit 7\n")
    result = run_script(
        script,
        [],
        cwd=tmp_path,
        shell=shell,
        timeout_s=5.0,
        max_output_bytes=1024,
        log=log,
    )
    assert result.phase == "completed"
    assert result.exit_code == 7


def test_argv_extra_is_forwarded_as_positional_argument(tmp_path, shell, log) -> None:
    script = _write_script(tmp_path, '#!/usr/bin/env bash\ncat "$1" > "$1.seen"\n')
    params_path = tmp_path / "params.json"
    params_path.write_text("hello-params")
    result = run_script(
        script,
        [str(params_path)],
        cwd=tmp_path,
        shell=shell,
        timeout_s=5.0,
        max_output_bytes=1024,
        log=log,
    )
    assert result.phase == "completed"
    assert result.exit_code == 0
    assert (tmp_path / "params.json.seen").read_text() == "hello-params"


def test_cwd_is_honored(tmp_path, shell, log) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    script = _write_script(tmp_path, "#!/usr/bin/env bash\npwd > pwd.out\n")
    result = run_script(
        script,
        [],
        cwd=workdir,
        shell=shell,
        timeout_s=5.0,
        max_output_bytes=1024,
        log=log,
    )
    assert result.phase == "completed"
    recorded = (workdir / "pwd.out").read_text().strip()
    assert os.path.realpath(recorded) == os.path.realpath(str(workdir))


def test_stdin_is_devnull_so_read_sees_immediate_eof(tmp_path, shell, log) -> None:
    script = _write_script(
        tmp_path, '#!/usr/bin/env bash\nread -t 2 line\necho "read_rc=$?"\n'
    )
    start = time.monotonic()
    result = run_script(
        script,
        [],
        cwd=tmp_path,
        shell=shell,
        timeout_s=5.0,
        max_output_bytes=1024,
        log=log,
    )
    elapsed = time.monotonic() - start
    assert result.phase == "completed"
    assert "read_rc=" in result.stdout_tail
    assert "read_rc=0" not in result.stdout_tail
    assert elapsed < 1.5


def test_timeout_kills_the_entire_process_group(tmp_path, shell, log) -> None:
    script = _write_script(
        tmp_path,
        "#!/usr/bin/env bash\nsleep 100 &\necho $! > child.pid\nwait\n",
    )
    result = run_script(
        script,
        [],
        cwd=tmp_path,
        shell=shell,
        timeout_s=0.3,
        max_output_bytes=1024,
        log=log,
    )
    assert result.phase == "timeout"
    pid = int((tmp_path / "child.pid").read_text().strip())
    _assert_process_dead(pid)


def test_long_output_is_bounded(tmp_path, shell, log) -> None:
    script = _write_script(tmp_path, "#!/usr/bin/env bash\nyes | head -c 10000\n")
    result = run_script(
        script,
        [],
        cwd=tmp_path,
        shell=shell,
        timeout_s=5.0,
        max_output_bytes=100,
        log=log,
    )
    assert result.phase == "completed"
    assert len(result.stdout_tail.encode("utf-8", errors="replace")) <= 100
    assert result.stdout_truncated is True


def test_binary_output_does_not_raise(tmp_path, shell, log) -> None:
    script = _write_script(tmp_path, "#!/usr/bin/env bash\nprintf '\\xff\\xfe'\n")
    result = run_script(
        script,
        [],
        cwd=tmp_path,
        shell=shell,
        timeout_s=5.0,
        max_output_bytes=1024,
        log=log,
    )
    assert result.phase == "completed"
    assert isinstance(result.stdout_tail, str)


def test_keyboard_interrupt_is_contained_then_reraised(
    tmp_path, shell, log, monkeypatch
) -> None:
    script = _write_script(
        tmp_path,
        "#!/usr/bin/env bash\nsleep 100 &\necho $! > child.pid\nwait\n",
    )
    real_wait = subprocess.Popen.wait
    calls = {"n": 0}
    pid_file = tmp_path / "child.pid"

    def fake_wait(self, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            deadline = time.monotonic() + 2.0
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            raise KeyboardInterrupt
        return real_wait(self, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", fake_wait)

    with pytest.raises(KeyboardInterrupt):
        run_script(
            script,
            [],
            cwd=tmp_path,
            shell=shell,
            timeout_s=None,
            max_output_bytes=1024,
            log=log,
        )

    pid = int((tmp_path / "child.pid").read_text().strip())
    _assert_process_dead(pid)


def test_launch_error_is_reported_not_raised(tmp_path, log) -> None:
    script = _write_script(tmp_path, "#!/usr/bin/env bash\nexit 0\n")
    result = run_script(
        script,
        [],
        cwd=tmp_path,
        shell=Path("/nonexistent/not-a-real-shell"),
        timeout_s=5.0,
        max_output_bytes=1024,
        log=log,
    )
    assert result.phase == "launch_error"
    assert result.exit_code is None
    assert result.error
