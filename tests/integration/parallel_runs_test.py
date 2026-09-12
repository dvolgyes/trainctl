"""Behavioral test: two real Lightning trainings running as separate OS
subprocesses at the same time, each starting its own Trainctl REST server and
`torch.distributed` debug server while requesting trainctl's *default* ports.

Exercises the concern raised for concurrent runs sharing a host: REST port
collision avoidance (`runtime.runtime._bind_free_socket`), torch_debug port
collision avoidance (`TorchDebugManager._select_free_port`), and MASTER_PORT
isolation for `torch.distributed.init_process_group` -- and proves an API
client can address each run's own server without cross-talk.

Runs as separate processes rather than threads because
`torch.distributed`/`torch.distributed.debug` hold process-wide state and
cannot be initialized twice in one interpreter.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx2

_FIXTURE = Path(__file__).parent / "_parallel_training_fixture.py"


def _launch(tmp_path: Path, index: int) -> tuple[subprocess.Popen, Path]:
    log_dir = tmp_path / f"run{index}"
    log_dir.mkdir()
    status_file = tmp_path / f"status{index}.json"
    log_file = (tmp_path / f"run{index}.log").open("w")
    env = {
        **os.environ,
        "TRAINCTL_TEST_MASTER_PORT": str(29601 + index),
        "TRAINCTL_TEST_LOG_DIR": str(log_dir),
        "TRAINCTL_TEST_STATUS_FILE": str(status_file),
    }
    proc = subprocess.Popen(
        [sys.executable, str(_FIXTURE)],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return proc, status_file


def _wait_for_status(
    proc: subprocess.Popen, status_file: Path, timeout: float = 45.0
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if status_file.exists():
            try:
                return json.loads(status_file.read_text())
            except json.JSONDecodeError:
                pass
        if proc.poll() is not None:
            log_text = _read_log(status_file)
            raise RuntimeError(
                f"fixture subprocess exited early with code {proc.returncode} "
                f"before writing {status_file}:\n{log_text}"
            )
        time.sleep(0.1)
    raise TimeoutError(f"{status_file} was not written within {timeout}s")


def _read_log(status_file: Path) -> str:
    log_file = status_file.parent / f"{status_file.stem.replace('status', 'run')}.log"
    return log_file.read_text() if log_file.exists() else "(no log)"


def test_two_parallel_trainings_get_distinct_ports_and_no_cross_talk(
    tmp_path: Path,
) -> None:
    proc_a, status_a = _launch(tmp_path, 0)
    proc_b, status_b = _launch(tmp_path, 1)
    try:
        info_a = _wait_for_status(proc_a, status_a)
        info_b = _wait_for_status(proc_b, status_b)

        # run_id is process-local (trainctl.runtime.commands.next_id), so two
        # independent processes may legitimately produce the same run_id; the
        # field that actually distinguishes concurrent runs sharing a host is
        # artifacts_path (trainctl.runtime.state.RuntimeSnapshot.artifacts_path).
        assert info_a["artifacts_path"] != info_b["artifacts_path"]
        assert info_a["rest_port"] != info_b["rest_port"]
        assert info_a["torch_debug_url"] is not None
        assert info_b["torch_debug_url"] is not None
        assert info_a["torch_debug_url"] != info_b["torch_debug_url"]

        run_a = httpx2.get(
            f"http://127.0.0.1:{info_a['rest_port']}/run", timeout=5
        ).json()
        run_b = httpx2.get(
            f"http://127.0.0.1:{info_b['rest_port']}/run", timeout=5
        ).json()
        assert run_a["run_id"] == info_a["run_id"]
        assert run_b["run_id"] == info_b["run_id"]

        httpx2.post(f"http://127.0.0.1:{info_a['rest_port']}/control/stop", timeout=5)
        assert proc_a.wait(timeout=20) == 0

        run_b_after = httpx2.get(
            f"http://127.0.0.1:{info_b['rest_port']}/run", timeout=5
        ).json()
        assert run_b_after["run_id"] == info_b["run_id"]
        assert run_b_after["status"] == "running"

        httpx2.post(f"http://127.0.0.1:{info_b['rest_port']}/control/stop", timeout=5)
        assert proc_b.wait(timeout=20) == 0
    finally:
        for proc in (proc_a, proc_b):
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
