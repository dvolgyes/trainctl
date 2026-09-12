"""Higher-risk, equally-supported capture mode: runs arbitrary Python inside a live target process via GDB call-injection.

Neither this project's nor a typical system CPython ships the DWARF-based
`python-gdb.py` extension needed for the standard `py-bt`/`py-list` gdb commands.
This exploits a different fact instead: a live CPython binary's dynamic symbol
table still exports its public C-API (`PyRun_SimpleString`, `Py_AddPendingCall`,
...) even fully stripped, and gdb can call these by address via
`-data-evaluate-expression`.

Safety history (do not relax without re-reading this): an earlier variant used
`PyGILState_Ensure()` + `PyRun_SimpleString(...)` directly, which permanently
deadlocked a live CUDA training process -- PyTorch's `.item()` releases the GIL
during a blocking CUDA sync, and gdb's `call` hijacking the current thread's
execution context there can deadlock; a stray `call` landing mid-`backward()`
also tripped up autograd engine locking. `Py_AddPendingCall` is markedly safer --
it schedules `func(arg)` for the interpreter's own next eval-loop checkpoint
instead of hijacking in-flight execution -- but is NOT proven fully immune: a
repeat test still deadlocked a live CUDA target once. Arming a trainctl hold
first (see `trainctl.debugger.hold_client`) and confirming it is active converts
this from an unrecoverable deadlock into a recoverable, retry-able reliability
issue -- not a full fix, and no help at all against an already-hung process.
Prefer `trainctl.debugger.pyspy_capture` whenever a read-only stack/locals dump
is enough; reach for this module when you need to mutate live state or call an
arbitrary live function instead.
"""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pygdbmi.gdbcontroller import GdbController

from trainctl.debugger.hold_client import HoldClient


class GdbUnavailableError(RuntimeError):
    """No usable `gdb --interpreter=mi3` binary was found."""


class GdbAttachError(RuntimeError):
    """A gdb attach/call attempt failed for a non-retryable reason."""


class GdbAbandonedEvaluationError(RuntimeError):
    """gdb abandoned an evaluation mid-call.

    Typically a forking/exiting DataLoader worker delivering a stray signal
    during the call. Retryable: stress-testing found this never corrupts or
    hangs the target, even after dozens of consecutive failures in one trial.
    """


def check_gdb_available(gdb_bin: str = "gdb") -> None:
    """Raises `GdbUnavailableError` with a clear message if `gdb_bin` is unusable.

    Checked upfront so a missing/broken gdb surfaces here, not as an opaque
    failure deep inside `pygdbmi`.
    """
    resolved = shutil.which(gdb_bin)
    if resolved is None:
        raise GdbUnavailableError(f"{gdb_bin!r} not found on PATH")
    probe = subprocess.run(  # noqa: S603 -- capability probe of an operator-specified gdb binary
        [resolved, "--version"], capture_output=True, text=True, timeout=5, check=False
    )
    if probe.returncode != 0:
        raise GdbUnavailableError(
            f"{resolved} --version failed: {probe.stderr.strip()}"
        )


def _mi_eval(gdbmi: GdbController, expr: str, timeout_sec: float = 10.0) -> str:
    """Evaluates a C expression (including function calls) in the target via MI.

    Raises:
        GdbAbandonedEvaluationError: gdb returned `^done` with no `value` payload --
            a call abandoned mid-evaluation (signal collision), not a real error.
        GdbAttachError: gdb reported a genuine evaluation error, or no result
            record was produced at all.
    """
    for record in gdbmi.write(
        f'-data-evaluate-expression "{expr}"', timeout_sec=timeout_sec
    ):
        if record["type"] != "result":
            continue
        if record["message"] == "done":
            if not record["payload"] or "value" not in record["payload"]:
                raise GdbAbandonedEvaluationError(
                    f"call to {expr!r} was abandoned mid-evaluation (signal collision)"
                )
            return str(record["payload"]["value"])
        raise GdbAttachError(f"gdb error evaluating {expr!r}: {record['payload']}")
    raise GdbAttachError(f"no result record for {expr!r}")


def _attempt_schedule(pid: int, snippet_path: Path, gdb_bin: str) -> None:
    """One attach-call-detach attempt; see `schedule` for why this can need retries."""
    gdbmi = GdbController(
        command=[gdb_bin, "--nx", "--quiet", "--interpreter=mi3", "-p", str(pid)]
    )
    try:
        gdbmi.get_gdb_response(timeout_sec=5)
        gdbmi.write("-gdb-set scheduler-locking on", timeout_sec=5)
        c_call = (
            "(int) Py_AddPendingCall("
            "(int (*)(void *)) PyRun_SimpleString, "
            f"(void *) \\\"exec(open('{snippet_path}').read(), {{}})\\\")"
        )
        rc = _mi_eval(gdbmi, c_call)
        if rc != "0":
            raise GdbAttachError(f"Py_AddPendingCall failed to enqueue (rc={rc})")
        gdbmi.write("-target-detach", timeout_sec=5)
    finally:
        gdbmi.exit()


def schedule(
    pid: int, python_snippet: str, *, max_attempts: int = 20, gdb_bin: str = "gdb"
) -> Path:
    """Attaches to `pid`, enqueues `python_snippet` via `Py_AddPendingCall`, detaches immediately -- the snippet itself runs later, on the target's own schedule.

    Retries on `GdbAbandonedEvaluationError` (see that class's docstring); each
    retry re-attaches fresh rather than reusing the aborted session.

    Args:
        pid: Target process id. Must already be the real interpreter process, not
            a launcher wrapper (e.g. `uv run python ...` does not execve -- resolve
            the target's own `os.getpid()`-reported value instead, such as
            trainctl's `RankInfo.pid`).
        python_snippet: Python source executed in a fresh, isolated globals dict
            (`exec(source, {})`) inside the target's own interpreter -- never in
            `__main__`'s real globals, which can silently clobber the target's own
            top-level variables of the same name.
        max_attempts: Retry budget for abandoned evaluations.
        gdb_bin: `gdb` executable to invoke.

    Returns:
        The temp file path the snippet writes its own output to once serviced.

    Raises:
        GdbUnavailableError: `gdb_bin` is missing or unusable.
        GdbAttachError: every attempt failed for a non-retryable reason, or the
            retry budget was exhausted.
    """
    check_gdb_available(gdb_bin)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(python_snippet)
        snippet_path = Path(f.name)

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            _attempt_schedule(pid, snippet_path, gdb_bin)
            return snippet_path
        except GdbAbandonedEvaluationError as exc:
            last_error = exc
            time.sleep(0.1 * attempt)
    raise GdbAttachError(f"gave up after {max_attempts} attempts: {last_error}")


def schedule_and_wait(
    pid: int,
    python_snippet: str,
    *,
    poll_timeout_s: float = 10.0,
    max_attempts: int = 20,
    gdb_bin: str = "gdb",
) -> dict[str, Any]:
    """`schedule()` plus polling for the snippet's own JSON report to appear.

    `python_snippet` must write a JSON report to the path given by the literal
    substring `{out_path}` (used as a Python expression, e.g. `open({out_path},
    "w")`), which this function replaces with a quoted string literal before
    scheduling. A plain substring replacement is used (not `str.format`) so the
    snippet is free to contain its own literal `{`/`}` (dict/set literals,
    f-strings, ...) without colliding with the placeholder.

    Raises:
        TimeoutError: the target's eval loop never serviced the pending call
            within `poll_timeout_s` -- e.g. it is genuinely stuck in a call that
            never returns control to the interpreter.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = Path(f.name)
    out_path.unlink()
    snippet_path = schedule(
        pid,
        python_snippet.replace("{out_path}", repr(str(out_path))),
        max_attempts=max_attempts,
        gdb_bin=gdb_bin,
    )
    deadline = time.monotonic() + poll_timeout_s
    try:
        while time.monotonic() < deadline:
            if out_path.exists():
                return dict(json.loads(out_path.read_text(encoding="utf-8")))
            time.sleep(0.05)
        raise TimeoutError(
            f"pid {pid} never serviced the pending call within {poll_timeout_s}s "
            "-- it may be stuck in a call that never returns to the interpreter"
        )
    finally:
        snippet_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def capture(
    pid: int,
    python_snippet: str,
    *,
    poll_timeout_s: float = 10.0,
    max_attempts: int = 20,
    gdb_bin: str = "gdb",
    hold_client: HoldClient | None = None,
    hold_command_id: str | None = None,
) -> dict[str, Any]:
    """Runs `python_snippet` inside `pid` via gdb call-injection; peer of `pyspy_capture.capture`.

    When `hold_client`/`hold_command_id` are given, refuses to attach unless that
    hold is confirmed currently active (see `HoldClient.require_hold_active`).
    This does not eliminate the residual deadlock risk documented at module
    level -- it only converts it from unrecoverable to recoverable/retryable.
    """
    if hold_client is not None and hold_command_id is not None:
        hold_client.require_hold_active(hold_command_id)
    return schedule_and_wait(
        pid,
        python_snippet,
        poll_timeout_s=poll_timeout_s,
        max_attempts=max_attempts,
        gdb_bin=gdb_bin,
    )
