"""Safe, read-only capture mode using py-spy.

py-spy's dump mode pauses the target just long enough to read its memory and
reconstruct Python frames -- it never makes the target execute injected code, so
it is structurally immune to the CUDA/autograd deadlock class documented in
`gdb_capture`'s module docstring. Peer of `gdb_capture.capture`: equally
supported, not a "safe default" for a "risky opt-in" -- reach for this whenever a
read-only stack/locals dump is enough, and for `gdb_capture` when you need to
mutate live state or call an arbitrary live function instead.
"""

import json
import subprocess
from typing import Any

from trainctl.debugger.hold_client import HoldClient


class PySpyError(RuntimeError):
    """py-spy failed to produce a stack dump."""


def capture(
    pid: int,
    *,
    locals_verbosity: int = 0,
    native: bool = False,
    nonblocking: bool = False,
    subprocesses: bool = False,
    py_spy_bin: str = "py-spy",
    hold_client: HoldClient | None = None,
    hold_command_id: str | None = None,
) -> dict[str, Any]:
    """Captures a stack dump of `pid` via `py-spy dump --json`.

    Args:
        pid: Target process id.
        locals_verbosity: 0 for no locals, 1 for `-l`, 2+ for `-ll` (more detail).
        native: Include native (C/C++/Cython) frames.
        nonblocking: Read memory without pausing the target -- less consistent, but
            removes even the brief pause py-spy's default mode uses.
        subprocesses: Also profile subprocesses of `pid`.
        py_spy_bin: `py-spy` executable to invoke, resolved via `PATH`.
        hold_client: When given with `hold_command_id`, refuses to run unless that
            hold is confirmed currently active (see `HoldClient.require_hold_active`).
            Optional here (unlike `gdb_capture`) since a read-only dump carries no
            deadlock risk of its own; still recommended for the coherent-collective
            reasons in the Chapter 2 design doc's "cooperative attach" workflow.
        hold_command_id: The hold command id to check when `hold_client` is given.

    Returns:
        `{"pid": pid, "threads": [...]}` -- `threads` is py-spy's own per-thread dump
        list (each entry has `thread_id`, `active`, `owns_gil`, `frames`, ...).

    Raises:
        PySpyError: py-spy exited non-zero or produced unparsable output.
        HoldNotActiveError: a hold check was requested but the hold isn't active.
    """
    if hold_client is not None and hold_command_id is not None:
        hold_client.require_hold_active(hold_command_id)
    argv = [py_spy_bin, "dump", "--pid", str(pid), "--json"]
    argv.extend(["-l"] * max(locals_verbosity, 0))
    if native:
        argv.append("--native")
    if nonblocking:
        argv.append("--nonblocking")
    if subprocesses:
        argv.append("--subprocesses")
    result = subprocess.run(  # noqa: S603 -- operator-specified py-spy binary and pid
        argv, capture_output=True, text=True, timeout=30, check=False
    )
    if result.returncode != 0:
        raise PySpyError(
            f"py-spy dump failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    try:
        threads = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PySpyError(f"py-spy dump produced unparsable output: {exc}") from exc
    return {"pid": pid, "threads": threads}
