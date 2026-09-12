"""Request bodies accepted by the Trainctl REST API."""

from typing import Any

from pydantic import BaseModel


class ExecutionSpec(BaseModel):
    """When a `/commands` submission should become due. See `trainctl.runtime.commands.parse_when`."""

    when: Any = "now"


class CommandIn(BaseModel):
    """Body for `POST /commands`."""

    kind: str
    args: dict[str, Any] = {}
    execution: ExecutionSpec = ExecutionSpec()


class HoldIn(BaseModel):
    """Body for `POST /holds`.

    Attributes:
        timeout_s: How long the request blocks waiting for the hold to actually
            engage before falling back to a `202` pollable response. `0` skips the
            wait and returns the async response immediately.
    """

    when: Any = "now"
    reason: str = ""
    timeout_s: float = 5.0


class CheckpointIn(BaseModel):
    """Body for `POST /checkpoints`."""

    when: Any = "now"
    label: str = "manual"


class SnapshotIn(BaseModel):
    """Body for `POST /snapshots`."""

    when: Any = "now"
    kind: str = "runtime"


class DebugCaptureIn(BaseModel):
    """Body for `POST /debug/captures`. Persists a capture taken by an external debugger tool."""

    kind: str
    files: dict[str, str] = {}
    metadata: dict[str, Any] = {}
