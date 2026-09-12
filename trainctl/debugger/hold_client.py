"""Thin stdlib HTTP client for a target training process's Trainctl REST API.

Used by external debugger tooling to arm/confirm/release cooperative holds and to
persist capture results, without importing anything from `trainctl.runtime`.
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any


class HoldClientError(RuntimeError):
    """A REST call to the target's trainctl process failed."""


class HoldNotActiveError(RuntimeError):
    """The hold a capture was gated on is not currently the active hold."""


class HoldClient:
    """Talks to one training process's Trainctl REST API over HTTP.

    Attributes:
        base_url: REST base URL, e.g. `http://127.0.0.1:8090`.
        timeout: Per-request timeout in seconds.
    """

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        request = urllib.request.Request(  # noqa: S310 -- operator-specified trainctl REST endpoint
            url, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                return dict(json.loads(response.read()))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise HoldClientError(
                f"{method} {path} failed: HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise HoldClientError(f"{method} {path} failed: {exc}") from exc

    def arm_hold(
        self, when: Any = "now", reason: str = "", timeout_s: float = 5.0
    ) -> dict[str, Any]:
        """Arms a hold; see `POST /holds` for the bounded-wait/`202`-fallback semantics."""
        return self._request(
            "POST", "/holds", {"when": when, "reason": reason, "timeout_s": timeout_s}
        )

    def get_holds(self) -> dict[str, Any]:
        """Returns `{"holds": [...], "active": {...} | None}`."""
        return self._request("GET", "/holds")

    def get_command(self, command_id: str) -> dict[str, Any]:
        """Returns one command's current status/result."""
        return self._request("GET", f"/commands/{command_id}")

    def release_hold(self, command_id: str) -> dict[str, Any]:
        """Releases the active hold armed by `command_id`."""
        return self._request("POST", f"/holds/{command_id}/release")

    def push_capture(
        self, kind: str, files: dict[str, str], metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Persists a debug capture via `POST /debug/captures`; returns its manifest."""
        return self._request(
            "POST",
            "/debug/captures",
            {"kind": kind, "files": files, "metadata": metadata or {}},
        )

    def wait_for_hold(
        self, command_id: str, timeout_s: float, poll_interval_s: float = 0.2
    ) -> dict[str, Any]:
        """Polls `GET /commands/{command_id}` until it leaves `queued`, or `timeout_s` elapses.

        Complements `arm_hold`'s own server-side bounded wait, for a caller that wants
        to keep waiting past that window -- e.g. an infrequent safe point such as
        `train_epoch_end` -- rather than treating a `202` response as final.
        """
        deadline = time.monotonic() + timeout_s
        command = self.get_command(command_id)
        while command.get("status") == "queued" and time.monotonic() < deadline:
            time.sleep(poll_interval_s)
            command = self.get_command(command_id)
        return command

    def require_hold_active(self, command_id: str) -> None:
        """Raises `HoldNotActiveError` unless `command_id` is the currently active hold.

        Re-checks live state rather than trusting a previously-observed `succeeded`
        status, since a hold can be released between when it engaged and when a
        capture actually attaches.
        """
        command = self.get_command(command_id)
        if command.get("status") != "succeeded":
            raise HoldNotActiveError(
                f"hold {command_id!r} has not engaged (status={command.get('status')!r})"
            )
        active = self.get_holds().get("active")
        if active is None or active.get("command_id") != command_id:
            raise HoldNotActiveError(
                f"hold {command_id!r} is not the currently active hold"
            )
