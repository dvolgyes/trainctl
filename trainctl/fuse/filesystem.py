"""Read-only mfusepy filesystem projecting a TrainctlRuntime's state onto files.

All virtual files are generated fresh from the current `RuntimeSnapshot`/runtime
object on every `open()`; the mount uses FUSE direct I/O (see `TrainctlRuntime._start_fuse`)
so the kernel never serves a stale cached read. A generated file's bytes are
materialized once per `open()` and reused for the `read()` calls against that handle
(see `TrainctlFS.open`/`read`/`release`), matching the open-handle cache policy.
Checkpoints, snapshots, and debug captures are real files on the regular filesystem
under `TrainctlRuntime.artifacts.root` (a sibling of this mount, not projected into
it) -- see `meta/artifacts-path`.

This class is a thin mfusepy adapter: what each virtual path actually contains is
owned by `trainctl.runtime.virtual_fs`, shared with the `GET /files/{path}` REST
route so the two transports never derive the same content twice.
"""

import errno
import os
import stat
import threading
import time
from typing import TYPE_CHECKING, Any

import mfusepy
from loguru import logger

from trainctl.runtime import virtual_fs

if TYPE_CHECKING:
    from trainctl.runtime.runtime import TrainctlRuntime


class TrainctlFS(mfusepy.Operations):
    """Read-only, direct-I/O mfusepy `Operations` backed by a `TrainctlRuntime`."""

    use_ns = True

    def __init__(self, runtime: "TrainctlRuntime") -> None:
        self._runtime = runtime
        self._ready = threading.Event()
        self._handles: dict[int, bytes] = {}
        self._next_fh = 1
        self._handles_lock = threading.Lock()

    def init(self, path: str) -> None:
        del path
        self._ready.set()

    def wait_ready(self, timeout: float) -> bool:
        """Blocks until libfuse has called `init()`, signalling the mount is live."""
        return self._ready.wait(timeout)

    def getattr(self, path: str, fh: int | None = None) -> dict[str, Any]:
        del fh
        owner = {
            "st_uid": os.getuid(),
            "st_gid": os.getgid(),
            "st_mtime": int(time.time() * 1e9),
        }
        if virtual_fs.is_virtual_dir(self._runtime, path):
            return {"st_mode": (stat.S_IFDIR | 0o755), "st_nlink": 2, **owner}
        if virtual_fs.is_live_file(path):
            # st_size is unknowable without running the blocking capture; never trigger
            # it just to stat() the file -- only open()/read() do that.
            return {
                "st_mode": (stat.S_IFREG | 0o444),
                "st_nlink": 1,
                "st_size": 0,
                **owner,
            }
        content = virtual_fs.generate(self._runtime, path)
        if content is None:
            raise mfusepy.FuseOSError(errno.ENOENT)
        return {
            "st_mode": (stat.S_IFREG | 0o444),
            "st_nlink": 1,
            "st_size": len(content),
            **owner,
        }

    def readdir(self, path: str, fh: int) -> list[str]:
        del fh
        return [".", "..", *virtual_fs.list_dir(self._runtime, path)]

    def open(self, path: str, flags: int) -> int:
        del flags
        try:
            content = virtual_fs.generate(self._runtime, path)
        except TimeoutError as exc:
            logger.warning("trainctl FUSE read of {} timed out: {}", path, exc)
            raise mfusepy.FuseOSError(errno.ETIMEDOUT) from exc
        except Exception as exc:
            logger.warning("trainctl FUSE read of {} failed: {}", path, exc)
            raise mfusepy.FuseOSError(errno.EIO) from exc
        if content is None:
            raise mfusepy.FuseOSError(errno.ENOENT)
        with self._handles_lock:
            handle = self._next_fh
            self._next_fh += 1
            self._handles[handle] = content
        return handle

    def read(self, path: str, size: int, offset: int, fh: int) -> bytes:
        del path
        with self._handles_lock:
            content = self._handles.get(fh, b"")
        return content[offset : offset + size]

    def release(self, path: str, fh: int) -> int:
        del path
        with self._handles_lock:
            self._handles.pop(fh, None)
        return 0

    def statfs(self, path: str) -> dict[str, int]:
        del path
        return {}
