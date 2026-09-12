"""Bridges Lightning's standard-library logging into Loguru for one training session.

Bridges only the namespaces trainctl itself cares about (`lightning.pytorch`,
`pytorch_lightning`, `lightning.fabric`); it never touches the root logger unless
`install_root_bridge` is called explicitly. Only trainctl code that binds a session's
logger (`LoggingSession.bind()`) is routed into that session's run-file sink --
pre-existing, unbound `trainctl.*` diagnostics are unaffected by this module.
"""

import logging
import threading
from pathlib import Path
from typing import Any

from loguru import logger

_BRIDGED_LOGGERS: tuple[str, ...] = (
    "lightning.pytorch",
    "pytorch_lightning",
    "lightning.fabric",
)


def _resolve_logger(name: str) -> logging.Logger:
    """Resolves `name` to a standard-library logger, special-casing `"root"`.

    `logging.getLogger("root")` is *not* the actual root logger (it is an ordinary
    child logger named `"root"`); only `logging.getLogger()` with no argument returns
    it. `"root"` is this module's sentinel name for that logger, matching the same
    convention used by `logging.config`'s dictConfig schema.
    """
    return logging.getLogger() if name == "root" else logging.getLogger(name)


class _SavedLoggerState:
    """A standard-library logger's handlers/propagate/level, saved for restoration."""

    def __init__(self, std_logger: logging.Logger) -> None:
        self.std_logger = std_logger
        self.handlers = list(std_logger.handlers)
        self.propagate = std_logger.propagate
        self.level = std_logger.level

    def restore(self) -> None:
        """Reinstates the exact handlers/propagate/level captured at save time."""
        self.std_logger.handlers = list(self.handlers)
        self.std_logger.propagate = self.propagate
        self.std_logger.setLevel(self.level)


class _LoguruBridgeHandler(logging.Handler):
    """Forwards one standard-library logger's records into Loguru as one session.

    Guards against reentrancy: a Loguru sink that itself logs through a bridged
    standard-library logger would otherwise recurse forever; the guard drops that
    nested call instead of forwarding it.

    Attributes:
        session_id: The owning `LoggingSession`'s identity, used to detect an
            overlapping install onto an already-bridged logger.
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(level=logging.NOTSET)
        self.session_id = session_id
        self._guard = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._guard, "active", False):
            return
        self._guard.active = True
        try:
            bound = logger.bind(
                trainctl_session=self.session_id,
                std_logger_name=record.name,
                std_pathname=record.pathname,
                std_lineno=record.lineno,
                std_funcname=record.funcName,
            )
            bound.opt(exception=record.exc_info).log(
                record.levelno, record.getMessage()
            )
        finally:
            self._guard.active = False


class LoggingSession:
    """Owns one training run's standard-logging bridge and its Loguru run-file sink.

    Detaches the bridged loggers' own handlers so nothing double-logs, forwards their
    records into Loguru bound with `session_id`, and (optionally) attaches a Loguru file
    sink filtered to that same `session_id` -- so a later, unrelated session in the same
    process cannot leak into this run's file. `install`/`close` are each idempotent.

    `install` never changes a bridged logger's own level: Python's logging module
    computes a message's effective threshold (walking to the nearest ancestor with a
    set level) independently of `propagate`, so a bridged logger still needs its own
    level set below a message's severity for that message to reach this bridge at all.
    Lightning sets its own loggers to `INFO` at import time; this class respects that
    policy rather than overriding it.

    Attributes:
        session_id: This session's identity, bound onto every forwarded record and the
            run-file sink's filter.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._saved: list[_SavedLoggerState] = []
        self._handler: _LoguruBridgeHandler | None = None
        self._file_sink_id: int | None = None

    def install(self, logger_names: tuple[str, ...] = _BRIDGED_LOGGERS) -> None:
        """Detaches `logger_names`' own handlers and installs the Loguru bridge.

        A second call on an already-installed session is a no-op. Installing onto a
        logger already bridged by a *different* session is rejected outright --
        detected from the handler instance already present, not a separate registry --
        so overlapping sessions can never silently shadow one another.

        Args:
            logger_names: Standard-library logger names to bridge; `"root"` refers to
                the actual root logger (see `_resolve_logger`).

        Raises:
            RuntimeError: A logger in `logger_names` is already bridged by a different,
                still-open `LoggingSession`.
        """
        if self._handler is not None:
            return
        for name in logger_names:
            for existing in _resolve_logger(name).handlers:
                if isinstance(existing, _LoguruBridgeHandler):
                    raise RuntimeError(
                        f"logger {name!r} is already bridged by session "
                        f"{existing.session_id!r}; close that session first"
                    )
        handler = _LoguruBridgeHandler(self.session_id)
        for name in logger_names:
            std_logger = _resolve_logger(name)
            self._saved.append(_SavedLoggerState(std_logger))
            for existing in list(std_logger.handlers):
                std_logger.removeHandler(existing)
            std_logger.addHandler(handler)
            std_logger.propagate = False
        self._handler = handler

    def attach_run_file(self, path: Path) -> None:
        """Adds a Loguru file sink at `path`, filtered to this session's own records.

        A second call is a no-op. I/O failures (e.g. an unwritable parent directory)
        propagate as-is from `Path.mkdir`/`loguru.logger.add` -- the caller decides
        whether that is fatal.
        """
        if self._file_sink_id is not None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        session_id = self.session_id
        self._file_sink_id = logger.add(
            str(path),
            filter=lambda record: record["extra"].get("trainctl_session") == session_id,
            level="DEBUG",
            diagnose=False,
            backtrace=False,
        )

    def bind(self) -> Any:
        """Returns a Loguru logger bound to this session's id.

        Trainctl code that wants its own diagnostics routed into this session's run
        file (`attach_run_file`) must log through this bound logger rather than the
        bare module-level `loguru.logger`.
        """
        return logger.bind(trainctl_session=self.session_id)

    def close(self) -> None:
        """Restores every intercepted logger and removes the run-file sink.

        Idempotent -- safe to call from both the normal-teardown and the
        `on_exception` paths without double-restoring or erroring on a second call.
        """
        for saved in self._saved:
            saved.restore()
        self._saved = []
        self._handler = None
        if self._file_sink_id is not None:
            logger.remove(self._file_sink_id)
            self._file_sink_id = None


def install_root_bridge(session: LoggingSession) -> None:
    """Installs `session`'s bridge including the actual root logger.

    For applications that want *all* standard-library logging -- not just Lightning's
    namespaces -- unified into the same run file. Call this instead of
    `session.install()`, not in addition to it: `install` is a no-op once a session
    already holds a handler, so a later call would silently skip adding root.
    """
    session.install(logger_names=(*_BRIDGED_LOGGERS, "root"))
