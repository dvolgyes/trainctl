"""Behavioral tests for LoggingSession's standard-logging bridge and run-file sink."""

import logging

import pytest
from loguru import logger

from trainctl.logging import LoggingSession, _LoguruBridgeHandler, install_root_bridge

_NAME = "lightning.pytorch"


@pytest.fixture(autouse=True)
def _isolate_bridged_logger():
    """Snapshots and restores `lightning.pytorch`'s logger state around each test.

    Prevents one test's bridge installation from leaking into the next, independent of
    whether the test under exercise closes its own session correctly.
    """
    std_logger = logging.getLogger(_NAME)
    handlers = list(std_logger.handlers)
    propagate = std_logger.propagate
    level = std_logger.level
    yield
    std_logger.handlers = handlers
    std_logger.propagate = propagate
    std_logger.setLevel(level)


def _capture_records() -> tuple[list[dict], int]:
    records: list[dict] = []
    sink_id = logger.add(lambda message: records.append(message.record), level=0)
    return records, sink_id


def test_install_detaches_existing_handler_and_forwards_records() -> None:
    std_logger = logging.getLogger(_NAME)
    marker = logging.StreamHandler()
    std_logger.addHandler(marker)

    session = LoggingSession("session-a")
    records, sink_id = _capture_records()
    try:
        session.install()
        assert marker not in std_logger.handlers
        assert std_logger.propagate is False

        std_logger.warning("hello %s", "world")

        assert len(records) == 1
        assert records[0]["message"] == "hello world"
        assert records[0]["extra"]["trainctl_session"] == "session-a"
        assert records[0]["extra"]["std_logger_name"] == _NAME
    finally:
        session.close()
        logger.remove(sink_id)


def test_close_restores_original_handler_propagate_and_level() -> None:
    std_logger = logging.getLogger(_NAME)
    marker = logging.StreamHandler()
    std_logger.addHandler(marker)
    std_logger.propagate = True
    std_logger.setLevel(logging.WARNING)

    session = LoggingSession("session-b")
    session.install()
    session.close()

    assert std_logger.handlers == [marker]
    assert std_logger.propagate is True
    assert std_logger.level == logging.WARNING


def test_close_is_idempotent() -> None:
    session = LoggingSession("session-c")
    session.install()
    session.close()
    session.close()  # must not raise or double-restore


def test_exception_info_is_forwarded_only_when_present() -> None:
    std_logger = logging.getLogger(_NAME)
    session = LoggingSession("session-d")
    records, sink_id = _capture_records()
    try:
        session.install()
        try:
            raise ValueError("inner")
        except ValueError:
            std_logger.error("failed", exc_info=True)
        std_logger.error("plain, no traceback")

        assert records[0]["exception"] is not None
        assert records[1]["exception"] is None
    finally:
        session.close()
        logger.remove(sink_id)


def test_custom_numeric_level_is_forwarded_without_a_name_lookup() -> None:
    std_logger = logging.getLogger(_NAME)
    session = LoggingSession("session-e")
    records, sink_id = _capture_records()
    try:
        session.install()
        std_logger.log(35, "custom level")
        assert records[0]["level"].no == 35
        assert records[0]["message"] == "custom level"
    finally:
        session.close()
        logger.remove(sink_id)


def test_reentrant_bridged_logging_from_a_sink_is_dropped_not_recursed() -> None:
    std_logger = logging.getLogger(_NAME)
    session = LoggingSession("session-f")
    reentrant_calls = 0

    def reentrant_sink(message) -> None:
        nonlocal reentrant_calls
        reentrant_calls += 1
        if reentrant_calls < 5:
            std_logger.warning("nested call %d", reentrant_calls)

    sink_id = logger.add(reentrant_sink, level=0)
    try:
        session.install()
        std_logger.warning("outer call")
        assert reentrant_calls == 1
    finally:
        session.close()
        logger.remove(sink_id)


def test_install_rejects_a_logger_already_bridged_by_another_session() -> None:
    first = LoggingSession("session-g")
    second = LoggingSession("session-h")
    first.install()
    try:
        with pytest.raises(RuntimeError, match="session-g"):
            second.install()
    finally:
        first.close()


def test_install_on_an_already_installed_session_is_a_noop() -> None:
    session = LoggingSession("session-i")
    session.install()
    try:
        session.install()  # must not raise, must not re-save state
        std_logger = logging.getLogger(_NAME)
        assert (
            sum(isinstance(h, _LoguruBridgeHandler) for h in std_logger.handlers) == 1
        )
    finally:
        session.close()


def test_attach_run_file_records_only_this_sessions_bound_messages(tmp_path) -> None:
    log_path = tmp_path / "trainctl.log"
    session_a = LoggingSession("session-j")
    session_b = LoggingSession("session-k")
    session_a.attach_run_file(log_path)
    try:
        session_a.bind().info("from a")
        session_b.bind().info("from b")
    finally:
        session_a.close()

    content = log_path.read_text()
    assert "from a" in content
    assert "from b" not in content


def test_attach_run_file_creates_parent_directories(tmp_path) -> None:
    log_path = tmp_path / "nested" / "dir" / "trainctl.log"
    session = LoggingSession("session-l")
    session.attach_run_file(log_path)
    try:
        session.bind().info("hello")
    finally:
        session.close()
    assert log_path.exists()
    assert "hello" in log_path.read_text()


def test_attach_run_file_propagates_a_real_open_failure(tmp_path) -> None:
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("x")
    session = LoggingSession("session-m")
    with pytest.raises(OSError):
        session.attach_run_file(blocking_file / "trainctl.log")


def test_install_root_bridge_also_bridges_the_actual_root_logger() -> None:
    root_logger = logging.getLogger()
    saved_handlers = list(root_logger.handlers)
    saved_propagate = root_logger.propagate
    session = LoggingSession("session-n")
    records, sink_id = _capture_records()
    try:
        install_root_bridge(session)
        logging.getLogger("some.unrelated.module").warning("root-routed message")
        assert any(r["message"] == "root-routed message" for r in records)
    finally:
        session.close()
        logger.remove(sink_id)
        root_logger.handlers = saved_handlers
        root_logger.propagate = saved_propagate


def test_repeated_sessions_do_not_interfere_when_closed_before_the_next_install() -> (
    None
):
    std_logger = logging.getLogger(_NAME)

    first = LoggingSession("session-o")
    first.install()
    first.close()

    second = LoggingSession("session-p")
    records, sink_id = _capture_records()
    try:
        second.install()
        std_logger.warning("second session message")
        assert len(records) == 1
        assert records[0]["extra"]["trainctl_session"] == "session-p"
    finally:
        second.close()
        logger.remove(sink_id)
