"""Behavioral tests for the diagnostics-milestone static FUSE paths.

Exercises `virtual_fs.generate`/`is_virtual_dir` directly against a `TrainctlRuntime`,
matching the existing static-path test convention -- no real FUSE mount involved.
"""

import json

from trainctl.config import TrainctlConfig
from trainctl.fuse.filesystem import TrainctlFS
from trainctl.lightning_backend import load_backend
from trainctl.runtime import virtual_fs
from trainctl.runtime.runtime import TrainctlRuntime
from trainctl.runtime.state import ExceptionBreakpointInfo


def _make_fs(**overrides: object) -> TrainctlFS:
    config = TrainctlConfig(
        rest_enabled=False, fuse_enabled=False, torch_debug_enabled=False, **overrides
    )
    runtime = TrainctlRuntime(config, load_backend("lightning.pytorch"))
    return TrainctlFS(runtime)


def test_breakpoint_json_is_null_when_no_exception_held() -> None:
    fs = _make_fs()
    assert json.loads(virtual_fs.generate(fs._runtime, "/state/breakpoint.json")) is None


def test_breakpoint_json_reports_active_exception() -> None:
    fs = _make_fs()
    info = ExceptionBreakpointInfo(
        exception_type="ValueError",
        exception_message="boom",
        epoch=0,
        global_step=1,
        batch_idx=2,
        rank=0,
        started_at=123.0,
    )
    fs._runtime.state.update(status="exception_held", exception=info)

    payload = json.loads(virtual_fs.generate(fs._runtime, "/state/breakpoint.json"))

    assert payload["exception_type"] == "ValueError"
    assert payload["exception_message"] == "boom"


def test_dataloader_finite_check_json_reflects_guard_state() -> None:
    fs = _make_fs(dataloader_finite_check_enabled=True)
    payload = json.loads(
        virtual_fs.generate(fs._runtime, "/state/dataloader/finite-check.json")
    )
    assert payload == {"enabled": True, "checked_batches": 0, "failures": 0}
    assert virtual_fs.is_virtual_dir(fs._runtime, "/state/dataloader")


def test_pipeline_config_json_reflects_config() -> None:
    fs = _make_fs(pipeline_pressure_enabled=True, pipeline_pressure_window=64)
    payload = json.loads(virtual_fs.generate(fs._runtime, "/state/pipeline/config.json"))
    assert payload == {"enabled": True, "window": 64}
    assert virtual_fs.is_virtual_dir(fs._runtime, "/state/pipeline")


def test_pipeline_latest_json_is_null_before_first_batch() -> None:
    fs = _make_fs()
    assert json.loads(virtual_fs.generate(fs._runtime, "/state/pipeline/latest.json")) is None


def test_pipeline_latest_and_summary_json_after_a_batch() -> None:
    fs = _make_fs()
    fs._runtime.pipeline.batch_start(0, epoch=0, global_step=0, batch_idx=0)
    fs._runtime.pipeline.batch_end(1_000)

    latest = json.loads(virtual_fs.generate(fs._runtime, "/state/pipeline/latest.json"))
    assert latest["batch_total_ns"] == 1_000

    summary = json.loads(virtual_fs.generate(fs._runtime, "/state/pipeline/summary.json"))
    assert summary["samples"] == 1
