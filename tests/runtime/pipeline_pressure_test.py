"""Behavioral tests for PipelinePressureTracker's phase timing and rolling summary."""

from trainctl.runtime.pipeline_pressure import PipelinePressureTracker


def test_first_batch_has_no_input_gap() -> None:
    tracker = PipelinePressureTracker(window=8)

    tracker.batch_start(1_000, epoch=0, global_step=0, batch_idx=0)
    sample = tracker.batch_end(2_000)

    assert sample.input_gap_ns is None
    assert sample.batch_total_ns == 1_000


def test_second_batch_records_input_gap_from_previous_batch_end() -> None:
    tracker = PipelinePressureTracker(window=8)
    tracker.batch_start(1_000, epoch=0, global_step=0, batch_idx=0)
    tracker.batch_end(2_000)

    tracker.batch_start(2_500, epoch=0, global_step=1, batch_idx=1)
    sample = tracker.batch_end(3_000)

    assert sample.input_gap_ns == 500


def test_full_phase_breakdown_recorded_in_order() -> None:
    tracker = PipelinePressureTracker(window=8)
    tracker.batch_start(0, epoch=0, global_step=0, batch_idx=0)
    tracker.before_backward(100)
    tracker.after_backward(150)
    tracker.before_optimizer(150)
    tracker.optimizer_step(180)
    sample = tracker.batch_end(200)

    assert sample.forward_loss_ns == 100
    assert sample.backward_ns == 50
    assert sample.post_backward_ns == 50
    assert sample.optimizer_step_ns == 30
    assert sample.batch_total_ns == 200


def test_missing_backward_phases_are_none() -> None:
    tracker = PipelinePressureTracker(window=8)
    tracker.batch_start(0, epoch=0, global_step=0, batch_idx=0)
    sample = tracker.batch_end(100)

    assert sample.forward_loss_ns is None
    assert sample.backward_ns is None
    assert sample.post_backward_ns is None
    assert sample.optimizer_step_ns is None


def test_window_evicts_oldest_samples() -> None:
    tracker = PipelinePressureTracker(window=2)
    for i in range(5):
        tracker.batch_start(i * 100, epoch=0, global_step=i, batch_idx=i)
        tracker.batch_end(i * 100 + 10)

    summary = tracker.summary()
    assert summary["samples"] == 2
    assert summary["latest_step"] == 4


def test_latest_returns_none_before_first_batch() -> None:
    tracker = PipelinePressureTracker(window=8)
    assert tracker.latest() is None


def test_summary_before_any_batch_reports_zero_samples() -> None:
    tracker = PipelinePressureTracker(window=8)
    summary = tracker.summary()
    assert summary == {"window": 8, "samples": 0}


def test_summary_identifies_largest_phase() -> None:
    tracker = PipelinePressureTracker(window=8)
    tracker.batch_start(0, epoch=0, global_step=0, batch_idx=0)
    tracker.before_backward(10)
    tracker.after_backward(1_000)
    sample = tracker.batch_end(1_010)

    assert sample.backward_ns == 990
    summary = tracker.summary()
    assert summary["largest_phase"] == "backward"
