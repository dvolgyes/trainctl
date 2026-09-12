"""Behavioral tests for the shared optional-inspector availability registry."""

from trainctl.runtime import inspectors_registry


def test_availability_snapshot_reports_all_four_inspectors() -> None:
    snapshot = inspectors_registry.availability_snapshot()
    assert set(snapshot) == {"torchinfo", "torchview", "torchviz", "torchlens"}
    for entry in snapshot.values():
        assert isinstance(entry["available"], bool)
