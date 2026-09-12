"""Behavioral tests for `graph_inspectors.torchview_dot`.

Unlike `torchview_capture`, this never renders, so it needs only the `torchview`
Python package -- deliberately not skipped by the `shutil.which("dot") is None` guard
that gates `graph_inspectors_test.py`'s render-based tests.
"""

import sys

import pytest
import torch

from trainctl.runtime import graph_inspectors


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_torchview_dot_returns_source_without_dot_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    source = graph_inspectors.torchview_dot(_Model(), torch.randn(3, 4))
    assert "digraph" in source


def test_torchview_dot_raises_without_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torchview", None)
    with pytest.raises(ImportError):
        graph_inspectors.torchview_dot(_Model(), torch.randn(3, 4))
