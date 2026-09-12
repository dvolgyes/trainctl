"""Behavioral tests for trainctl.hooks.payload's light-hook path (T2)."""

import json

import torch

from trainctl.hooks.payload import (
    OccurrenceContext,
    build_light_params,
    describe_arguments,
    new_id,
    write_json_file,
)


def _context(**overrides) -> OccurrenceContext:
    fields = {
        "session_id": "s-1",
        "hook": "on_train_batch_start",
        "stage": "fit",
        "epoch": 0,
        "global_step": 0,
        "batch_idx": 0,
        "dataloader_idx": None,
        "rank": 0,
        "world_size": 1,
    }
    fields.update(overrides)
    return OccurrenceContext(**fields)


def test_primitives_pass_through_unchanged() -> None:
    described, truncated, reason = describe_arguments(
        {"a": None, "b": True, "c": 3, "d": 3.5, "e": "short"},
        max_items=10,
        max_depth=4,
    )
    assert described == {"a": None, "b": True, "c": 3, "d": 3.5, "e": "short"}
    assert truncated is False
    assert reason is None


def test_tensor_is_described_but_never_evaluated(monkeypatch) -> None:
    def _forbidden(self, *args, **kwargs):
        raise AssertionError("tensor value was evaluated on the light path")

    monkeypatch.setattr(torch.Tensor, "item", _forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", _forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", _forbidden)

    tensor = torch.zeros(2, 3, dtype=torch.float32, requires_grad=True)
    described, truncated, _reason = describe_arguments(
        {"batch": tensor}, max_items=10, max_depth=4
    )

    assert described["batch"] == {
        "kind": "tensor",
        "shape": [2, 3],
        "dtype": "torch.float32",
        "device": "cpu",
        "requires_grad": True,
    }
    assert truncated is False


def test_long_string_is_truncated_with_reason() -> None:
    long_value = "x" * 500
    described, truncated, reason = describe_arguments(
        {"note": long_value}, max_items=10, max_depth=4
    )
    assert described["note"].endswith("...(truncated)")
    assert len(described["note"]) < len(long_value)
    assert truncated is True
    assert reason is not None
    assert "string_length" in reason


def test_nested_dict_and_list_traversal_within_bounds() -> None:
    arguments = {
        "batch": {
            "inputs": [1, 2, {"deep": "value"}],
            "meta": {"nested": [3, 4]},
        }
    }
    described, truncated, reason = describe_arguments(
        arguments, max_items=10, max_depth=4
    )
    assert described == {
        "batch": {
            "inputs": [1, 2, {"deep": "value"}],
            "meta": {"nested": [3, 4]},
        }
    }
    assert truncated is False
    assert reason is None


def test_max_items_truncates_dict_and_marks_omitted_count() -> None:
    arguments = {"batch": {str(i): i for i in range(5)}}
    described, truncated, reason = describe_arguments(
        arguments, max_items=3, max_depth=4
    )
    described_dict = described["batch"]
    assert len(described_dict) == 4  # 3 kept + the "__truncated__" marker
    assert described_dict["__truncated__"] == "2 more item(s) omitted"
    assert truncated is True
    assert reason is not None
    assert "max_items" in reason


def test_max_items_truncates_list_and_appends_truncated_marker() -> None:
    arguments = {"batch": list(range(5))}
    described, truncated, reason = describe_arguments(
        arguments, max_items=3, max_depth=4
    )
    described_list = described["batch"]
    assert described_list[:3] == [0, 1, 2]
    assert described_list[3] == {
        "kind": "truncated",
        "reason": "2 more item(s) omitted",
    }
    assert truncated is True
    assert reason is not None
    assert "max_items" in reason


def test_max_depth_truncates_at_cutoff() -> None:
    arguments = {"batch": {"a": {"b": {"c": "too deep"}}}}
    described, truncated, reason = describe_arguments(
        arguments, max_items=10, max_depth=2
    )
    assert described["batch"]["a"]["b"] == {"kind": "truncated", "reason": "max_depth"}
    assert truncated is True
    assert reason is not None
    assert "max_depth" in reason


def test_cycle_is_detected_and_does_not_recurse_forever() -> None:
    cyclic: dict = {}
    cyclic["self"] = cyclic
    described, truncated, _reason = describe_arguments(
        {"batch": cyclic}, max_items=10, max_depth=50
    )
    assert described["batch"]["self"] == {"kind": "cycle"}
    assert truncated is False


class _Widget:
    pass


def test_unknown_object_is_described_by_kind_and_type() -> None:
    described, _truncated, _reason = describe_arguments(
        {"thing": _Widget()}, max_items=10, max_depth=4
    )
    assert described["thing"] == {"kind": "object", "type": "_Widget"}


def test_build_light_params_assembles_all_schema_fields() -> None:
    context = _context(hook="on_train_batch_end", stage="fit", epoch=2, global_step=17)
    params, truncated, reason = build_light_params(
        context=context,
        occurrence_id="occ-1",
        invocation_id="inv-1",
        arguments={"loss": 1.5},
        max_params_bytes=1_000_000,
        max_metadata_items=10,
        max_metadata_depth=4,
    )
    assert params["schema_version"] == 1
    assert params["session_id"] == "s-1"
    assert params["occurrence_id"] == "occ-1"
    assert params["invocation_id"] == "inv-1"
    assert params["hook"] == "on_train_batch_end"
    assert params["modality"] == "light"
    assert isinstance(params["timestamp"], float)
    assert isinstance(params["pid"], int)
    assert params["rank"] == 0
    assert params["world_size"] == 1
    assert params["stage"] == "fit"
    assert params["epoch"] == 2
    assert params["global_step"] == 17
    assert params["batch_idx"] == 0
    assert params["dataloader_idx"] is None
    assert params["arguments"] == {"loss": 1.5}
    assert truncated is False
    assert reason is None


def test_build_light_params_drops_arguments_entirely_when_over_byte_budget() -> None:
    context = _context()
    params, truncated, reason = build_light_params(
        context=context,
        occurrence_id="occ-1",
        invocation_id="inv-1",
        arguments={"payload": "x" * 1000},
        max_params_bytes=10,
        max_metadata_items=10,
        max_metadata_depth=4,
    )
    assert params["arguments"] is None
    assert truncated is True
    assert reason is not None
    assert "max_params_bytes" in reason


def test_new_id_returns_distinct_hex_strings() -> None:
    first = new_id()
    second = new_id()
    assert first != second
    assert bytes.fromhex(first)
    assert bytes.fromhex(second)


def test_write_json_file_writes_reloadable_json_and_returns_path(tmp_path) -> None:
    target = tmp_path / "params.json"
    document = {"hello": "world", "n": 3}
    returned = write_json_file(target, document)
    assert returned == target
    assert json.loads(target.read_text()) == document
