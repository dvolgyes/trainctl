"""Behavioral tests for trainctl.hooks.payload's light-hook path (T2) and heavy-hook
export path (T3)."""

import json

import numpy
import torch

from trainctl.hooks.payload import (
    HeavyCaptureFailure,
    OccurrenceContext,
    build_heavy_manifest,
    build_light_params,
    collect_tensor_leaves,
    describe_arguments,
    export_tensors,
    new_id,
    prepare_heavy_export,
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


# --- Heavy-hook export path (I3, T3) ---


def test_collect_tensor_leaves_only_walks_named_tensor_args() -> None:
    tensor_a = torch.zeros(2)
    tensor_b = torch.ones(3)
    tensor_unrelated = torch.full((4,), 7.0)
    arguments = {
        "batch": {"images": [tensor_a, tensor_b]},
        "unrelated": tensor_unrelated,
    }
    leaves, truncated, reason = collect_tensor_leaves(
        arguments, ("batch",), max_items=10, max_depth=4
    )
    paths = {leaf.path for leaf in leaves}
    assert paths == {"batch/images/0", "batch/images/1"}
    assert truncated is False
    assert reason is None


def test_collect_tensor_leaves_respects_bounds_and_is_cycle_safe() -> None:
    cyclic: dict = {}
    cyclic["self"] = cyclic
    cyclic["t"] = torch.zeros(1)
    leaves, truncated, _reason = collect_tensor_leaves(
        {"batch": cyclic}, ("batch",), max_items=10, max_depth=50
    )
    assert isinstance(leaves, list)  # returned without RecursionError or hanging

    wide = {"batch": {str(i): torch.zeros(1) for i in range(10)}}
    leaves, truncated, reason = collect_tensor_leaves(
        wide, ("batch",), max_items=3, max_depth=4
    )
    assert len(leaves) == 3
    assert truncated is True
    assert "max_items" in reason

    deep = {"batch": {"a": {"b": {"c": torch.zeros(1)}}}}
    leaves, truncated, reason = collect_tensor_leaves(
        deep, ("batch",), max_items=10, max_depth=2
    )
    assert leaves == []
    assert truncated is True
    assert "max_depth" in reason


def test_prepare_heavy_export_succeeds_empty_when_argument_absent_or_no_tensor_args() -> (
    None
):
    leaves, truncated, reason = prepare_heavy_export(
        {"other": "value"}, ("batch",), max_items=10, max_depth=4, max_export_bytes=1024
    )
    assert leaves == []
    assert truncated is False
    assert reason is None

    leaves, truncated, reason = prepare_heavy_export(
        {"batch": torch.zeros(4)}, (), max_items=10, max_depth=4, max_export_bytes=1024
    )
    assert leaves == []
    assert truncated is False
    assert reason is None


def test_prepare_heavy_export_fails_on_unsupported_layout() -> None:
    sparse = torch.zeros(4, 4).to_sparse()
    result = prepare_heavy_export(
        {"batch": sparse},
        ("batch",),
        max_items=10,
        max_depth=4,
        max_export_bytes=1024**3,
    )
    assert isinstance(result, HeavyCaptureFailure)
    assert "batch" in result.reason


def test_prepare_heavy_export_fails_when_predicted_size_exceeds_budget() -> None:
    big = torch.zeros(1000, 1000, dtype=torch.float32)
    result = prepare_heavy_export(
        {"batch": big}, ("batch",), max_items=10, max_depth=4, max_export_bytes=100
    )
    assert isinstance(result, HeavyCaptureFailure)
    assert "100" in result.reason
    assert str(1000 * 1000 * 4) in result.reason


def test_prepare_heavy_export_succeeds_for_bfloat16() -> None:
    tensor = torch.ones(2, 2, dtype=torch.bfloat16)
    leaves, truncated, reason = prepare_heavy_export(
        {"batch": tensor},
        ("batch",),
        max_items=10,
        max_depth=4,
        max_export_bytes=1024**3,
    )
    assert len(leaves) == 1
    assert truncated is False
    assert reason is None


def test_export_tensors_converts_bfloat16_to_float32(tmp_path) -> None:
    tensor = torch.full((2, 2), 1.5, dtype=torch.bfloat16)
    leaves, _truncated, _reason = prepare_heavy_export(
        {"batch": tensor},
        ("batch",),
        max_items=10,
        max_depth=4,
        max_export_bytes=1024**3,
    )
    records = export_tensors(leaves, tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record.dtype == "float32"
    assert record.original_dtype == "torch.bfloat16"

    array = numpy.load(tmp_path / record.file, allow_pickle=False)
    assert array.dtype == numpy.float32
    assert numpy.allclose(array, 1.5, atol=0.05)


def test_export_tensors_float32_exact_roundtrip_and_sequential_filenames(
    tmp_path,
) -> None:
    tensor_a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tensor_b = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    leaves, _truncated, _reason = prepare_heavy_export(
        {"batch": {"first": tensor_a, "second": tensor_b}},
        ("batch",),
        max_items=10,
        max_depth=4,
        max_export_bytes=1024**3,
    )
    records = export_tensors(leaves, tmp_path)
    assert [r.file for r in records] == ["tensor-000001.npy", "tensor-000002.npy"]
    for record, tensor in zip(records, (tensor_a, tensor_b), strict=True):
        assert record.original_dtype is None
        array = numpy.load(tmp_path / record.file, allow_pickle=False)
        assert numpy.array_equal(array, tensor.numpy())


def test_build_heavy_manifest_has_all_schema_fields_and_restricts_arguments(
    tmp_path,
) -> None:
    context = _context(hook="on_train_batch_start", stage="fit", epoch=1, global_step=9)
    tensor = torch.zeros(2, 2)
    leaves, truncated, reason = prepare_heavy_export(
        {"batch": tensor, "unrelated": "not exported"},
        ("batch",),
        max_items=10,
        max_depth=4,
        max_export_bytes=1024**3,
    )
    records = export_tensors(leaves, tmp_path)
    manifest = build_heavy_manifest(
        context=context,
        occurrence_id="occ-1",
        invocation_id="inv-1",
        arguments={"batch": tensor, "unrelated": "not exported"},
        tensor_arg_names=("batch",),
        tensor_records=records,
        max_metadata_items=10,
        max_metadata_depth=4,
        truncated=truncated,
        truncated_reason=reason,
    )
    for field in (
        "schema_version",
        "session_id",
        "occurrence_id",
        "invocation_id",
        "hook",
        "modality",
        "timestamp",
        "pid",
        "rank",
        "world_size",
        "stage",
        "epoch",
        "global_step",
        "batch_idx",
        "dataloader_idx",
        "arguments",
        "tensor_records",
        "truncated",
        "truncated_reason",
    ):
        assert field in manifest
    assert manifest["modality"] == "heavy"
    assert manifest["hook"] == "on_train_batch_start"
    assert manifest["session_id"] == "s-1"
    assert "unrelated" not in manifest["arguments"]
    assert manifest["arguments"]["batch"]["kind"] == "tensor"
    assert manifest["tensor_records"] == [
        {
            "argument_path": records[0].argument_path,
            "file": records[0].file,
            "shape": records[0].shape,
            "dtype": records[0].dtype,
            "original_dtype": records[0].original_dtype,
            "device": records[0].device,
        }
    ]


def test_build_heavy_manifest_for_no_tensor_callback_is_valid_and_empty() -> None:
    context = _context(hook="on_train_start")
    manifest = build_heavy_manifest(
        context=context,
        occurrence_id="occ-1",
        invocation_id="inv-1",
        arguments={},
        tensor_arg_names=(),
        tensor_records=[],
        max_metadata_items=10,
        max_metadata_depth=4,
        truncated=False,
        truncated_reason=None,
    )
    assert manifest["tensor_records"] == []
    assert manifest["arguments"] == {}
    assert manifest["truncated"] is False
    assert manifest["truncated_reason"] is None
