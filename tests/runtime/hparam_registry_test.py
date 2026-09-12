"""Behavioral tests for the generic tunable-hyperparameter registry.

`resolve`/`describe`/`cast_and_assign` are plain functions over arbitrary
parent/leaf attribute pairs, independent of `TrainctlRuntime` --
`tests/runtime/set_hparam_command_test.py` covers the end-to-end command path.
"""

import pytest
import torch

from trainctl.runtime import hparam_registry


class _Hparams:
    def __init__(self) -> None:
        self.augmentation_strength = 0.5


class _Model:
    def __init__(self) -> None:
        self.hparams = _Hparams()
        self.flag = True
        self.label = "baseline"
        self.weight = torch.tensor([1.0, 2.0, 3.0])


class _Trainer:
    def __init__(self, datamodule: object) -> None:
        self.datamodule = datamodule


def test_resolve_plain_attribute_path() -> None:
    model = _Model()

    parent, leaf = hparam_registry.resolve(model, trainer=None, path="flag")

    assert parent is model
    assert leaf == "flag"


def test_resolve_nested_pl_module_path() -> None:
    model = _Model()

    parent, leaf = hparam_registry.resolve(
        model, trainer=None, path="hparams.augmentation_strength"
    )

    assert parent is model.hparams
    assert leaf == "augmentation_strength"


def test_resolve_trainer_rooted_path() -> None:
    model = _Model()
    trainer = _Trainer(datamodule=model)

    parent, leaf = hparam_registry.resolve(
        model, trainer, path="trainer.datamodule.flag"
    )

    assert parent is model
    assert leaf == "flag"


def test_describe_bool_before_int_ordering() -> None:
    result = hparam_registry.describe(True)

    assert result.type == "bool"
    assert result.value is True


def test_describe_int() -> None:
    assert hparam_registry.describe(3).type == "int"


def test_describe_float() -> None:
    assert hparam_registry.describe(0.5).type == "float"


def test_describe_str() -> None:
    assert hparam_registry.describe("baseline").type == "str"


def test_describe_tensor_reports_dtype_device_and_shape() -> None:
    tensor = torch.tensor([1.0, 2.0, 3.0])

    result = hparam_registry.describe(tensor)

    assert result.type == "tensor"
    assert result.dtype == "torch.float32"
    assert result.device == "cpu"
    assert result.shape == [3]
    assert result.value == [1.0, 2.0, 3.0]


def test_describe_oversized_tensor_omits_value_but_keeps_metadata() -> None:
    size = hparam_registry._MAX_INLINE_TENSOR_ELEMENTS + 1
    tensor = torch.zeros(size)

    result = hparam_registry.describe(tensor)

    assert result.value is None
    assert result.shape == [size]
    assert result.dtype == "torch.float32"


def test_describe_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        hparam_registry.describe([1, 2, 3])


def test_cast_and_assign_bool_before_int_ordering() -> None:
    model = _Model()

    result = hparam_registry.cast_and_assign(model, "flag", 0)

    assert result.type == "bool"
    assert result.value is False
    assert model.flag is False


def test_cast_and_assign_int() -> None:
    model = _Model()
    model.count = 1

    result = hparam_registry.cast_and_assign(model, "count", "5")

    assert result.type == "int"
    assert result.value == 5


def test_cast_and_assign_float() -> None:
    model = _Model()

    result = hparam_registry.cast_and_assign(
        model.hparams, "augmentation_strength", "0.75"
    )

    assert result.type == "float"
    assert result.value == pytest.approx(0.75)


def test_cast_and_assign_str() -> None:
    model = _Model()

    result = hparam_registry.cast_and_assign(model, "label", 123)

    assert result.type == "str"
    assert result.value == "123"


def test_cast_and_assign_tensor_matches_dtype_moves_device_and_preserves_identity() -> (
    None
):
    model = _Model()
    original = model.weight

    result = hparam_registry.cast_and_assign(model, "weight", [4.0, 5.0, 6.0])

    assert model.weight is original
    assert torch.equal(model.weight, torch.tensor([4.0, 5.0, 6.0]))
    assert result.dtype == "torch.float32"
    assert result.device == "cpu"


def test_cast_and_assign_tensor_shape_mismatch_raises() -> None:
    model = _Model()

    with pytest.raises(ValueError, match="shape mismatch"):
        hparam_registry.cast_and_assign(model, "weight", [1.0, 2.0])


def test_cast_and_assign_unsupported_current_type_raises() -> None:
    model = _Model()
    model.unsupported = object()

    with pytest.raises(ValueError, match="unsupported"):
        hparam_registry.cast_and_assign(model, "unsupported", 1)


# ---- collect_tunable_hparams: MRO merge ------------------------------------


class _MixinA:
    trainctl_tunable_hparams = ["a_field"]


class _MixinB:
    trainctl_tunable_hparams = ["b_field"]


class _CombinedListForm(_MixinA, _MixinB):
    pass


def test_collect_tunable_hparams_merges_bare_list_across_mro() -> None:
    specs = hparam_registry.collect_tunable_hparams(_CombinedListForm)

    assert set(specs) == {"a_field", "b_field"}
    assert specs["a_field"] == hparam_registry.TunableHparamSpec(path="a_field")
    assert specs["b_field"] == hparam_registry.TunableHparamSpec(path="b_field")


class _DictForm:
    trainctl_tunable_hparams = {
        "hparams.lr": {"min": 0.0, "max": 1.0, "label": "Learning rate"},
    }


def test_collect_tunable_hparams_dict_form_carries_min_max_label() -> None:
    specs = hparam_registry.collect_tunable_hparams(_DictForm)

    assert specs["hparams.lr"] == hparam_registry.TunableHparamSpec(
        path="hparams.lr", min=0.0, max=1.0, label="Learning rate"
    )


class _AncestorBounded:
    trainctl_tunable_hparams = {"shared": {"min": 0, "max": 10, "label": "ancestor"}}


class _LeafReplaces(_AncestorBounded):
    trainctl_tunable_hparams = {"shared": {"label": "leaf"}}


def test_collect_tunable_hparams_leaf_replaces_ancestor_spec_wholesale() -> None:
    specs = hparam_registry.collect_tunable_hparams(_LeafReplaces)

    assert specs["shared"] == hparam_registry.TunableHparamSpec(
        path="shared", label="leaf"
    )


class _MixinListForm:
    trainctl_tunable_hparams = ["mixin_field"]


class _SubclassDictForm(_MixinListForm):
    trainctl_tunable_hparams = {"dict_field": {"label": "Dict field"}}


def test_collect_tunable_hparams_mixes_list_and_dict_forms_across_mro() -> None:
    specs = hparam_registry.collect_tunable_hparams(_SubclassDictForm)

    assert specs["mixin_field"] == hparam_registry.TunableHparamSpec(path="mixin_field")
    assert specs["dict_field"] == hparam_registry.TunableHparamSpec(
        path="dict_field", label="Dict field"
    )


class _BareStrForm:
    trainctl_tunable_hparams = "oops"


def test_collect_tunable_hparams_bare_str_raises_type_error() -> None:
    with pytest.raises(TypeError, match="brackets"):
        hparam_registry.collect_tunable_hparams(_BareStrForm)


class _NonDictMetadataForm:
    trainctl_tunable_hparams = {"field": "not-a-dict"}


def test_collect_tunable_hparams_non_dict_metadata_raises_type_error() -> None:
    with pytest.raises(TypeError, match="field"):
        hparam_registry.collect_tunable_hparams(_NonDictMetadataForm)


# ---- dict-nested paths (get_field/set_field/resolve/cast_and_assign) ------


class _NestedModel:
    def __init__(self) -> None:
        self.config = {"optimizer": {"lr_schedule": {"warmup_steps": 100}}}
        self.state = {"weight": torch.tensor([1.0, 2.0])}


def test_resolve_walks_into_nested_dict() -> None:
    model = _NestedModel()

    parent, leaf = hparam_registry.resolve(
        model, trainer=None, path="config.optimizer.lr_schedule.warmup_steps"
    )

    assert parent is model.config["optimizer"]["lr_schedule"]
    assert leaf == "warmup_steps"


def test_cast_and_assign_reads_and_writes_dict_nested_leaf() -> None:
    model = _NestedModel()
    inner = model.config["optimizer"]["lr_schedule"]

    result = hparam_registry.cast_and_assign(inner, "warmup_steps", "200")

    assert result.value == 200
    assert inner["warmup_steps"] == 200


def test_cast_and_assign_preserves_dict_nested_tensor_identity_dtype_device() -> None:
    model = _NestedModel()
    original = model.state["weight"]

    result = hparam_registry.cast_and_assign(model.state, "weight", [3.0, 4.0])

    assert model.state["weight"] is original
    assert torch.equal(model.state["weight"], torch.tensor([3.0, 4.0]))
    assert result.dtype == "torch.float32"
    assert result.device == "cpu"


def test_get_field_raises_when_key_is_neither_attribute_nor_mapping_key() -> None:
    with pytest.raises(AttributeError):
        hparam_registry.get_field({"a": 1}, "b")


def test_set_field_raises_when_key_is_neither_attribute_nor_mapping_key() -> None:
    with pytest.raises(AttributeError):
        hparam_registry.set_field(object(), "missing", 1)


# ---- validate_bounds --------------------------------------------------------


def test_validate_bounds_noop_when_unbounded() -> None:
    spec = hparam_registry.TunableHparamSpec(path="x")

    hparam_registry.validate_bounds(spec, 999)  # must not raise


def test_validate_bounds_scalar_within_range_passes() -> None:
    spec = hparam_registry.TunableHparamSpec(path="x", min=0, max=10)

    hparam_registry.validate_bounds(spec, 5)  # must not raise


def test_validate_bounds_scalar_below_min_raises() -> None:
    spec = hparam_registry.TunableHparamSpec(path="x", min=0, max=10)

    with pytest.raises(ValueError, match=">= 0"):
        hparam_registry.validate_bounds(spec, -1)


def test_validate_bounds_scalar_above_max_raises() -> None:
    spec = hparam_registry.TunableHparamSpec(path="x", min=0, max=10)

    with pytest.raises(ValueError, match="<= 10"):
        hparam_registry.validate_bounds(spec, 11)


def test_validate_bounds_rank0_tensor_behaves_like_scalar() -> None:
    spec = hparam_registry.TunableHparamSpec(path="x", min=0, max=10)

    hparam_registry.validate_bounds(spec, torch.tensor(5.0))  # must not raise
    with pytest.raises(ValueError, match=">= 0"):
        hparam_registry.validate_bounds(spec, torch.tensor(-1.0))


def test_validate_bounds_multi_element_tensor_rejects_any_out_of_range_element() -> (
    None
):
    spec = hparam_registry.TunableHparamSpec(path="x", min=0, max=10)

    with pytest.raises(ValueError, match=">= 0"):
        hparam_registry.validate_bounds(spec, [5.0, -1.0])


def test_validate_bounds_multi_element_tensor_all_in_range_passes() -> None:
    spec = hparam_registry.TunableHparamSpec(path="x", min=0, max=10)

    hparam_registry.validate_bounds(
        spec, torch.tensor([1.0, 5.0, 9.0])
    )  # must not raise
