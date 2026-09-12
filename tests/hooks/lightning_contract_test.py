"""Static contract tests: `_HookMethods` against both installed Lightning namespaces.

No `Trainer`/`Trainer.fit` here -- purely signature and class-shape inspection, so
these tests run in milliseconds and catch a Lightning version bump that renames,
reorders, or removes a callback parameter before any real training run would.
"""

import inspect

import pytest

from trainctl.hooks.catalog import CALLBACK_NAMES, CALLBACKS
from trainctl.hooks.lightning import _HookMethods, build_hooks_callback_class
from trainctl.lightning_backend import load_backend

_BACKEND_NAMES = ("lightning.pytorch", "pytorch_lightning")

# Callback methods that manage the callback's own persisted state, not a
# Lightning-triggered training lifecycle event; deliberately not catalogued.
_NON_LIFECYCLE_METHODS = frozenset({"state_dict", "load_state_dict"})


def _own_params(method: object) -> list[str]:
    return [name for name in inspect.signature(method).parameters if name != "self"]


@pytest.fixture(params=_BACKEND_NAMES)
def backend(request: pytest.FixtureRequest):
    return load_backend(request.param)


@pytest.mark.parametrize("spec", CALLBACKS, ids=lambda spec: spec.name)
def test_catalogue_signature_matches_installed_callback(backend, spec) -> None:
    installed_method = getattr(backend.callback_class, spec.name, None)
    assert installed_method is not None, (
        f"{backend.name} Callback has no method {spec.name!r}"
    )
    adapter_method = getattr(_HookMethods, spec.name)

    installed_params = _own_params(installed_method)
    adapter_params = _own_params(adapter_method)

    assert installed_params == adapter_params, (
        f"{backend.name}.Callback.{spec.name} has parameters {installed_params}, "
        f"but _HookMethods.{spec.name} has {adapter_params}"
    )


def test_every_installed_lifecycle_method_is_catalogued(backend) -> None:
    installed_names = {
        name
        for name, value in vars(backend.callback_class).items()
        if not name.startswith("_") and callable(value)
    } - _NON_LIFECYCLE_METHODS

    missing = installed_names - CALLBACK_NAMES
    assert not missing, (
        f"{backend.name} Callback defines lifecycle method(s) {sorted(missing)} "
        "with no CallbackSpec entry -- catalogue drift"
    )


def test_every_catalogue_name_has_a_hook_method() -> None:
    missing = CALLBACK_NAMES - {
        name for name in vars(_HookMethods) if not name.startswith("_")
    }
    assert not missing, f"_HookMethods is missing method(s) for {sorted(missing)}"


def test_every_catalogue_name_exists_on_installed_callback(backend) -> None:
    missing = {
        name for name in CALLBACK_NAMES if not hasattr(backend.callback_class, name)
    }
    assert not missing, (
        f"{backend.name} Callback has no method for catalogued name(s) {sorted(missing)}"
    )


def test_hook_methods_defines_no_stray_lifecycle_looking_method() -> None:
    candidate_names = {
        name
        for name in vars(_HookMethods)
        if not name.startswith("_")
        and (name.startswith("on_") or name in {"setup", "teardown"})
    }
    stray = candidate_names - CALLBACK_NAMES
    assert not stray, (
        f"_HookMethods defines callback-looking method(s) {sorted(stray)} "
        "absent from the catalogue"
    )


def test_build_hooks_callback_class_composes_mro_correctly(backend) -> None:
    adapter_class = build_hooks_callback_class(backend)

    assert issubclass(adapter_class, backend.callback_class)
    assert issubclass(adapter_class, _HookMethods)
    mro = adapter_class.__mro__
    assert mro.index(_HookMethods) < mro.index(backend.callback_class)

    instance = adapter_class()
    assert isinstance(instance, backend.callback_class)


def test_build_hooks_callback_class_is_namespace_isolated() -> None:
    lightning_pytorch_backend = load_backend("lightning.pytorch")
    pytorch_lightning_backend = load_backend("pytorch_lightning")

    assert (
        lightning_pytorch_backend.callback_class
        is not pytorch_lightning_backend.callback_class
    )

    lightning_pytorch_adapter = build_hooks_callback_class(lightning_pytorch_backend)
    pytorch_lightning_adapter = build_hooks_callback_class(pytorch_lightning_backend)

    assert lightning_pytorch_adapter is not pytorch_lightning_adapter
    assert not issubclass(
        lightning_pytorch_adapter, pytorch_lightning_backend.callback_class
    )
    assert not issubclass(
        pytorch_lightning_adapter, lightning_pytorch_backend.callback_class
    )
