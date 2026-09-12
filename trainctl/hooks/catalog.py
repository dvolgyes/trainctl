"""Immutable catalogue of the 37 Lightning lifecycle callbacks hooks can subscribe to.

Both supported namespaces (`lightning.pytorch` and `pytorch_lightning`) share this exact
public `Callback` API at the pinned version (2.6.5); `tests/hooks/lightning_contract_test.py`
(Increment I4) verifies the catalogue against each namespace's installed `Callback` class and
fails on drift. This module owns only the static description, not the adapter's method bodies.
"""

from dataclasses import dataclass

CATALOGUE_VERSION = 1


@dataclass(frozen=True)
class CallbackSpec:
    """One Lightning `Callback` method this catalogue tracks.

    Attributes:
        name: The exact `Callback` method name (also the script's stem: `light/<name>.sh`).
        group: A human-readable grouping label used in generated documentation.
        tensor_args: Names of this callback's arguments that may carry tensors, in the order
            Lightning passes them (excluding `trainer`/`pl_module`). These are the arguments a
            heavy hook exports by default (Increment I3); an empty tuple means the callback
            never carries tensors, so its heavy manifest is always empty.
        description: A one-line explanation of when Lightning calls this method.
    """

    name: str
    group: str
    tensor_args: tuple[str, ...]
    description: str


CALLBACKS: tuple[CallbackSpec, ...] = (
    CallbackSpec(
        "setup",
        "Setup and fit",
        (),
        "Called once per stage (fit/validate/test/predict) before the loop starts, on every "
        "process.",
    ),
    CallbackSpec(
        "teardown",
        "Setup and fit",
        (),
        "Called once per stage, mirroring setup, including during an orderly Trainer shutdown.",
    ),
    CallbackSpec(
        "on_fit_start",
        "Setup and fit",
        (),
        "Called at the very start of fit, after sanity checking is scheduled but before it runs.",
    ),
    CallbackSpec(
        "on_fit_end",
        "Setup and fit",
        (),
        "Called at the very end of fit, after training and validation have both finished.",
    ),
    CallbackSpec(
        "on_sanity_check_start",
        "Sanity check",
        (),
        "Called before Lightning's pre-training validation sanity check runs.",
    ),
    CallbackSpec(
        "on_sanity_check_end",
        "Sanity check",
        (),
        "Called after the sanity check completes.",
    ),
    CallbackSpec(
        "on_train_start",
        "Training",
        (),
        "Called at the beginning of training, after sanity checking.",
    ),
    CallbackSpec(
        "on_train_end",
        "Training",
        (),
        "Called at the end of training, before fit's own on_fit_end.",
    ),
    CallbackSpec(
        "on_train_epoch_start",
        "Training",
        (),
        "Called at the beginning of each training epoch.",
    ),
    CallbackSpec(
        "on_train_epoch_end",
        "Training",
        (),
        "Called at the end of each training epoch, after all its batches and any interleaved "
        "validation.",
    ),
    CallbackSpec(
        "on_train_batch_start",
        "Training",
        ("batch",),
        "Called before each training batch is run through the training step.",
    ),
    CallbackSpec(
        "on_train_batch_end",
        "Training",
        ("outputs", "batch"),
        "Called after each training batch's training step and optimizer step complete.",
    ),
    CallbackSpec(
        "on_validation_start",
        "Validation",
        (),
        "Called at the beginning of validation.",
    ),
    CallbackSpec(
        "on_validation_end",
        "Validation",
        (),
        "Called at the end of validation.",
    ),
    CallbackSpec(
        "on_validation_epoch_start",
        "Validation",
        (),
        "Called at the beginning of a validation epoch.",
    ),
    CallbackSpec(
        "on_validation_epoch_end",
        "Validation",
        (),
        "Called at the end of a validation epoch, after all its batches.",
    ),
    CallbackSpec(
        "on_validation_batch_start",
        "Validation",
        ("batch",),
        "Called before each validation batch is run through the validation step.",
    ),
    CallbackSpec(
        "on_validation_batch_end",
        "Validation",
        ("outputs", "batch"),
        "Called after each validation batch's validation step completes.",
    ),
    CallbackSpec(
        "on_test_start",
        "Test",
        (),
        "Called at the beginning of testing.",
    ),
    CallbackSpec(
        "on_test_end",
        "Test",
        (),
        "Called at the end of testing.",
    ),
    CallbackSpec(
        "on_test_epoch_start",
        "Test",
        (),
        "Called at the beginning of a test epoch.",
    ),
    CallbackSpec(
        "on_test_epoch_end",
        "Test",
        (),
        "Called at the end of a test epoch, after all its batches.",
    ),
    CallbackSpec(
        "on_test_batch_start",
        "Test",
        ("batch",),
        "Called before each test batch is run through the test step.",
    ),
    CallbackSpec(
        "on_test_batch_end",
        "Test",
        ("outputs", "batch"),
        "Called after each test batch's test step completes.",
    ),
    CallbackSpec(
        "on_predict_start",
        "Prediction",
        (),
        "Called at the beginning of prediction.",
    ),
    CallbackSpec(
        "on_predict_end",
        "Prediction",
        (),
        "Called at the end of prediction.",
    ),
    CallbackSpec(
        "on_predict_epoch_start",
        "Prediction",
        (),
        "Called at the beginning of a predict epoch.",
    ),
    CallbackSpec(
        "on_predict_epoch_end",
        "Prediction",
        (),
        "Called at the end of a predict epoch, after all its batches.",
    ),
    CallbackSpec(
        "on_predict_batch_start",
        "Prediction",
        ("batch",),
        "Called before each predict batch is run through the predict step.",
    ),
    CallbackSpec(
        "on_predict_batch_end",
        "Prediction",
        ("outputs", "batch"),
        "Called after each predict batch's predict step completes.",
    ),
    CallbackSpec(
        "on_before_backward",
        "Optimization",
        ("loss",),
        "Called with the loss tensor immediately before backward() runs.",
    ),
    CallbackSpec(
        "on_after_backward",
        "Optimization",
        (),
        "Called immediately after backward() runs, before the optimizer step.",
    ),
    CallbackSpec(
        "on_before_optimizer_step",
        "Optimization",
        (),
        "Called with the optimizer immediately before it takes a step.",
    ),
    CallbackSpec(
        "on_before_zero_grad",
        "Optimization",
        (),
        "Called with the optimizer immediately before gradients are zeroed.",
    ),
    CallbackSpec(
        "on_save_checkpoint",
        "Checkpoint and failure",
        ("checkpoint",),
        "Called with the checkpoint dict immediately before it is written to disk.",
    ),
    CallbackSpec(
        "on_load_checkpoint",
        "Checkpoint and failure",
        ("checkpoint",),
        "Called with the checkpoint dict immediately after it is read from disk.",
    ),
    CallbackSpec(
        "on_exception",
        "Checkpoint and failure",
        (),
        "Called once if training raises an uncaught exception; the module's own teardown hook "
        "does not run on this path.",
    ),
)

CALLBACK_NAMES: frozenset[str] = frozenset(spec.name for spec in CALLBACKS)

CALLBACK_BY_NAME: dict[str, CallbackSpec] = {spec.name: spec for spec in CALLBACKS}

MODALITIES: tuple[str, ...] = ("light", "heavy")
