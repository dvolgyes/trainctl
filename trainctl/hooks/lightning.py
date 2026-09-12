"""Lightning `Callback` adapter: explicit signatures for all 37 lifecycle callbacks
the catalogue tracks (Increment I4).

Deliberately not built by runtime reflection over the installed `Callback` class --
`tests/hooks/lightning_contract_test.py` compares this module's coverage against
both namespaces' installed `Callback` API and fails on drift, but production code
here never discovers a method that way.
"""

from typing import Any

from trainctl.lightning_backend import LightningBackend


class _HookMethods:
    """Mixed in ahead of a backend's `Callback` base by `build_hooks_callback_class`.

    Each method forwards only its own named arguments (never `trainer`/`pl_module`)
    to `TrainctlRuntime.dispatch_hook`, which re-derives stage/epoch/step/rank
    identity fresh from `trainer` for every call. `setup`/`teardown` additionally
    own the hooks session's lifetime: Lightning always calls `setup` before any
    other callback method for a stage `_run`, and always pairs it with a matching
    `teardown` afterward (barring an uncaught exception, handled by `on_exception`
    below) -- so opening in `setup` and closing in `teardown` cannot leak or
    outlive the `_run` it belongs to, even for a stage entered without `fit`/
    `validate`/`test`/`predict` ever completing normally.
    """

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        pl_module._trainctl.ensure_hook_session(trainer, pl_module)  # noqa: SLF001 -- intentional internal-integration coupling
        _dispatch(trainer, pl_module, "setup", stage_override=stage)

    def teardown(self, trainer: Any, pl_module: Any, stage: str) -> None:
        try:
            _dispatch(trainer, pl_module, "teardown", stage_override=stage)
        finally:
            pl_module._trainctl.finalize_hooks_and_logging()  # noqa: SLF001 -- intentional internal-integration coupling

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_fit_start")

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_fit_end")

    def on_sanity_check_start(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_sanity_check_start")

    def on_sanity_check_end(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_sanity_check_end")

    def on_train_start(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_train_start")

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_train_end")

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_train_epoch_start")

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_train_epoch_end")

    def on_train_batch_start(
        self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int
    ) -> None:
        _dispatch(
            trainer,
            pl_module,
            "on_train_batch_start",
            arguments={"batch": batch},
            batch_idx=batch_idx,
        )

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        _dispatch(
            trainer,
            pl_module,
            "on_train_batch_end",
            arguments={"outputs": outputs, "batch": batch},
            batch_idx=batch_idx,
        )

    def on_validation_start(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_validation_start")

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_validation_end")

    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_validation_epoch_start")

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_validation_epoch_end")

    def on_validation_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _dispatch(
            trainer,
            pl_module,
            "on_validation_batch_start",
            arguments={"batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _dispatch(
            trainer,
            pl_module,
            "on_validation_batch_end",
            arguments={"outputs": outputs, "batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_test_start(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_test_start")

    def on_test_end(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_test_end")

    def on_test_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_test_epoch_start")

    def on_test_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_test_epoch_end")

    def on_test_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _dispatch(
            trainer,
            pl_module,
            "on_test_batch_start",
            arguments={"batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _dispatch(
            trainer,
            pl_module,
            "on_test_batch_end",
            arguments={"outputs": outputs, "batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_predict_start(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_predict_start")

    def on_predict_end(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_predict_end")

    def on_predict_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_predict_epoch_start")

    def on_predict_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_predict_epoch_end")

    def on_predict_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _dispatch(
            trainer,
            pl_module,
            "on_predict_batch_start",
            arguments={"batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_predict_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _dispatch(
            trainer,
            pl_module,
            "on_predict_batch_end",
            arguments={"outputs": outputs, "batch": batch},
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    def on_before_backward(self, trainer: Any, pl_module: Any, loss: Any) -> None:
        _dispatch(trainer, pl_module, "on_before_backward", arguments={"loss": loss})

    def on_after_backward(self, trainer: Any, pl_module: Any) -> None:
        _dispatch(trainer, pl_module, "on_after_backward")

    def on_before_optimizer_step(
        self, trainer: Any, pl_module: Any, optimizer: Any
    ) -> None:
        _dispatch(
            trainer,
            pl_module,
            "on_before_optimizer_step",
            arguments={"optimizer": optimizer},
        )

    def on_before_zero_grad(self, trainer: Any, pl_module: Any, optimizer: Any) -> None:
        _dispatch(
            trainer, pl_module, "on_before_zero_grad", arguments={"optimizer": optimizer}
        )

    def on_save_checkpoint(
        self, trainer: Any, pl_module: Any, checkpoint: dict[str, Any]
    ) -> None:
        _dispatch(
            trainer,
            pl_module,
            "on_save_checkpoint",
            arguments={"checkpoint": checkpoint},
        )

    def on_load_checkpoint(
        self, trainer: Any, pl_module: Any, checkpoint: dict[str, Any]
    ) -> None:
        _dispatch(
            trainer,
            pl_module,
            "on_load_checkpoint",
            arguments={"checkpoint": checkpoint},
        )

    def on_exception(
        self, trainer: Any, pl_module: Any, exception: BaseException
    ) -> None:
        try:
            _dispatch(
                trainer,
                pl_module,
                "on_exception",
                arguments={"exception": exception},
            )
        finally:
            pl_module._trainctl.finalize_hooks_and_logging()  # noqa: SLF001 -- intentional internal-integration coupling


def _dispatch(
    trainer: Any,
    pl_module: Any,
    hook: str,
    *,
    stage_override: str | None = None,
    arguments: dict[str, Any] | None = None,
    batch_idx: int | None = None,
    dataloader_idx: int | None = None,
) -> None:
    runtime = pl_module._trainctl  # noqa: SLF001 -- intentional internal-integration coupling
    runtime.dispatch_hook(
        trainer,
        hook,
        stage_override=stage_override,
        arguments=arguments,
        batch_idx=batch_idx,
        dataloader_idx=dataloader_idx,
    )


def build_hooks_callback_class(backend: LightningBackend) -> type:
    """Builds this backend's concrete Trainctl hooks `Callback` adapter class.

    One class per backend family; `TrainctlMixin.__init_subclass__` builds and
    caches it once per subclass, alongside the backend detection it already does.
    """
    return type("TrainctlHooksCallback", (_HookMethods, backend.callback_class), {})
