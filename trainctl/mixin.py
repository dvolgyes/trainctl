"""TrainctlMixin: the sole Trainctl integration surface for a Lightning `LightningModule`.

Compose it before either supported Lightning namespace's `LightningModule`:

    import lightning.pytorch as PL
    from trainctl import TrainctlMixin

    class MyModel(TrainctlMixin, PL.LightningModule):
        ...

or the same with `import pytorch_lightning as PL`. The two Lightning namespaces must
not be mixed inside one model/Trainer runtime; `__init_subclass__` rejects a class
whose MRO spans both, and `setup()` rejects a Trainer from the other namespace.

If a subclass overrides one of the reserved hooks below (`setup`, `on_fit_start`,
`teardown`, `on_train_batch_start`, `on_before_backward`, `backward`,
`on_after_backward`, `on_train_batch_end`, `on_train_epoch_end`,
`on_validation_epoch_end`, `on_before_optimizer_step`, `on_fit_end`,
`configure_callbacks`), it must call `super()`.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from trainctl.config import TrainctlConfig
from trainctl.hooks.lightning import build_hooks_callback_class
from trainctl.lightning_backend import (
    LightningBackend,
    detect_lightning_backend_from_mro,
)
from trainctl.runtime.events import SafePoint
from trainctl.runtime.runtime import TrainctlRuntime


class TrainctlMixin:
    """Starts and owns a `TrainctlRuntime` for the Lightning model it is composed with.

    Attributes:
        __trainctl_lightning_backend__: The Lightning backend detected from the
            subclass's MRO at class-definition time (`lightning.pytorch` or
            `pytorch_lightning`); set once per subclass by `__init_subclass__`.
        __trainctl_hooks_callback_class__: This subclass's concrete lifecycle-hooks
            `Callback` adapter class, built once per subclass by `__init_subclass__`.
    """

    __trainctl_lightning_backend__: LightningBackend
    __trainctl_hooks_callback_class__: type

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__trainctl_lightning_backend__ = detect_lightning_backend_from_mro(
            cls.__mro__
        )
        cls.__trainctl_hooks_callback_class__ = build_hooks_callback_class(
            cls.__trainctl_lightning_backend__
        )

    def __init__(
        self,
        *args: Any,
        trainctl_enabled: bool = True,
        fuse_enabled: bool = True,
        fuse_mountpoint: Path | str | None = None,
        rest_enabled: bool = True,
        rest_host: str = "127.0.0.1",
        rest_port: int = 8090,
        rest_port_search: int = 100,
        torch_debug_enabled: bool = True,
        torch_debug_port: int = 25999,
        torch_debug_port_search: int = 100,
        paired_port_search_attempts: int = 20,
        artifact_dir: Path | str | None = None,
        inspection_strict: bool = False,
        pipeline_pressure_enabled: bool = True,
        pipeline_pressure_window: int = 256,
        break_on_backward_exception: bool = True,
        dataloader_finite_check_enabled: bool = False,
        optimizer_surgery_enabled: bool = True,
        hooks_enabled: bool = True,
        hooks_source_dir: Path | str | None = None,
        hooks_timeout_s: float | None = None,
        hooks_rank_policy: str = "rank_zero",
        hooks_shell: Path | str | None = None,
        hooks_max_params_bytes: int = 256 * 1024,
        hooks_max_metadata_items: int = 1000,
        hooks_max_metadata_depth: int = 8,
        hooks_max_export_bytes: int = 1024**3,
        hooks_max_output_bytes: int = 64 * 1024,
        intercept_lightning_logging: bool = True,
        run_log_enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        config = TrainctlConfig(
            enabled=trainctl_enabled,
            fuse_enabled=fuse_enabled,
            fuse_mountpoint=fuse_mountpoint,
            rest_enabled=rest_enabled,
            rest_host=rest_host,
            rest_port=rest_port,
            rest_port_search=rest_port_search,
            torch_debug_enabled=torch_debug_enabled,
            torch_debug_port=torch_debug_port,
            torch_debug_port_search=torch_debug_port_search,
            paired_port_search_attempts=paired_port_search_attempts,
            artifact_dir=artifact_dir,
            inspection_strict=inspection_strict,
            pipeline_pressure_enabled=pipeline_pressure_enabled,
            pipeline_pressure_window=pipeline_pressure_window,
            break_on_backward_exception=break_on_backward_exception,
            dataloader_finite_check_enabled=dataloader_finite_check_enabled,
            optimizer_surgery_enabled=optimizer_surgery_enabled,
            hooks_enabled=hooks_enabled,
            hooks_source_dir=hooks_source_dir,
            hooks_timeout_s=hooks_timeout_s,
            hooks_rank_policy=hooks_rank_policy,
            hooks_shell=hooks_shell,
            hooks_max_params_bytes=hooks_max_params_bytes,
            hooks_max_metadata_items=hooks_max_metadata_items,
            hooks_max_metadata_depth=hooks_max_metadata_depth,
            hooks_max_export_bytes=hooks_max_export_bytes,
            hooks_max_output_bytes=hooks_max_output_bytes,
            intercept_lightning_logging=intercept_lightning_logging,
            run_log_enabled=run_log_enabled,
        )
        self._trainctl = TrainctlRuntime(
            config, type(self).__trainctl_lightning_backend__
        )
        self._trainctl_hooks_adapter: Any | None = None
        super().__init__(*args, **kwargs)

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self._trainctl.config.enabled:
            self._validate_trainer_backend()
            self._trainctl.setup_environment(self, stage)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if self._trainctl.config.enabled:
            self._trainctl.on_fit_start(self)

    def teardown(self, stage: str) -> None:
        try:
            if self._trainctl.config.enabled:
                self._trainctl.teardown(self, stage)
        finally:
            super().teardown(stage)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
        result = super().on_train_batch_start(batch, batch_idx)
        if self._trainctl.config.enabled:
            self._trainctl.begin_train_batch(self, batch, batch_idx)
        return result

    def on_before_backward(self, loss: Any) -> None:
        super().on_before_backward(loss)
        if self._trainctl.config.enabled:
            self._trainctl.before_backward(loss)

    def backward(self, loss: Any, *args: Any, **kwargs: Any) -> None:
        try:
            super().backward(loss, *args, **kwargs)
        except Exception as exc:
            if self._trainctl.config.enabled:
                self._trainctl.handle_backward_exception(self, loss, exc)
            raise

    def on_after_backward(self) -> None:
        super().on_after_backward()
        if self._trainctl.config.enabled:
            self._trainctl.after_backward()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        try:
            super().on_train_batch_end(outputs, batch, batch_idx)
            if self._trainctl.config.enabled:
                self._trainctl.process_safe_point(
                    self, SafePoint.TRAIN_BATCH_END, batch_idx=batch_idx
                )
        finally:
            if self._trainctl.config.enabled:
                self._trainctl.finish_train_batch(self)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self._trainctl.config.enabled:
            self._trainctl.process_safe_point(self, SafePoint.TRAIN_EPOCH_END)

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        if self._trainctl.config.enabled:
            self._trainctl.process_safe_point(self, SafePoint.VALIDATION_EPOCH_END)

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        super().on_before_optimizer_step(optimizer)
        if self._trainctl.config.enabled:
            self._trainctl.before_optimizer_step()
            self._trainctl.process_safe_point(self, SafePoint.BEFORE_OPTIMIZER_STEP)

    def on_fit_end(self) -> None:
        super().on_fit_end()
        if self._trainctl.config.enabled:
            self._trainctl.process_safe_point(self, SafePoint.FIT_END)

    def configure_callbacks(self) -> Sequence[Any]:
        callbacks = _normalize_callbacks(super().configure_callbacks())  # type: ignore[misc]  # provided by the composed LightningModule
        if self._trainctl.config.enabled and self._trainctl.config.hooks_enabled:
            if self._trainctl_hooks_adapter is None:
                self._trainctl_hooks_adapter = type(
                    self
                ).__trainctl_hooks_callback_class__()
            callbacks.append(self._trainctl_hooks_adapter)
        return callbacks

    def _validate_trainer_backend(self) -> None:
        backend = type(self).__trainctl_lightning_backend__
        trainer = self.trainer  # type: ignore[attr-defined]  # provided by the composed LightningModule
        if not isinstance(trainer, backend.Trainer):
            raise RuntimeError(
                f"Trainctl Lightning backend mismatch: model uses {backend.name}, but Trainer is "
                f"{type(trainer).__module__}.{type(trainer).__name__}. "
                "Do not mix lightning.pytorch and pytorch_lightning."
            )


def _normalize_callbacks(value: Any) -> list[Any]:
    """Coerces a `configure_callbacks()` return value to a fresh, appendable list."""
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]
