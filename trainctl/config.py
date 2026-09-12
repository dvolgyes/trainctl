"""Configuration for the services TrainctlMixin starts around a Lightning run."""

import os
from dataclasses import dataclass
from pathlib import Path

_VALID_HOOKS_RANK_POLICIES = ("rank_zero", "all")

_POSITIVE_INT_HOOKS_FIELDS = (
    "hooks_max_params_bytes",
    "hooks_max_metadata_items",
    "hooks_max_metadata_depth",
    "hooks_max_export_bytes",
    "hooks_max_output_bytes",
)


@dataclass(frozen=True)
class TrainctlConfig:
    """Options controlling which Trainctl services start and how.

    Attributes:
        enabled: Master switch; when False, TrainctlMixin does not start any service.
        fuse_enabled: Start the read-only mfusepy inspection filesystem.
        fuse_mountpoint: Explicit mount directory; auto-selected under
            `<trainer.log_dir>/procfs` when a Trainer reports a log directory,
            else `$XDG_RUNTIME_DIR/trainctl/<run-id>/procfs` (or
            `/tmp/trainctl-<uid>/<run-id>/procfs`), when None.
        rest_enabled: Start the REST control/debug API.
        rest_host: Bind host for the REST server.
        rest_port: First REST port to try.
        rest_port_search: Number of sequential ports to try if `rest_port` is taken.
        torch_debug_enabled: Start `torch.distributed.debug`'s HTTP server, when available.
        torch_debug_port: First frontend port to try for the distributed debug server.
        torch_debug_port_search: Number of sequential ports to try for the debug frontend
            when it starts without a paired REST server (see `paired_port_search_attempts`).
        paired_port_search_attempts: When both `rest_enabled` and `torch_debug_enabled`
            are true (rank zero only), REST and the debug frontend are started as one
            paired unit: both ports are offset from their configured base port by the
            same multiple of 100 (e.g. offset 200 pairs REST 8290 with debug 26199) --
            a human-legible pairing that also keeps the two services from independently
            drifting to unrelated ports under concurrent contention. This many distinct
            offsets are tried before giving up.
        artifact_dir: Directory for checkpoints/snapshots/debug captures; auto-selected
            under `<trainer.log_dir>/artifacts` when a Trainer reports a log
            directory, else `$XDG_RUNTIME_DIR/trainctl/<run-id>/artifacts` (or
            `/tmp/trainctl-<uid>/<run-id>/artifacts`), when None.
        inspection_strict: Raise instead of logging a warning when a Trainctl service
            fails to start. Intended for development/tests.
        pipeline_pressure_enabled: Record cheap per-batch phase timing (input gap,
            forward/loss, backward, post-backward).
        pipeline_pressure_window: Number of recent batches kept for rolling
            pipeline-pressure statistics.
        break_on_backward_exception: Intercept an exception escaping
            `LightningModule.backward()`, freeze the failing batch/loss for
            inspection, and hold until released before re-raising. Does not itself
            enable PyTorch/Lightning anomaly detection (`Trainer(detect_anomaly=True)`).
        dataloader_finite_check_enabled: Check every floating/complex tensor in each
            training batch for NaN/Inf before the training step runs. Disabled by
            default -- expensive and can force a host/device sync per batch.
        optimizer_surgery_enabled: Allow the `reset_momentum` command and the
            `param_group`-selector form of `set_learning_rate`. The unconditional,
            all-groups `set_learning_rate` remains available regardless.
        hooks_enabled: Master switch for the lifecycle shell-hooks facility: seeding a
            run-local `hooks/` tree and dispatching enabled scripts at Lightning
            callbacks. `False` skips all hook filesystem and dispatch work.
        hooks_source_dir: Optional baseline directory (`light/`, `heavy/`, and
            supporting files) copied into the run-local `hooks/` tree at bootstrap.
            Never executed in place. Must exist and be a directory when `hooks_enabled`
            is true.
        hooks_timeout_s: Positive per-script timeout in seconds, or `None` for a
            deliberately unbounded run -- appropriate for interactive debugging, but
            able to stall training and any distributed watchdog indefinitely.
        hooks_rank_policy: `rank_zero` dispatches hooks only on the global-rank-zero
            process; `all` dispatches on every rank.
        hooks_shell: Explicit interpreter used to launch a hook script; must be an
            existing executable file when given. `None` resolves to the first `bash`
            found on `PATH` at bootstrap.
        hooks_max_params_bytes: Byte limit for a light hook's encoded parameters JSON;
            an oversized payload is truncated with the reason recorded, not silently
            dropped.
        hooks_max_metadata_items: Maximum number of items described per traversed
            container when building a hook's bounded argument metadata.
        hooks_max_metadata_depth: Maximum traversal depth for the same bounded
            argument-metadata description.
        hooks_max_export_bytes: Byte limit for a heavy hook's predicted converted
            tensor export size, checked before any host copy is allocated.
        hooks_max_output_bytes: Byte limit for the stdout/stderr tail retained per
            hook invocation for its execution-result record.
        intercept_lightning_logging: Bridge `lightning.pytorch`/`pytorch_lightning`/
            `lightning.fabric`'s standard-library logging into Loguru for this run.
        run_log_enabled: Attach a Loguru file sink (`trainctl.log`, or
            `trainctl-rank-N.log` under an `all` rank policy) in the run directory.
    """

    enabled: bool = True

    fuse_enabled: bool = True
    fuse_mountpoint: Path | str | None = None

    rest_enabled: bool = True
    rest_host: str = "127.0.0.1"
    rest_port: int = 8090
    rest_port_search: int = 100

    torch_debug_enabled: bool = True
    torch_debug_port: int = 25999
    torch_debug_port_search: int = 100
    paired_port_search_attempts: int = 20

    artifact_dir: Path | str | None = None

    inspection_strict: bool = False

    pipeline_pressure_enabled: bool = True
    pipeline_pressure_window: int = 256

    break_on_backward_exception: bool = True

    dataloader_finite_check_enabled: bool = False

    optimizer_surgery_enabled: bool = True

    hooks_enabled: bool = True
    hooks_source_dir: Path | str | None = None
    hooks_timeout_s: float | None = None
    hooks_rank_policy: str = "rank_zero"
    hooks_shell: Path | str | None = None
    hooks_max_params_bytes: int = 256 * 1024
    hooks_max_metadata_items: int = 1000
    hooks_max_metadata_depth: int = 8
    hooks_max_export_bytes: int = 1024**3
    hooks_max_output_bytes: int = 64 * 1024

    intercept_lightning_logging: bool = True
    run_log_enabled: bool = True

    def __post_init__(self) -> None:
        """Canonicalizes path-like fields and validates hook/logging settings.

        Raises:
            ValueError: A hook setting is out of range, or an explicitly given
                `hooks_source_dir`/`hooks_shell` does not exist as required.
        """
        if self.fuse_mountpoint is not None:
            object.__setattr__(self, "fuse_mountpoint", Path(self.fuse_mountpoint))
        if self.artifact_dir is not None:
            object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))
        hooks_source_dir = self.hooks_source_dir
        if hooks_source_dir is not None:
            hooks_source_dir = Path(hooks_source_dir)
            object.__setattr__(self, "hooks_source_dir", hooks_source_dir)
        hooks_shell = self.hooks_shell
        if hooks_shell is not None:
            hooks_shell = Path(hooks_shell)
            object.__setattr__(self, "hooks_shell", hooks_shell)

        if self.hooks_rank_policy not in _VALID_HOOKS_RANK_POLICIES:
            raise ValueError(
                f"hooks_rank_policy must be one of {_VALID_HOOKS_RANK_POLICIES}, "
                f"got {self.hooks_rank_policy!r}"
            )
        if self.hooks_timeout_s is not None and self.hooks_timeout_s <= 0:
            raise ValueError(
                f"hooks_timeout_s must be positive or None, got {self.hooks_timeout_s!r}"
            )
        for field_name in _POSITIVE_INT_HOOKS_FIELDS:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value!r}")

        if (
            self.hooks_enabled
            and hooks_source_dir is not None
            and not hooks_source_dir.is_dir()
        ):
            raise ValueError(
                f"hooks_source_dir {hooks_source_dir} does not exist or is not a directory"
            )
        if (
            self.hooks_enabled
            and hooks_shell is not None
            and not (hooks_shell.is_file() and os.access(hooks_shell, os.X_OK))
        ):
            raise ValueError(
                f"hooks_shell {hooks_shell} does not exist or is not executable"
            )
