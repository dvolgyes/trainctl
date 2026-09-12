"""Runs one hook script for one Lightning callback occurrence and reports failure.

Kept Lightning-agnostic: callers pass plain values already extracted from the
`Trainer`/`LightningModule` (Increment I4's adapter owns that extraction), so this
module -- and `HookSession` in `trainctl.hooks.session`, which calls it -- can be
exercised without constructing a real Trainer.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trainctl.hooks.discovery import resolve_enabled_script
from trainctl.hooks.payload import (
    HeavyCaptureFailure,
    OccurrenceContext,
    build_heavy_manifest,
    build_light_params,
    export_tensors,
    new_id,
    prepare_heavy_export,
    write_json_file,
)
from trainctl.hooks.runner import ExecutionResult, run_script

_FAILED_PHASES = ("timeout", "launch_error", "capture_failed")


@dataclass(frozen=True)
class HandleResult:
    """Both modalities' outcomes for one dispatched callback occurrence.

    Attributes:
        light: The light hook's `ExecutionResult`, or `None` if it wasn't enabled.
        heavy: The heavy hook's `ExecutionResult`, or `None` if it wasn't enabled.
    """

    light: ExecutionResult | None
    heavy: ExecutionResult | None


def dispatch_light(
    live_dir: Path,
    context: OccurrenceContext,
    arguments: dict[str, Any],
    *,
    occurrence_id: str,
    shell: Path,
    timeout_s: float | None,
    max_params_bytes: int,
    max_metadata_items: int,
    max_metadata_depth: int,
    max_output_bytes: int,
    tmp_root: Path,
    log: Any,
) -> ExecutionResult | None:
    """Builds `params.json` and runs the light hook for `context.hook`, if enabled.

    Revalidates enablement twice: once before spending time building the metadata
    (which is cheap, but heavy's counterpart is not), and again immediately before
    launch, per the discovery contract's re-check requirement.

    Args:
        occurrence_id: This callback firing's identity, shared with the heavy hook
            dispatched for the same occurrence (each still gets its own invocation id).

    Returns:
        `None` if the light script is not currently enabled (an ordinary skip, not
        logged as a failure); otherwise the completed `ExecutionResult`. A failing
        result is logged here as one ERROR record before being returned.
    """
    if resolve_enabled_script(live_dir, "light", context.hook) is None:
        return None
    invocation_id = new_id()
    params, _truncated, _reason = build_light_params(
        context=context,
        occurrence_id=occurrence_id,
        invocation_id=invocation_id,
        arguments=arguments,
        max_params_bytes=max_params_bytes,
        max_metadata_items=max_metadata_items,
        max_metadata_depth=max_metadata_depth,
    )
    script = resolve_enabled_script(live_dir, "light", context.hook)
    if script is None:
        return None
    invocation_dir = tmp_root / invocation_id
    invocation_dir.mkdir(mode=0o700, parents=True)
    try:
        params_path = write_json_file(invocation_dir / "params.json", params)
        result = run_script(
            script,
            [str(params_path)],
            cwd=live_dir,
            shell=shell,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
            log=log,
        )
    finally:
        shutil.rmtree(invocation_dir, ignore_errors=True)
    if _is_failure(result):
        _log_failure(
            script, "light", context, occurrence_id, invocation_id, result, log
        )
    return result


def dispatch_heavy(
    live_dir: Path,
    context: OccurrenceContext,
    occurrence_id: str,
    arguments: dict[str, Any],
    tensor_arg_names: tuple[str, ...],
    *,
    shell: Path,
    timeout_s: float | None,
    max_metadata_items: int,
    max_metadata_depth: int,
    max_export_bytes: int,
    max_output_bytes: int,
    tmp_root: Path,
    log: Any,
) -> ExecutionResult | None:
    """Builds `manifest.json` plus `.npy` tensors and runs the heavy hook, if enabled.

    Validates every tensor `context.hook` would export ("all or failed") before any
    host allocation; a callback with no tensor arguments produces a valid empty
    manifest rather than being skipped. A failed capture is logged as one ERROR
    record with `phase="capture_failed"` and the script is never launched.

    Returns:
        `None` if the heavy script is not currently enabled; otherwise the
        completed (or capture-failed) `ExecutionResult`.
    """
    if resolve_enabled_script(live_dir, "heavy", context.hook) is None:
        return None
    invocation_id = new_id()
    prepared = prepare_heavy_export(
        arguments,
        tensor_arg_names,
        max_items=max_metadata_items,
        max_depth=max_metadata_depth,
        max_export_bytes=max_export_bytes,
    )
    script = resolve_enabled_script(live_dir, "heavy", context.hook)
    if script is None:
        return None
    if isinstance(prepared, HeavyCaptureFailure):
        result = ExecutionResult(
            phase="capture_failed",
            exit_code=None,
            signal_number=None,
            duration_s=0.0,
            stdout_tail="",
            stderr_tail="",
            stdout_truncated=False,
            stderr_truncated=False,
            error=prepared.reason,
        )
        _log_failure(script, "heavy", context, occurrence_id, invocation_id, result, log)
        return result
    leaves, truncated, reason = prepared
    invocation_dir = tmp_root / invocation_id
    invocation_dir.mkdir(mode=0o700, parents=True)
    try:
        tensor_records = export_tensors(leaves, invocation_dir)
        manifest = build_heavy_manifest(
            context=context,
            occurrence_id=occurrence_id,
            invocation_id=invocation_id,
            arguments=arguments,
            tensor_arg_names=tensor_arg_names,
            tensor_records=tensor_records,
            max_metadata_items=max_metadata_items,
            max_metadata_depth=max_metadata_depth,
            truncated=truncated,
            truncated_reason=reason,
        )
        manifest_path = write_json_file(invocation_dir / "manifest.json", manifest)
        result = run_script(
            script,
            [str(manifest_path)],
            cwd=live_dir,
            shell=shell,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
            log=log,
        )
    finally:
        shutil.rmtree(invocation_dir, ignore_errors=True)
    if _is_failure(result):
        _log_failure(script, "heavy", context, occurrence_id, invocation_id, result, log)
    return result


def _is_failure(result: ExecutionResult) -> bool:
    return result.phase in _FAILED_PHASES or (
        result.phase == "completed" and result.exit_code != 0
    )


def _log_failure(
    script: Path,
    modality: str,
    context: OccurrenceContext,
    occurrence_id: str,
    invocation_id: str,
    result: ExecutionResult,
    log: Any,
) -> None:
    log.error(  # noqa: PLE1205 -- loguru brace style, not stdlib %-logging
        "hook failed: hook={} modality={} script={} session={} occurrence={} "
        "invocation={} rank={} stage={} epoch={} global_step={} phase={} exit_code={} "
        "signal={} duration_s={:.3f} error={} stdout_tail={!r} stderr_tail={!r}",
        context.hook,
        modality,
        script,
        context.session_id,
        occurrence_id,
        invocation_id,
        context.rank,
        context.stage,
        context.epoch,
        context.global_step,
        result.phase,
        result.exit_code,
        result.signal_number,
        result.duration_s,
        result.error,
        result.stdout_tail,
        result.stderr_tail,
    )
