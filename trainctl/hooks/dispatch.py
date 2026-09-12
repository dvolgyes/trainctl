"""Runs one hook script for one Lightning callback occurrence and reports failure.

Kept Lightning-agnostic: callers pass plain values already extracted from the
`Trainer`/`LightningModule` (Increment I4's adapter owns that extraction), so this
module -- and `HookSession` in `trainctl.hooks.session`, which calls it -- can be
exercised without constructing a real Trainer.
"""

import shutil
from pathlib import Path
from typing import Any

from trainctl.hooks.discovery import resolve_enabled_script
from trainctl.hooks.payload import (
    OccurrenceContext,
    build_light_params,
    new_id,
    write_json_file,
)
from trainctl.hooks.runner import ExecutionResult, run_script

_FAILED_PHASES = ("timeout", "launch_error")


def dispatch_light(
    live_dir: Path,
    context: OccurrenceContext,
    arguments: dict[str, Any],
    *,
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
    (which is cheap, but heavy's future counterpart is not), and again immediately
    before launch, per the discovery contract's re-check requirement.

    Returns:
        `None` if the light script is not currently enabled (an ordinary skip, not
        logged as a failure); otherwise the completed `ExecutionResult`. A failing
        result is logged here as one ERROR record before being returned.
    """
    if resolve_enabled_script(live_dir, "light", context.hook) is None:
        return None
    occurrence_id = new_id()
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
