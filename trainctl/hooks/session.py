"""`HookSession`: the per-run object that owns the live hooks tree and dispatches
one Lightning callback occurrence to its light (and, from Increment I3, heavy)
script.

Owned by `TrainctlRuntime` (Increment I4), constructed once per run right after
`manager.bootstrap_hooks_tree` publishes or resumes the live tree. Lightning-agnostic:
`handle()` takes plain values, not a `Trainer`/`LightningModule`, so it is testable
without constructing either.
"""

import shutil
from pathlib import Path
from typing import Any

from trainctl.hooks.catalog import CALLBACK_BY_NAME
from trainctl.hooks.discovery import EpochScanState
from trainctl.hooks.dispatch import HandleResult, dispatch_heavy, dispatch_light
from trainctl.hooks.payload import OccurrenceContext, new_id


class HookSession:
    """Dispatches lifecycle callback occurrences against one run's live hooks tree.

    Attributes:
        live_dir: The published `hooks/` tree this session dispatches against.
        session_id: This run's hooks session id (shared with the bootstrap marker).
        rank: This process's `trainer.global_rank`.
        world_size: This process's `trainer.world_size`.
    """

    def __init__(
        self,
        live_dir: Path,
        *,
        session_id: str,
        rank: int,
        world_size: int,
        rank_policy: str,
        shell: Path,
        timeout_s: float | None,
        max_params_bytes: int,
        max_metadata_items: int,
        max_metadata_depth: int,
        max_export_bytes: int,
        max_output_bytes: int,
        log: Any,
    ) -> None:
        self.live_dir = live_dir
        self.session_id = session_id
        self.rank = rank
        self.world_size = world_size
        self._rank_policy = rank_policy
        self._shell = shell
        self._timeout_s = timeout_s
        self._max_params_bytes = max_params_bytes
        self._max_metadata_items = max_metadata_items
        self._max_metadata_depth = max_metadata_depth
        self._max_export_bytes = max_export_bytes
        self._max_output_bytes = max_output_bytes
        self._log = log
        self._epoch_scan = EpochScanState()
        self._tmp_root = live_dir.parent / ".hooks-tmp"

    def handle(
        self,
        hook: str,
        *,
        stage: str | None,
        epoch: int | None,
        global_step: int | None,
        batch_idx: int | None = None,
        dataloader_idx: int | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> HandleResult | None:
        """Dispatches one callback occurrence's light and heavy hooks, if this rank is eligible.

        Light and heavy share this firing's occurrence id but get distinct invocation
        ids and capture timestamps -- light runs first, so the two are not a single
        simultaneous snapshot. Heavy's tensor selection comes from the catalogue's
        static `tensor_args` for `hook`; a callback with none still dispatches heavy
        (a valid empty manifest) if its script is enabled.

        Args:
            hook: The exact `Callback` method name.
            stage: `str(trainer.state.fn)`, or `None` if unavailable.
            epoch: `trainer.current_epoch`, or `None`.
            global_step: `trainer.global_step`, or `None`.
            batch_idx: The batch index Lightning passed, if any.
            dataloader_idx: The dataloader index Lightning passed, if any.
            arguments: This callback's own named arguments, for the bounded
                metadata description and, for tensor-carrying arguments, heavy
                export. Defaults to an empty mapping.

        Returns:
            `None` if this rank is ineligible under the configured rank policy.
            Otherwise a `HandleResult` whose `light`/`heavy` fields are each `None`
            when that modality's script isn't currently enabled.
        """
        if self._rank_policy == "rank_zero" and self.rank != 0:
            return None
        self._epoch_scan.scan_if_due(self.live_dir, (stage, epoch), self._log)
        context = OccurrenceContext(
            session_id=self.session_id,
            hook=hook,
            stage=stage,
            epoch=epoch,
            global_step=global_step,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
            rank=self.rank,
            world_size=self.world_size,
        )
        self._tmp_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        occurrence_id = new_id()
        args = arguments or {}
        light_result = dispatch_light(
            self.live_dir,
            context,
            args,
            occurrence_id=occurrence_id,
            shell=self._shell,
            timeout_s=self._timeout_s,
            max_params_bytes=self._max_params_bytes,
            max_metadata_items=self._max_metadata_items,
            max_metadata_depth=self._max_metadata_depth,
            max_output_bytes=self._max_output_bytes,
            tmp_root=self._tmp_root,
            log=self._log,
        )
        spec = CALLBACK_BY_NAME.get(hook)
        tensor_arg_names = spec.tensor_args if spec is not None else ()
        heavy_result = dispatch_heavy(
            self.live_dir,
            context,
            occurrence_id,
            args,
            tensor_arg_names,
            shell=self._shell,
            timeout_s=self._timeout_s,
            max_metadata_items=self._max_metadata_items,
            max_metadata_depth=self._max_metadata_depth,
            max_export_bytes=self._max_export_bytes,
            max_output_bytes=self._max_output_bytes,
            tmp_root=self._tmp_root,
            log=self._log,
        )
        return HandleResult(light=light_result, heavy=heavy_result)

    def close(self) -> None:
        """Removes this session's private temporary-invocation root, if created.

        Idempotent. Any invocation directory still present here is a leak from an
        invocation that did not clean up after itself (e.g. a hard-killed process);
        this only ever removes paths under this run's own `.hooks-tmp/`.
        """
        shutil.rmtree(self._tmp_root, ignore_errors=True)
