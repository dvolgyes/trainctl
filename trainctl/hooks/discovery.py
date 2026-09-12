"""Enable/disable validation for one script, plus a rate-limited scan for unknown filenames.

The execute bit is the only enable switch, re-checked immediately before every use
(the current stat is never trusted from an earlier check or cache): `chmod +x`/`-x`
take effect at the next applicable callback, including within the same epoch. The
directory scan is a separate, coarser-grained diagnostic: it costs an `os.scandir`
per modality and is rate-limited to once per logical epoch token so a high-frequency
callback (e.g. `on_train_batch_start`) doesn't pay for it every invocation.
"""

import os
import stat as stat_module
from pathlib import Path
from typing import Any

from trainctl.hooks.catalog import CALLBACK_NAMES, MODALITIES


def resolve_enabled_script(live_dir: Path, modality: str, name: str) -> Path | None:
    """Returns `live_dir/modality/name.sh` if it is currently enabled, else `None`.

    "Enabled" means a regular file, not a symlink, with at least one POSIX execute
    bit set. Callers must call this again immediately before launch -- the exec bit,
    or the file itself, can change at any moment between checks.
    """
    script_path = live_dir / modality / f"{name}.sh"
    try:
        st = os.lstat(script_path)
    except OSError:
        return None
    if stat_module.S_ISLNK(st.st_mode) or not stat_module.S_ISREG(st.st_mode):
        return None
    if not st.st_mode & 0o111:
        return None
    return script_path


class EpochScanState:  # pylint: disable=too-few-public-methods
    """Rate-limits the unknown-filename scan to once per distinct epoch token.

    Attributes:
        last_epoch_token: The most recently scanned token, or `None` before the
            first scan. Callers pass `(trainer.state.fn, trainer.current_epoch)`.
    """

    def __init__(self) -> None:
        self.last_epoch_token: tuple[Any, ...] | None = None
        self._warned_paths: set[str] = set()

    def scan_if_due(
        self, live_dir: Path, epoch_token: tuple[Any, ...], log: Any
    ) -> None:
        """Scans `light/` and `heavy/` once per distinct `epoch_token`.

        Warns once (ever, per relative path) about a file directly under a modality
        directory whose name does not match a known callback -- the same diagnostic
        Increment I1 gives at bootstrap time, extended to files an operator adds
        after bootstrap. Never raises: an unreadable modality directory is skipped.
        """
        if epoch_token == self.last_epoch_token:
            return
        self.last_epoch_token = epoch_token
        for modality in MODALITIES:
            self._scan_modality(live_dir / modality, modality, log)

    def _scan_modality(self, modality_dir: Path, modality: str, log: Any) -> None:
        try:
            entries = os.scandir(modality_dir)
        except OSError:
            return
        with entries:
            for entry in entries:
                self._check_entry(entry, modality, log)

    def _check_entry(self, entry: os.DirEntry[str], modality: str, log: Any) -> None:
        if not entry.is_file(follow_symlinks=False):
            return
        stem = entry.name[: -len(".sh")] if entry.name.endswith(".sh") else ""
        if stem in CALLBACK_NAMES:
            return
        relative = f"{modality}/{entry.name}"
        if relative in self._warned_paths:
            return
        self._warned_paths.add(relative)
        log.warning(
            "hooks: {} does not match a known callback name and will never be "
            "dispatched",
            relative,
        )
