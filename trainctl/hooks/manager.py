"""Bootstraps a run-local, editable `hooks/` tree, optionally seeded from a baseline directory.

Owns exactly the filesystem lifecycle described in `LIFECYCLE_HOOKS_PLAN.md` sections 3 and 9:
staged construction, ownership/execution-bit preservation, disabled-example seeding, atomic
publication, and resume/conflict detection through a small external marker plus an internal
provenance record. It does not scan for enabled scripts or dispatch anything -- see
`trainctl/hooks/catalog.py` for the callback catalogue this module seeds slots for, and later
increments for discovery and execution.
"""

import json
import os
import shutil
import socket
import stat as stat_module
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

from trainctl.hooks.catalog import (
    CALLBACK_NAMES,
    CALLBACKS,
    CATALOGUE_VERSION,
    MODALITIES,
)

_SCHEMA_VERSION = 1
_MARKER_NAME = "trainctl-hooks-state.json"
_PROVENANCE_NAME = ".trainctl-provenance.json"
_DEFAULT_DIR_MODE = 0o755
_SHEBANG = "#!/usr/bin/env bash\n"


class HooksBootstrapError(RuntimeError):
    """A baseline copy or live-tree bootstrap step failed in a way that must not be retried
    silently.
    """


def bootstrap_hooks_tree(
    base_dir: Path,
    source_dir: Path | None,
    *,
    session_id: str,
    log: Any,
) -> Path:
    """Ensures `base_dir/hooks` exists, seeded/copied as needed, and returns its path.

    Idempotent: once `base_dir` holds a compatible marker (`trainctl-hooks-state.json`), a later
    call returns the existing live tree unchanged, regardless of `source_dir` -- resuming a run
    never re-copies, re-seeds, or restores a deleted script.

    Args:
        base_dir: The resolved run directory (typically the Trainer's log directory); this
            function may create it.
        source_dir: Optional baseline directory (`light/`, `heavy/`, and supporting files),
            copied into the live tree once. Never executed in place and never modified.
        session_id: This run's collision-resistant identity, recorded in the marker/provenance.
        log: A bound logger (e.g. `LoggingSession.bind()`) used for warnings/info; never a bare
            module-level logger, so callers control attribution.

    Returns:
        The live `hooks/` directory path (`base_dir / "hooks"`).

    Raises:
        HooksBootstrapError: `source_dir` is unsafe relative to `base_dir` (equal, containing, or
            contained by it), contains a symlink/special file/privileged mode bit, changed while
            being copied, or an existing `hooks/` tree is neither a compatible resume nor a fresh
            publish target (unknown-origin conflict, or a concurrent initializer won the race).
        PermissionError: Preserving a copied entry's ownership failed (message includes the path
            and the ownership that was required).
    """
    live_dir = base_dir / "hooks"
    marker_path = base_dir / _MARKER_NAME

    if live_dir.is_symlink():
        raise HooksBootstrapError(
            f"{live_dir} is a symlink, which is not supported as the live hooks tree"
        )

    if marker_path.exists():
        record = _try_read_record(marker_path)
        if record is None:
            raise HooksBootstrapError(
                f"{marker_path} exists but is not a readable, compatible Trainctl hooks marker; "
                "refusing to reuse or overwrite it"
            )
        if source_dir is not None and record.get("source_dir") != str(source_dir):
            log.warning(  # noqa: PLE1205 -- loguru brace style, not stdlib %-logging
                "hooks_source_dir {} differs from the baseline recorded in {} for this run ({}); "
                "the existing live tree is reused unchanged",
                source_dir,
                marker_path,
                record.get("source_dir"),
            )
        return live_dir

    resolved_source = None
    if source_dir is not None:
        resolved_source = _resolve_and_validate_source(source_dir, base_dir)

    if live_dir.exists():
        record = _try_read_record(live_dir / _PROVENANCE_NAME)
        if record is not None:
            _write_json(marker_path, record)
            return live_dir
        raise HooksBootstrapError(
            f"{live_dir} already exists but is not a Trainctl-owned hooks tree (no compatible "
            f"{marker_path.name} or {_PROVENANCE_NAME} found); refusing to reuse or overwrite it. "
            "Use a fresh run directory, or remove the conflicting tree yourself."
        )

    base_dir.mkdir(parents=True, exist_ok=True)
    live_dir = _publish_fresh_tree(base_dir, live_dir, resolved_source, session_id, log)
    return live_dir


def _publish_fresh_tree(
    base_dir: Path,
    live_dir: Path,
    resolved_source: Path | None,
    session_id: str,
    log: Any,
) -> Path:
    marker_path = base_dir / _MARKER_NAME
    staging_dir = Path(tempfile.mkdtemp(dir=base_dir, prefix=".hooks-staging-"))
    os.chmod(staging_dir, _DEFAULT_DIR_MODE)
    published = False
    try:
        (staging_dir / "light").mkdir(mode=_DEFAULT_DIR_MODE, exist_ok=True)
        (staging_dir / "heavy").mkdir(mode=_DEFAULT_DIR_MODE, exist_ok=True)

        found_slots: set[tuple[str, str]] = set()
        unknown_paths: list[str] = []
        dir_records: list[tuple[Path, int, int, int]] = []

        if resolved_source is not None:
            _copy_tree(
                resolved_source,
                staging_dir,
                rel=(),
                found_slots=found_slots,
                unknown_paths=unknown_paths,
                dir_records=dir_records,
            )
            top_st = os.lstat(resolved_source)
            dir_records.append(
                (
                    staging_dir,
                    top_st.st_uid,
                    top_st.st_gid,
                    stat_module.S_IMODE(top_st.st_mode),
                )
            )
        if unknown_paths:
            log.warning(  # noqa: PLE1205 -- loguru brace style, not stdlib %-logging
                "hooks_source_dir has {} file(s) directly under light/ or heavy/ that do not "
                "match a known callback name and will never be dispatched: {}",
                len(unknown_paths),
                ", ".join(sorted(unknown_paths)),
            )

        _seed_missing_slots(staging_dir, found_slots)
        _write_readme_if_absent(staging_dir)

        record = {
            "schema_version": _SCHEMA_VERSION,
            "catalogue_version": CATALOGUE_VERSION,
            "session_id": session_id,
            "source_dir": str(resolved_source) if resolved_source is not None else None,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        }
        _write_json(staging_dir / _PROVENANCE_NAME, record)

        _finalize_directory_modes(dir_records)

        try:
            staging_dir.rename(live_dir)
            published = True
        except OSError as exc:
            raise HooksBootstrapError(
                f"cannot publish {staging_dir} to {live_dir}: a concurrent initializer likely "
                f"already published a live hooks tree there ({exc})"
            ) from exc
    finally:
        if not published:
            shutil.rmtree(staging_dir, ignore_errors=True)

    _write_json(marker_path, record)
    log.info(  # noqa: PLE1205 -- loguru brace style, not stdlib %-logging
        "hooks: bootstrapped live tree at {} (source={})", live_dir, resolved_source
    )
    return live_dir


def _resolve_and_validate_source(source_dir: Path, base_dir: Path) -> Path:
    resolved_source = source_dir.resolve(strict=True)
    resolved_base = base_dir.resolve()
    _reject_unsafe(resolved_source, os.lstat(resolved_source))
    if not resolved_source.is_dir():
        raise HooksBootstrapError(f"hooks_source_dir is not a directory: {source_dir}")
    if (
        resolved_source == resolved_base
        or resolved_source.is_relative_to(resolved_base)
        or resolved_base.is_relative_to(resolved_source)
    ):
        raise HooksBootstrapError(
            f"hooks_source_dir {source_dir} must not equal, contain, or be contained by the run "
            f"directory {base_dir}"
        )
    return resolved_source


def _reject_unsafe(path: Path, st: os.stat_result) -> None:
    if stat_module.S_ISLNK(st.st_mode):
        raise HooksBootstrapError(
            f"hooks_source_dir contains a symlink, which is not supported: {path}"
        )
    if not (stat_module.S_ISREG(st.st_mode) or stat_module.S_ISDIR(st.st_mode)):
        raise HooksBootstrapError(
            f"hooks_source_dir contains a non-regular file, which is not supported: {path}"
        )
    if st.st_mode & (stat_module.S_ISUID | stat_module.S_ISGID | stat_module.S_ISVTX):
        raise HooksBootstrapError(
            f"hooks_source_dir contains privileged mode bits, which is not supported: {path}"
        )


def _copy_tree(
    src_dir: Path,
    dst_dir: Path,
    *,
    rel: tuple[str, ...],
    found_slots: set[tuple[str, str]],
    unknown_paths: list[str],
    dir_records: list[tuple[Path, int, int, int]],
) -> None:
    with os.scandir(src_dir) as it:
        entries = sorted(it, key=lambda e: e.name)
    for entry in entries:
        st = os.lstat(entry.path)
        entry_path = Path(entry.path)
        _reject_unsafe(entry_path, st)
        child_rel = (*rel, entry.name)
        dst_path = dst_dir / entry.name
        if stat_module.S_ISDIR(st.st_mode):
            dst_path.mkdir(mode=_DEFAULT_DIR_MODE, exist_ok=True)
            _copy_tree(
                entry_path,
                dst_path,
                rel=child_rel,
                found_slots=found_slots,
                unknown_paths=unknown_paths,
                dir_records=dir_records,
            )
            dir_records.append(
                (dst_path, st.st_uid, st.st_gid, stat_module.S_IMODE(st.st_mode))
            )
        else:
            shutil.copyfile(entry.path, dst_path)
            post = os.lstat(entry.path)
            if (st.st_size, st.st_mtime_ns) != (post.st_size, post.st_mtime_ns):
                raise HooksBootstrapError(
                    f"hooks_source_dir file {entry.path} changed while being copied; "
                    "do not edit the baseline during initialization"
                )
            _apply_ownership_and_mode(
                dst_path, st.st_uid, st.st_gid, stat_module.S_IMODE(st.st_mode)
            )
            if len(child_rel) == 2 and child_rel[0] in MODALITIES:
                stem = child_rel[1][:-3] if child_rel[1].endswith(".sh") else ""
                if stem in CALLBACK_NAMES:
                    found_slots.add((child_rel[0], stem))
                else:
                    unknown_paths.append("/".join(child_rel))


def _finalize_directory_modes(dir_records: list[tuple[Path, int, int, int]]) -> None:
    """Applies each recorded directory's preserved ownership/mode, deepest first."""
    for dst_path, uid, gid, mode in dir_records:
        _apply_ownership_and_mode(dst_path, uid, gid, mode)


def _apply_ownership_and_mode(path: Path, uid: int, gid: int, mode: int) -> None:
    current = path.stat()
    if current.st_uid != uid or current.st_gid != gid:
        try:
            os.chown(path, uid, gid)
        except PermissionError as exc:
            raise PermissionError(
                f"cannot set ownership of {path} to uid={uid}, gid={gid}: {exc}"
            ) from exc
    os.chmod(path, mode)


def _seed_missing_slots(staging_dir: Path, found_slots: set[tuple[str, str]]) -> None:
    templates = {
        modality: _load_template(f"{modality}.sh.tmpl") for modality in MODALITIES
    }
    for spec in CALLBACKS:
        for modality in MODALITIES:
            if (modality, spec.name) in found_slots:
                continue
            script_path = staging_dir / modality / f"{spec.name}.sh"
            script_path.write_text(_SHEBANG + _render(templates[modality], spec))
            script_path.chmod(0o644)


def _render(template_text: str, spec: Any) -> str:
    return (
        template_text.replace("@@NAME@@", spec.name)
        .replace("@@GROUP@@", spec.group)
        .replace("@@DESCRIPTION@@", spec.description)
    )


def _write_readme_if_absent(staging_dir: Path) -> None:
    readme_path = staging_dir / "README.md"
    if readme_path.exists():
        return
    template = _load_template("README.md.tmpl")
    slot_count = len(CALLBACKS) * len(MODALITIES)
    content = (
        template.replace("@@CATALOGUE_VERSION@@", str(CATALOGUE_VERSION))
        .replace("@@CALLBACK_COUNT@@", str(len(CALLBACKS)))
        .replace("@@SLOT_COUNT@@", str(slot_count))
    )
    readme_path.write_text(content)


def _load_template(name: str) -> str:
    return (resources.files("trainctl.hooks") / "templates" / name).read_text()


def _try_read_record(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not (
        isinstance(record, dict)
        and record.get("schema_version") == _SCHEMA_VERSION
        and isinstance(record.get("session_id"), str)
    ):
        return None
    return record


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
