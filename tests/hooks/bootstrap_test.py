"""Behavioral tests for trainctl.hooks.manager's baseline bootstrap (T1)."""

import json
import os
import shutil
import stat

import pytest
from loguru import logger

from trainctl.hooks.catalog import CALLBACKS, MODALITIES
from trainctl.hooks.manager import (
    HooksBootstrapError,
    _publish_fresh_tree,
    bootstrap_hooks_tree,
)

_SLOT_COUNT = len(CALLBACKS) * len(MODALITIES)


@pytest.fixture()
def log():
    return logger


def _all_slot_paths(live_dir) -> list:
    return [
        live_dir / modality / f"{spec.name}.sh"
        for spec in CALLBACKS
        for modality in MODALITIES
    ]


def test_no_source_creates_74_disabled_scripts_and_readme(tmp_path, log) -> None:
    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(base_dir, None, session_id="s-1", log=log)

    assert live_dir == base_dir / "hooks"
    slots = _all_slot_paths(live_dir)
    assert len(slots) == _SLOT_COUNT == 74
    for script in slots:
        assert script.is_file()
        assert stat.S_IMODE(script.stat().st_mode) & 0o111 == 0, (
            f"{script} must not be executable"
        )
        assert script.read_text().startswith("#!/usr/bin/env bash\n")
    assert (live_dir / "README.md").is_file()
    marker = json.loads((base_dir / "trainctl-hooks-state.json").read_text())
    assert marker["session_id"] == "s-1"
    assert marker["source_dir"] is None
    provenance = json.loads((live_dir / ".trainctl-provenance.json").read_text())
    assert provenance["session_id"] == "s-1"


def test_partial_source_fills_only_missing_slots_and_preserves_bytes_and_mode(
    tmp_path, log
) -> None:
    source = tmp_path / "baseline"
    (source / "light").mkdir(parents=True)
    (source / "heavy").mkdir(parents=True)
    custom_light = source / "light" / "on_fit_start.sh"
    custom_light.write_text("#!/usr/bin/env bash\necho custom-light\n")
    custom_light.chmod(0o755)
    custom_heavy = source / "heavy" / "on_train_batch_end.sh"
    custom_heavy.write_text("#!/usr/bin/env bash\necho custom-heavy\n")
    custom_heavy.chmod(0o750)

    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(base_dir, source, session_id="s-2", log=log)

    live_light = live_dir / "light" / "on_fit_start.sh"
    assert live_light.read_bytes() == custom_light.read_bytes()
    assert stat.S_IMODE(live_light.stat().st_mode) == 0o755

    live_heavy = live_dir / "heavy" / "on_train_batch_end.sh"
    assert live_heavy.read_bytes() == custom_heavy.read_bytes()
    assert stat.S_IMODE(live_heavy.stat().st_mode) == 0o750

    for script in _all_slot_paths(live_dir):
        if script in (live_light, live_heavy):
            continue
        assert stat.S_IMODE(script.stat().st_mode) & 0o111 == 0, (
            f"{script} must stay disabled"
        )


def test_helper_files_are_copied_recursively_with_preserved_mode(tmp_path, log) -> None:
    source = tmp_path / "baseline"
    (source / "helpers" / "nested").mkdir(parents=True)
    helper = source / "helpers" / "nested" / "notify.sh"
    helper.write_text("echo hi\n")
    helper.chmod(0o700)

    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(base_dir, source, session_id="s-3", log=log)

    live_helper = live_dir / "helpers" / "nested" / "notify.sh"
    assert live_helper.read_text() == "echo hi\n"
    assert stat.S_IMODE(live_helper.stat().st_mode) == 0o700


def test_symlink_in_source_is_rejected_and_nothing_is_published(tmp_path, log) -> None:
    source = tmp_path / "baseline"
    source.mkdir()
    target = tmp_path / "outside.sh"
    target.write_text("echo x\n")
    (source / "link.sh").symlink_to(target)

    base_dir = tmp_path / "run"
    with pytest.raises(HooksBootstrapError, match="symlink"):
        bootstrap_hooks_tree(base_dir, source, session_id="s-4", log=log)

    assert not (base_dir / "hooks").exists()
    assert not (base_dir / "trainctl-hooks-state.json").exists()
    assert list(base_dir.glob(".hooks-staging-*")) == []


def test_special_file_in_source_is_rejected(tmp_path, log) -> None:
    source = tmp_path / "baseline"
    source.mkdir()
    os.mkfifo(source / "pipe")

    base_dir = tmp_path / "run"
    with pytest.raises(HooksBootstrapError, match="non-regular file"):
        bootstrap_hooks_tree(base_dir, source, session_id="s-5", log=log)
    assert not (base_dir / "hooks").exists()


def test_setuid_bit_in_source_is_rejected(tmp_path, log) -> None:
    source = tmp_path / "baseline"
    source.mkdir()
    script = source / "weird.sh"
    script.write_text("echo x\n")
    script.chmod(0o755 | stat.S_ISUID)

    base_dir = tmp_path / "run"
    with pytest.raises(HooksBootstrapError, match="privileged mode bits"):
        bootstrap_hooks_tree(base_dir, source, session_id="s-6", log=log)
    assert not (base_dir / "hooks").exists()


def test_source_equal_to_run_directory_is_rejected(tmp_path, log) -> None:
    base_dir = tmp_path / "run"
    base_dir.mkdir()

    with pytest.raises(
        HooksBootstrapError, match="must not equal, contain, or be contained by"
    ):
        bootstrap_hooks_tree(base_dir, base_dir, session_id="s-7", log=log)


def test_source_inside_run_directory_is_rejected(tmp_path, log) -> None:
    base_dir = tmp_path / "run"
    nested_source = base_dir / "baseline"
    nested_source.mkdir(parents=True)

    with pytest.raises(
        HooksBootstrapError, match="must not equal, contain, or be contained by"
    ):
        bootstrap_hooks_tree(base_dir, nested_source, session_id="s-8", log=log)


def test_run_directory_inside_source_is_rejected(tmp_path, log) -> None:
    source = tmp_path / "baseline"
    base_dir = source / "nested" / "run"
    source.mkdir()

    with pytest.raises(
        HooksBootstrapError, match="must not equal, contain, or be contained by"
    ):
        bootstrap_hooks_tree(base_dir, source, session_id="s-9", log=log)


def test_repeat_setup_reuses_live_tree_and_never_restores_deleted_scripts(
    tmp_path, log
) -> None:
    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(base_dir, None, session_id="s-10", log=log)

    deleted = live_dir / "light" / "on_fit_start.sh"
    deleted.unlink()
    edited = live_dir / "light" / "on_fit_end.sh"
    edited.write_text("#!/usr/bin/env bash\necho edited\n")
    edited.chmod(0o755)

    live_dir_again = bootstrap_hooks_tree(
        base_dir, None, session_id="s-10-again", log=log
    )

    assert live_dir_again == live_dir
    assert not deleted.exists()
    assert edited.read_text() == "#!/usr/bin/env bash\necho edited\n"
    assert stat.S_IMODE(edited.stat().st_mode) & 0o111 != 0
    marker = json.loads((base_dir / "trainctl-hooks-state.json").read_text())
    assert marker["session_id"] == "s-10"  # first session's marker, not overwritten


def test_baseline_edits_after_bootstrap_do_not_propagate_on_resume(
    tmp_path, log
) -> None:
    source = tmp_path / "baseline"
    (source / "light").mkdir(parents=True)
    (source / "heavy").mkdir(parents=True)
    (source / "light" / "on_fit_start.sh").write_text(
        "#!/usr/bin/env bash\necho original\n"
    )

    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(base_dir, source, session_id="s-11", log=log)

    (source / "light" / "on_fit_start.sh").write_text(
        "#!/usr/bin/env bash\necho changed-after\n"
    )

    live_dir_again = bootstrap_hooks_tree(
        base_dir, source, session_id="s-11-again", log=log
    )

    assert live_dir_again == live_dir
    assert (
        live_dir / "light" / "on_fit_start.sh"
    ).read_text() == "#!/usr/bin/env bash\necho original\n"


def test_unknown_existing_live_tree_is_a_conflict_and_is_left_untouched(
    tmp_path, log
) -> None:
    base_dir = tmp_path / "run"
    live_dir = base_dir / "hooks"
    live_dir.mkdir(parents=True)
    (live_dir / "mystery.txt").write_text("not ours\n")

    with pytest.raises(HooksBootstrapError, match="not a Trainctl-owned hooks tree"):
        bootstrap_hooks_tree(base_dir, None, session_id="s-12", log=log)

    assert (live_dir / "mystery.txt").read_text() == "not ours\n"
    assert not (base_dir / "trainctl-hooks-state.json").exists()


def test_interrupted_publish_recovers_from_provenance_without_recopying(
    tmp_path, log
) -> None:
    base_dir = tmp_path / "run"
    live_dir = bootstrap_hooks_tree(base_dir, None, session_id="s-13", log=log)
    # Simulate a crash between the atomic rename and the external marker write.
    (base_dir / "trainctl-hooks-state.json").unlink()
    sentinel = live_dir / "light" / "on_fit_start.sh"
    sentinel.write_text("#!/usr/bin/env bash\necho untouched-by-recovery\n")

    live_dir_again = bootstrap_hooks_tree(
        base_dir, None, session_id="s-13-recovered", log=log
    )

    assert live_dir_again == live_dir
    marker = json.loads((base_dir / "trainctl-hooks-state.json").read_text())
    assert (
        marker["session_id"] == "s-13"
    )  # recovered from the live tree's own provenance
    assert sentinel.read_text() == "#!/usr/bin/env bash\necho untouched-by-recovery\n"


def test_concurrent_publish_race_is_rejected_and_cleans_up_staging(
    tmp_path, log
) -> None:
    base_dir = tmp_path / "run"
    base_dir.mkdir()
    live_dir = base_dir / "hooks"
    live_dir.mkdir()
    (live_dir / "already-here.txt").write_text("winner\n")

    with pytest.raises(HooksBootstrapError, match="concurrent initializer"):
        _publish_fresh_tree(base_dir, live_dir, None, "s-14", log)

    assert list(base_dir.glob(".hooks-staging-*")) == []
    assert not (base_dir / "trainctl-hooks-state.json").exists()
    assert (live_dir / "already-here.txt").read_text() == "winner\n"


def test_source_file_changing_during_copy_is_rejected(
    tmp_path, monkeypatch, log
) -> None:
    source = tmp_path / "baseline"
    (source / "light").mkdir(parents=True)
    (source / "heavy").mkdir(parents=True)
    changing = source / "light" / "on_fit_start.sh"
    changing.write_text("#!/usr/bin/env bash\necho v1\n")

    real_copyfile = shutil.copyfile

    def flaky_copyfile(src, dst):
        real_copyfile(src, dst)
        if str(src) == str(changing):
            with open(src, "a") as fh:
                fh.write("echo v2\n")

    monkeypatch.setattr("trainctl.hooks.manager.shutil.copyfile", flaky_copyfile)

    base_dir = tmp_path / "run"
    with pytest.raises(HooksBootstrapError, match="changed while being copied"):
        bootstrap_hooks_tree(base_dir, source, session_id="s-15", log=log)
    assert not (base_dir / "hooks").exists()
    assert list(base_dir.glob(".hooks-staging-*")) == []


def test_ownership_change_failure_is_reported_with_path_and_target_ids(
    tmp_path, monkeypatch, log
) -> None:
    from trainctl.hooks.manager import _apply_ownership_and_mode

    target = tmp_path / "file.sh"
    target.write_text("echo x\n")
    current = target.stat()

    def fake_chown(path, uid, gid):
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(os, "chown", fake_chown)

    with pytest.raises(PermissionError, match=str(target)):
        _apply_ownership_and_mode(target, current.st_uid + 1, current.st_gid, 0o644)
