"""ShellCheck coverage for the rendered disabled-example hook scripts.

The templates themselves (`trainctl/hooks/templates/*.sh.tmpl`) are never
committed with a `.sh` suffix or an execute bit -- pre-commit's
`check-shebang-scripts-are-executable` would otherwise refuse them. This test
renders a full disabled tree the same way `bootstrap_hooks_tree` does for a
real run and ShellChecks every generated script instead.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from loguru import logger

from trainctl.hooks.catalog import MODALITIES
from trainctl.hooks.manager import bootstrap_hooks_tree

_SHELLCHECK = shutil.which("shellcheck")


def test_rendered_examples_pass_shellcheck(tmp_path: Path) -> None:
    if _SHELLCHECK is None:
        pytest.skip("shellcheck is not on PATH in this environment")

    live_dir = bootstrap_hooks_tree(tmp_path / "run", None, session_id="s-1", log=logger)
    scripts = sorted(
        str(path) for modality in MODALITIES for path in (live_dir / modality).glob("*.sh")
    )
    assert scripts

    result = subprocess.run(
        [_SHELLCHECK, "--shell=bash", *scripts],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
