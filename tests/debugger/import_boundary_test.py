"""Keeps trainctl.debugger out-of-process.

It must never import trainctl.runtime, trainctl.mixin, or trainctl.fuse -- it may
only talk to a target process over REST (`hold_client`) and OS-level attach
(py-spy/gdb against a PID). AST-based, not text grep, so prose mentions of these
module names in docstrings (there are several, describing why the boundary
exists) don't produce false positives.
"""

import ast
from pathlib import Path

import trainctl.debugger

_FORBIDDEN_PREFIXES = ("trainctl.runtime", "trainctl.mixin", "trainctl.fuse")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _is_forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in _FORBIDDEN_PREFIXES
    )


def test_debugger_package_has_no_trainer_side_imports() -> None:
    package_dir = Path(trainctl.debugger.__file__).parent
    violations = [
        (path, module)
        for path in sorted(package_dir.rglob("*.py"))
        for module in _imported_modules(path)
        if _is_forbidden(module)
    ]
    assert not violations, f"trainctl.debugger must stay out-of-process: {violations}"
