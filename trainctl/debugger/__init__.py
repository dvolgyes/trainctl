"""Standalone, out-of-process tooling that attaches to a running Trainctl-managed training process.

Talks to the target over its REST API and OS-level process introspection
(py-spy, gdb). Never imported by `trainctl.mixin`/`trainctl.runtime`/`trainctl.fuse`
-- see `tests/debugger/import_boundary_test.py`.
"""
