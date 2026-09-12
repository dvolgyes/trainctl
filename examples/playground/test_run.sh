#!/usr/bin/env bash
# Runs the trainctl playground in the foreground: Lightning/PyTorch logs print to
# this terminal, and Ctrl+C stops training gracefully. Inspect the live run from a
# second terminal using the REST/FUSE commands printed at startup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"
exec uv run python -m examples.playground.run
