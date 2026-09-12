# trainctl

Trainctl attaches operational tooling to a PyTorch Lightning run: an inspection filesystem, a
REST control/debug API, checkpoint/artifact management, pipeline-pressure timing, backward-
exception capture, and a lifecycle shell-hooks facility for live, script-based debugging. Enable
services individually through `TrainctlConfig` / `TrainctlMixin` keyword arguments.

## Lifecycle shell hooks

An operator debugs a running Lightning job by editing ordinary shell scripts that live beside the
run's logs — no code changes, no restart. Every Lightning `Callback` method has a slot:
`hooks/light/<hook>.sh` (dispatched with a bounded JSON description of the callback's own
arguments) and `hooks/heavy/<hook>.sh` (reserved for tensor export; not yet implemented — see
Status below). Scripts run **synchronously**, on the training thread, at the real callback point.

### Enabling the facility

`hooks_enabled=True` (the default) makes `TrainctlMixin` seed a run-local `hooks/` tree the first
time it bootstraps for a run:

- `hooks_source_dir`: an optional baseline directory (`light/`, `heavy/`, and any supporting
  files) copied into the run's `hooks/` tree. Any `(modality, callback)` slot the baseline doesn't
  provide is seeded from a disabled example template. Never executed in place — only the *copy*
  under the run directory runs.
- The resulting tree has one script per callback per modality (currently 37 callbacks × 2
  modalities = 74 files), plus a generated `hooks/README.md` and a `trainctl-hooks-state.json`
  marker recording the session id, catalogue version, and source path, so a resumed run can detect
  whether it's compatible with the existing tree.
- Every seeded script starts **disabled** (mode `0o644`, no execute bit) regardless of whether it
  came from the baseline or a template.

### Enabling and disabling a hook

**The POSIX execute bit is the only switch.** To turn a hook on:

```sh
chmod +x <run-dir>/hooks/light/on_train_batch_end.sh
```

To turn it back off:

```sh
chmod -x <run-dir>/hooks/light/on_train_batch_end.sh
```

This takes effect at the *next* firing of that callback — including within the same epoch. The
execute bit is re-checked immediately before every dispatch; nothing about enablement is cached
from an earlier check. A regular file with at least one execute bit set is enabled; a symlink,
directory, or non-regular file is never treated as enabled even if its target would qualify.

Once per distinct `(stage, epoch)` — not per callback firing — trainctl also scans `light/` and
`heavy/` for files that don't match a known callback name and logs one warning per such path (so a
typo like `on_train_batch_ned.sh` is caught even though it will never run).

### The light hook payload

A light hook is invoked as `bash <script> <params.json>`, where `params.json` is written to a
private, per-invocation temporary directory (removed after the script exits) and contains:

```json
{
  "schema_version": 1,
  "session_id": "...", "occurrence_id": "...", "invocation_id": "...",
  "hook": "on_train_batch_end",
  "modality": "light",
  "timestamp": 1234567890.123, "pid": 12345,
  "rank": 0, "world_size": 1,
  "stage": "fit", "epoch": 0, "global_step": 41,
  "batch_idx": 41, "dataloader_idx": null,
  "arguments": { "...": "a bounded description of the callback's own arguments" },
  "truncated": false, "truncated_reason": null
}
```

A tensor argument is *described*, never evaluated — `{"kind": "tensor", "shape": [...], "dtype":
"...", "device": "...", "requires_grad": false}`. No `.item()`, `.tolist()`, or `.cpu()` runs on
this path. Nested mappings/lists/tuples are traversed up to `hooks_max_metadata_items` /
`hooks_max_metadata_depth`; a cycle is reported as `{"kind": "cycle"}` rather than recursing.
If the encoded document would exceed `hooks_max_params_bytes`, `arguments` is dropped entirely
(not partially re-truncated) and `truncated_reason` records why.

### Timeouts, output, and failure

- `hooks_timeout_s` (default `None`, unbounded) bounds how long training waits for a script.
  Unbounded is deliberate for interactive debugging but can stall a distributed job's watchdog
  indefinitely — set an explicit timeout outside interactive sessions.
- A script's stdout/stderr are streamed live to the run's log as they're produced, and up to
  `hooks_max_output_bytes` of each stream is retained as a tail for the failure record.
- On timeout, the script's *entire process group* is terminated (`SIGTERM`, then `SIGKILL` after a
  grace period) — a backgrounded child cannot outlive its script.
- A nonzero exit code, a timeout, or a failure to launch the script at all produces exactly one
  structured `ERROR` log record naming the hook, script, session/occurrence/invocation ids, rank,
  stage/epoch/global_step, phase, exit code, signal, duration, and the retained stdout/stderr
  tails. A disabled or successful (exit `0`) invocation logs nothing at that level.
- `hooks_rank_policy` (`"rank_zero"` default, or `"all"`) controls which ranks dispatch hooks at
  all.

### Status

Shipped: config surface and validation, the Lightning-logging-to-Loguru bridge
(`intercept_lightning_logging`), the run-local `hooks/` tree bootstrap and seeding, discovery and
enablement semantics, and light-hook dispatch with the supervised runner described above. **Not
yet wired into a live Lightning `Trainer`** — the pieces above are exercised directly by
`tests/hooks/*_test.py`, not yet attached via `configure_callbacks()`. Heavy (tensor-export) hooks,
the Lightning adapter, and the end-to-end operator walkthrough land in subsequent increments.
