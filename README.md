# trainctl

Trainctl attaches operational tooling to a PyTorch Lightning run: an inspection filesystem, a
REST control/debug API, checkpoint/artifact management, pipeline-pressure timing, backward-
exception capture, and a lifecycle shell-hooks facility for live, script-based debugging. Enable
services individually through `TrainctlConfig` / `TrainctlMixin` keyword arguments.

## Lifecycle shell hooks

An operator debugs a running Lightning job by editing ordinary shell scripts that live beside the
run's logs — no code changes, no restart. Every Lightning `Callback` method has a slot:
`hooks/light/<hook>.sh` (dispatched with a bounded JSON description of the callback's own
arguments) and `hooks/heavy/<hook>.sh` (dispatched with that same description plus the callback's
actual tensors, exported as `.npy` files). Scripts run **synchronously**, on the training thread,
at the real callback point.

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

### The heavy hook manifest

A heavy hook is invoked as `bash <script> <manifest.json>`. Its tensor selection is static per
callback (the catalogue's `tensor_args`, e.g. `batch` for `on_train_batch_start`, `loss` for
`on_before_backward`, `checkpoint` for `on_save_checkpoint`) — every tensor found nested inside
those named arguments is exported; a callback with no tensor-carrying arguments (or none present
this firing) still runs its heavy script, with a valid empty export. Light and heavy share one
firing's `occurrence_id` but get distinct `invocation_id`s and capture timestamps — light runs
first, so the two are not a single simultaneous snapshot.

```json
{
  "schema_version": 1,
  "session_id": "...", "occurrence_id": "...", "invocation_id": "...",
  "hook": "on_train_batch_start", "modality": "heavy",
  "timestamp": 1234567890.123, "pid": 12345,
  "rank": 0, "world_size": 1,
  "stage": "fit", "epoch": 0, "global_step": 41, "batch_idx": 41, "dataloader_idx": null,
  "arguments": { "batch": { "image": { "kind": "tensor", "shape": [4, 3, 8, 8], "...": "..." } } },
  "tensor_records": [
    { "argument_path": "batch/image", "file": "tensor-000001.npy",
      "shape": [4, 3, 8, 8], "dtype": "float32", "original_dtype": null, "device": "cpu" }
  ],
  "truncated": false, "truncated_reason": null
}
```

- `.npy` files are written with `numpy.save(..., allow_pickle=False)` — never a pickled Python
  object — and named sequentially (`tensor-000001.npy`, ...), never derived from an argument name
  or dict key. `argument_path` (e.g. `batch/image`) is how a script correlates a tensor record back
  to its place in the (separately described, non-tensor) `arguments` structure.
- `bfloat16` tensors are converted to `float32` for export (numpy has no native `bfloat16`); this
  is recorded per-tensor as `original_dtype`. No other conversion happens.
- **All-or-failed capture:** before any tensor is copied off its device, every tensor the callback
  would export is checked for a supported layout (dense/strided; sparse, quantized, meta-device,
  and nested tensors are rejected) and the predicted total converted size is checked against
  `hooks_max_export_bytes` (default 1 GiB). If any tensor is unsupported or the budget would be
  exceeded, the *entire* heavy capture fails — no partial manifest, no files written, the script is
  never launched — and this is logged as one `ERROR` record (`phase="capture_failed"`) naming the
  offending argument or the exceeded limit.
- The private per-invocation directory (manifest plus `.npy` files) exists for the heavy script's
  entire supervised lifetime and is removed afterward, on success, failure, or timeout alike — the
  same lifecycle the light hook's `params.json` directory has.

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

### Attaching to a Lightning `Trainer`

`TrainctlMixin.configure_callbacks()` appends one Trainctl hooks `Callback` instance (built once
per model instance and reused across `fit()` → `validate()` → `test()` → `predict()`) alongside
whatever your own `configure_callbacks()` override returns via `super()`. Nothing about your
model's or Trainer's own callbacks is removed or replaced.

The hooks session (the live `hooks/` tree plus the `trainctl.log` / `trainctl-rank-N.log` Loguru
bridge) is bootstrapped lazily, on the first callback firing observed for a run — normally the
adapter's own `setup(trainer, pl_module, stage)`, which Lightning calls *before* the composed
model's own `setup`. Rank and world size come directly from `trainer.global_rank` /
`trainer.world_size` at that point, not from any later Trainctl-internal rank-gathering step. A
resumed run (e.g. `fit()` followed by a separate `test()` call against the same model) detects the
existing marker and reuses the same live tree and session id rather than reseeding.

The hooks/logging session is finalized (temporary invocation directories removed, the logging
bridge restored) at the end of `TrainctlRuntime.teardown()` on the normal path. On an uncaught
exception, Lightning skips both the model's and the callback's own `teardown` entirely — so the
adapter's `on_exception` dispatches the `on_exception` light/heavy hooks itself and then finalizes
the session directly, in a `finally`, before the original exception propagates.

### Status

Shipped: config surface and validation, the Lightning-logging-to-Loguru bridge
(`intercept_lightning_logging`), the run-local `hooks/` tree bootstrap and seeding, discovery and
enablement semantics, light/heavy dispatch (including tensor export) with the supervised runner
described above, and the Lightning `Callback` adapter wired in through `configure_callbacks()`
(both `lightning.pytorch` and `pytorch_lightning`). Exercised directly by `tests/hooks/*_test.py`
plus an in-process `Trainer.fit`-based end-to-end suite. The example walkthrough (playground/CIFAR
updates) lands in the next increment.
