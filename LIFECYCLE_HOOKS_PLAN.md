# Lifecycle shell hooks and unified logging: implementation handover

Status: planning only. No hook implementation accompanies this document.

Prepared against the working tree on 2026-09-12, including its existing uncommitted changes. Installed Lightning
callback sources inspected: `lightning.pytorch` and `pytorch_lightning`, version 2.6.5. Recheck these integration points
before implementation because the dependency requirement permits later versions.

## 1. Outcome and scope

Let an operator inspect and debug a running Lightning job by editing ordinary shell scripts beside its training logs.
Each supported Lightning callback has a lightweight metadata hook and an optional tensor-exporting heavy hook. Scripts
run synchronously at the actual callback point; the training process continues after they finish. Failures are visible
through Loguru without replacing the original training failure.

The latest directory requirement supersedes an earlier proposal to execute directly from a configurable script
directory: the supplied directory is now a **baseline to copy**, never the live execution location.

This handover is complete when it defines the directory and payload contracts, callback integration, failure handling,
logging ownership, security boundaries, and dependency-ordered work with falsifiable acceptance checks. Feature
completion requires the implementation and tests below; a reviewed plan does not establish that those tests pass.

### Accepted requirements

| ID | Required behavior | Implementation / verification |
| -- | -- | -- |
| R1 | Dispatch at every supported Lightning lifecycle callback, in both supported namespaces. | P0, I4, T4 |
| R2 | Separate `light/<hook>.sh` and `heavy/<hook>.sh`; no tensor materialization on the light path. | I2, I3, T2, T3 |
| R3 | Copy an optional baseline into the run logs; preserve owner/group and executable bits, and execute only the copy. | I1, T1 |
| R4 | Seed missing hook scripts with disabled examples; with no baseline, seed all hooks. | I1, T1 |
| R5 | Discover filenames initially and once per epoch; recheck known files and permission bits before execution. | I2, T2 |
| R6 | Block until the subprocess finishes; pass heavy data through temporary files and then clean them up. | I2, I3, T3, T5 |
| R7 | Log unsuccessful executions with reason, exit status, output, and context through Loguru. | I0, I2, T5, T6 |
| R8 | Route Lightning's standard logging into Loguru; migrate remaining Trainctl diagnostic logging where necessary. | I0, I5, T6 |
| R9 | Defer native HTTP hooks; a shell script may use `curl`. | I5 documentation review |
| R10 | Preserve existing Trainctl controls, inspection behavior, user callbacks, and training semantics. | P0, I4, I5, T4, T7 |

“Missing hooks” means missing **per modality and callback**: supplying `light/on_fit_start.sh` does not suppress the
disabled example at `heavy/on_fit_start.sh`. This is the proposed interpretation of incomplete baseline coverage.

### Deliberately excluded

- Dataset wrappers, native `.GET`/`.POST` handlers, scheduling services, and remote script upload/edit APIs.
- Asynchronous dispatch, worker queues, background tensor export, and automatic retries.
- Returning values into Lightning, mutating its checkpoint dictionary through hook output, or controlling training
  through a shell script's exit code.
- Implicit export of the entire model, optimizer, or dataset at every callback.
- Sandboxing arbitrary scripts or guaranteeing their external side effects are reversible.

## 2. Current implementation and integration constraints

| Existing owner | Relevant observed behavior | Consequence |
| -- | -- | -- |
| [trainctl/mixin.py](trainctl/mixin.py) | Owns the model integration and forwards a subset of model hooks to the runtime. | Preserve these methods and their return values; do not expand the existing safe-point enum into a callback catalogue. |
| [trainctl/lightning_backend.py](trainctl/lightning_backend.py) | Resolves either Lightning namespace and rejects mixed backends. | A callback adapter must inherit from the selected backend's `Callback`. |
| [trainctl/config.py](trainctl/config.py) | Frozen configuration with explicit mixin constructor forwarding. | Add validated hook settings in both places without swallowing user model arguments. |
| [trainctl/runtime/runtime.py](trainctl/runtime/runtime.py) | Starts services in model `setup`, owns safe points, and tears down services. | Callback `setup` runs earlier than model `setup`; hook bootstrap must not depend on services already running. |
| [trainctl/runtime/events.py](trainctl/runtime/events.py) | Five safe points govern existing control operations. | Lifecycle notification is a separate responsibility from command scheduling. |
| [trainctl/runtime/current_step.py](trainctl/runtime/current_step.py) | Retains live batch/loss state only on the training thread. | Export callback data on that thread; never publish live tensor references to a subprocess or REST worker. |
| [trainctl/runtime/live_capture.py](trainctl/runtime/live_capture.py) | Some requests wait for a later training-thread safe point. | A blocking hook must not wait for such a request: that would deadlock its own training thread. |
| [trainctl/runtime/artifacts.py](trainctl/runtime/artifacts.py) | Owns persisted inspection artifacts. | Temporary hook tensors are a separate private resource, not automatically REST-served artifacts. |
| [examples/cifar/train.py](examples/cifar/train.py) | Already configures Loguru, currently using broad sink removal. | Reconcile ownership; examples must not remove a newly installed run sink. |
| [examples/playground/run.py](examples/playground/run.py) | Contains diagnostic `print` calls for startup/shutdown. | Move diagnostic messages to Loguru; preserve any intentionally pipeable stdout. |

Core runtime, REST, and FUSE diagnostics already use Loguru. Do not rewrite upstream Lightning or change its metric
logger API. TensorBoard, CSV, and similar training metrics are not Python diagnostic log records.

Installed Lightning source establishes that model-configured callbacks are attached before callback `setup`, and
callback `setup` precedes model `setup`. Normal callback/model `teardown` is not guaranteed on an exception path. Source
loci to recheck are `trainer/trainer.py`, `trainer/call.py`, `trainer/connectors/callback_connector.py`, and
`callbacks/callback.py` beneath each installed Lightning package.

## 3. Configuration and run directory contract

Names and additional policy defaults below are proposals for implementation, not existing API. Preserve the repository's
explicit dependency/configuration style; do not introduce a second global configuration mechanism.

| Setting | Proposed contract |
| -- | -- |
| `hooks_enabled` | Default `True` while Trainctl itself is enabled: create the directory and disabled examples. `False` skips all hook filesystem and dispatch work. |
| `hooks_source_dir: Path \| str \| None` | Optional baseline root containing `light/`, `heavy/`, and supporting files. Never a live directory override. |
| `hooks_timeout_s: float \| None` | Positive timeout, or `None` for deliberately unbounded interactive debugging. Proposed default `None`; document distributed timeout implications prominently. |
| `hooks_rank_policy` | Default `rank_zero`; optional `all` for rank-local debugging. No collectives inside the dispatcher. |
| Payload limits | Explicit finite limits for JSON size, traversal depth/items, export bytes, and output buffers. Select and document concrete defaults using P0 measurements. |
| Logging interception scope | Lightning namespaces by default; an explicit application-owned installer can additionally intercept root standard logging. |

The supplied baseline must exist, be a readable directory, and pass the safety checks before copying. Invalid explicit
configuration fails early with a descriptive error, not silent substitution with examples. Once a live directory is
initialized, resuming it does not require the original baseline still to exist.

Expose the baseline argument through the mixin and relevant example CLIs. Use an unambiguous long option such as
`--hooks-source-dir`, with an available short option checked against the existing CLI. Do not retain an undocumented
`hooks_dir` argument whose meaning alternates between source and destination.

### Layout

```text
/configuration/debug-baseline/           # optional source; not executed in place
  light/on_fit_start.sh                  # e.g. mode 0755
  heavy/on_train_batch_end.sh            # e.g. mode 0644
  helpers/notify.sh

<resolved-run-log-dir>/
  trainctl.log                          # diagnostic text, single-process/rank zero
  trainctl-hooks-state.json              # initialization identity and schema, outside copied tree
  hooks/                                # authoritative editable run-local copy
    README.md                           # generated only if absent from baseline
    light/
      on_fit_start.sh                    # baseline copy, still 0755
      on_train_batch_end.sh              # missing baseline slot: disabled example
      ...
    heavy/
      on_train_batch_end.sh              # baseline copy, still 0644
      ...
    helpers/notify.sh                    # supporting files copied, not auto-dispatched
```

Resolve the directory from `trainer.log_dir`, not from a checkpoint or inspection-artifact override. If no log directory
is available, follow Trainctl's existing XDG/runtime temporary-directory policy, but create a collision-resistant hook
session subdirectory. Current numeric `run_id` values can collide across processes; do not use them alone as filesystem
identity. Log the resolved directory once, including the fallback case. Never replace an intended remote logger URI with
an accidental local path; require a local filesystem destination or use the documented fallback.

In distributed runs, rank zero initializes a shared live directory for the default rank policy. An `all` policy needs
per-process diagnostic files, for example `trainctl-rank-1.log`, and unique event/temp identities. P0 must resolve
shared versus node-local log-directory behavior before promising all-rank support. Independent concurrent Trainers must
use different live run directories; initialization ownership must detect a conflicting independent session.

### Ownership and execution-right preservation

The user's clarified preservation requirement is **ownership and execution rights**, not an exact metadata archive.
Existing file contents must remain byte-identical. For each copied file/directory, preserve owner and group (`st_uid`,
`st_gid`) and the three execution bits (`st_mode & 0o111`). Copy ordinary read/write permission bits as well where
possible so scripts and their helpers remain usable; never accidentally grant execution. Timestamps, ACLs, flags, and
extended attributes are not acceptance requirements and need no custom preservation machinery.

`copy2` alone is insufficient: Python's copy helpers do not preserve owner/group. Read source ownership, copy the file,
apply ownership only when it differs, then restore permission bits after any ownership change and verify the result. Use
an explicit supported ownership operation such as `os.chown`; do not rely on the copying process's default ownership.
[Python shutil documentation](https://docs.python.org/3/library/shutil.html)

For a same-user/same-group baseline this normally needs no ownership change. If the training identity cannot retain a
different UID/GID, fail initialization with the path, required ownership, and permission error. Do not elevate
privileges or silently substitute the current owner. The operator can supply a baseline owned by the training identity
or arrange permitted ownership outside Trainctl. Newly generated examples belong to the training identity, have
documented normal read/write permissions, and have all execution bits cleared.

Additional proposed v1 boundary: ordinary files and directories only. Reject symlinks, devices, sockets, FIFOs, and
privilege-bearing metadata before publication, with the offending path and reason. Rejecting links is simpler and safer
than retaining links back into the baseline or silently materializing unrelated files. If real baseline use requires
links, the owner must expand this contract before implementation crosses that boundary. Reject unsupported privileged
mode bits explicitly; do not treat them as ordinary execution rights.

Copy supporting files recursively, but recognize hook subscriptions only at exact supported `light/<hook>.sh` and
`heavy/<hook>.sh` paths. Warn once about unknown hook names in those two directories; nested helpers remain support
files. Do not replace a baseline `README.md` or add unrelated configuration files.

### First initialization, repeat setup, and recovery

1. Establish run identity, logging ownership, and an exclusive initialization claim outside `hooks/`. Resolve paths and
   reject source/destination equality, containment in either direction, and destination symlinks.
2. If the state marker identifies a completed compatible initialization, reuse the live tree untouched. A different
   baseline on resume produces a warning and is ignored; intentional reinitialization requires a new run directory.
3. On a genuinely new run, stage a complete tree in a uniquely named sibling directory on the same filesystem. Copy the
   baseline if supplied. Do not copy into an existing live tree with overwrite-enabled merge semantics.
4. Fill each absent `(modality, callback)` slot with an example, created without executable bits. Do not modify,
   replace, or chmod an existing slot. A directory or unsupported object occupying a required script path is a
   configuration error.
5. Complete staging while directory permissions permit writing; restore copied directory ownership/modes after adding
   missing files. Preserve restrictive baseline permissions even if the operator must subsequently chmod the live
   directory to edit it. Newly created directories use documented owner-writable defaults.
6. Validate required files, owner/group, and execution rights; write the initialization description in staging metadata;
   publish the tree atomically and complete the external marker. Never advertise a partial tree as ready.
7. Scan the published live directory, including disabled files, and dispatch the first callback only after it is ready.

The publication/marker gap needs explicit recovery: a retained ownership record identifies a tree this initializer
published, so it can finish its marker without recopying. An existing unmarked tree of unknown origin is never
overwritten or adopted silently; stop with an actionable conflict. A failed copy leaves the source untouched and removes
only its own identified staging directory. Detect a changing source during copying where practical and reject the
inconsistent snapshot; operators must not edit the baseline during initialization.

Never reseed during discovery or ordinary resume. In particular, a deleted example stays deleted. Do not add newly
introduced callback examples automatically to an older initialized directory; report catalogue-version drift and
document an explicit future upgrade path. A restarted process may discover new live filenames, but it must not reset
live contents.

### Examples

Generate both modalities for all supported callbacks: currently 37 callbacks, hence 74 script slots. Each script
explains its callback, argument contract, available data, and synchronous/temporary-file restrictions. A light example
prints useful metadata; a heavy example reads the manifest and summarizes arrays without assuming a tensor exists at
every callback. Every generated `.sh` has all execute bits cleared. Enabling it must require an operator's explicit
`chmod +x`.

Package a small number of reusable templates plus callback metadata, not 74 independently maintained implementations.
Render the shebang into generated files. Store templates as non-executable template resources so the repository's
`check-shebang-scripts-are-executable` gate does not conflict with deliberately disabled runtime examples. Verify wheel
resource inclusion and run ShellCheck against rendered examples. An optional `curl` example must have no embedded
endpoint credentials and remain disabled; it does not introduce a Python HTTP dependency.

## 4. Callback integration and ordering

Use one backend-compatible Lightning `Callback` adapter, attached through the mixin's `configure_callbacks()`
composition. Preserve the parent's supported single-callback/list/empty return forms and all user callbacks; install the
Trainctl adapter exactly once. A subclass overriding `configure_callbacks()` must compose with `super()`, and the
documentation must state that requirement. Do not monkey-patch Lightning's private dispatch loop.

The adapter supplies explicit, correctly typed lifecycle signatures for the supported version. A tested immutable
catalogue owns hook names and parameter descriptions. A test may inspect the installed public `Callback` API to detect
coverage drift; production must not execute arbitrary methods found by unconstrained reflection.

| Group | Exact callback names |
| -- | -- |
| Setup and fit | `setup`, `teardown`, `on_fit_start`, `on_fit_end` |
| Sanity check | `on_sanity_check_start`, `on_sanity_check_end` |
| Training | `on_train_start`, `on_train_end`, `on_train_epoch_start`, `on_train_epoch_end`, `on_train_batch_start`, `on_train_batch_end` |
| Validation | `on_validation_start`, `on_validation_end`, `on_validation_epoch_start`, `on_validation_epoch_end`, `on_validation_batch_start`, `on_validation_batch_end` |
| Test | `on_test_start`, `on_test_end`, `on_test_epoch_start`, `on_test_epoch_end`, `on_test_batch_start`, `on_test_batch_end` |
| Prediction | `on_predict_start`, `on_predict_end`, `on_predict_epoch_start`, `on_predict_epoch_end`, `on_predict_batch_start`, `on_predict_batch_end` |
| Optimization | `on_before_backward`, `on_after_backward`, `on_before_optimizer_step`, `on_before_zero_grad` |
| Checkpoint and failure | `on_save_checkpoint`, `on_load_checkpoint`, `on_exception` |

`state_dict`, `load_state_dict`, and `state_key` define callback persistence rather than lifecycle notification; do not
invent shell return protocols for them. Do not add names such as `on_after_optimizer_step` that are not in this API.

Order within a dispatched callback:

1. Ensure minimal hook/logging state exists; do not start unrelated runtime services early.
2. Perform the initial or due epoch discovery scan.
3. Apply rank policy and revalidate the known light script; build its bounded metadata only if enabled; execute it.
4. Revalidate the heavy script; only if enabled, prepare its metadata and tensor snapshot, then execute it.
5. Release invocation resources and return Lightning's normal callback result.

The event identifies the actual callback position, not “after all other callbacks” or “after the module hook”. Preserve
Lightning's own callback ordering, including monitoring/checkpoint special cases. Do not emit duplicate lifecycle hooks
from the mixin's existing safe-point methods. Each emitted event is observational: ignore successful stdout as a command
channel, do not change callback arguments, and do not retain references after dispatch.

Heavy snapshots are taken immediately before the heavy invocation. Light runs first; therefore the two scripts are not
promised a single simultaneous tensor snapshot. They share the callback occurrence identity but have distinct invocation
identities and capture timestamps.

Keep logging alive through the `teardown` script and existing runtime teardown messages. On failure, run `on_exception`
at most once for Lightning's actual notification, clean up the hook session independently of normal teardown, and
preserve the original exception. If Trainctl's existing backward-exception hold delays re-raising, Lightning's
`on_exception` hook is correspondingly delayed; this plan does not change the hold policy.

## 5. Discovery and enablement state

Scan only the two live modality directories at initial bootstrap and once at the start of each training epoch, before
its `on_train_epoch_start` script. Standalone validate/test/predict calls get a fresh startup scan; if their loops
expose multiple epochs, scan once at each epoch start. Validation inside a training epoch does not trigger repeated
rescans. Record a logical epoch token so repeated callbacks do not rescan the same epoch.

The catalogue retains disabled and previously discovered paths, not only executable ones. Before preparing any payload,
check the candidate's current existence, file type, permission bits, and containment. Check again immediately before
launch, because heavy serialization can take appreciable time. Do not trust an earlier cached mode or shell contents.

| Operator change in the live tree | Observable result |
| -- | -- |
| `chmod +x` on a known disabled script | Runs at its next applicable callback, including during the same epoch. |
| `chmod -x` on a known script | Skipped at its next callback, without tensor export. Does not stop a process already running. |
| Edit/atomically replace a known script | New contents are used on the next callback after validation. |
| Delete a known script | Skipped; no automatic example replacement. |
| Recreate a previously known path | Eligible at the next callback after validation. |
| Create a previously unknown supported filename | Discovered at the next epoch/startup scan. |
| Edit the original baseline after bootstrap | No effect on the current run. |
| Create a filename while training is held | Discovery waits for the next scan; an explicit REST rescan command is deferred. |

An enabled script must have at least one POSIX execution bit and be executable/readable by the training identity.
Permission bits, not suffix alone, are the enable switch. Execute via a configured, validated Bash binary and an argv
list, never `shell=True` or `bash -c` assembled from event data. Passing a script to Bash does not enforce its execute
bits, so the dispatcher must enforce them itself. An enabled but unreadable/unlaunchable file is a logged failure, not a
silent disabled state. The initial supported platform is POSIX with Bash; a native Windows implementation requires a
separate enablement contract.

There remains an ordinary stat-to-launch race with concurrent edits. Revalidate and contain I/O failures; do not claim
this mechanism protects against an adversary able to modify trusted scripts as the same user. Reject live symlinks
rather than following a newly replaced path outside the run tree.

## 6. Invocation and payload contract

The following is a proposed version-1 interface. Options are separate argv elements; JSON is one argument, not shell
code.

```text
bash /absolute/run/hooks/light/on_train_batch_end.sh --params-json <JSON>
bash /absolute/run/hooks/heavy/on_train_batch_end.sh --params-json <JSON> --manifest <absolute-manifest-path>
```

Set `cwd` to the live `hooks/` root so scripts can find copied helpers consistently. Use closed stdin by default;
scripts must not steal input from the training terminal. Inherit the normal environment without logging it, and document
that trusted scripts consequently have access to the operator's credentials. Do not put secrets or bulk data in
command-line JSON, because argv can be observable to other local processes.

### Light metadata

Include a schema version; collision-resistant session, occurrence, and invocation IDs; per-process occurrence sequence;
hook and modality; timestamp; PID and available rank/world-size; stage; epoch/global-step; batch/dataloader indices when
supplied; and a bounded description of supported callback arguments. Unavailable early/late fields are `null`, not
guessed. Record whether metadata was truncated and why.

Primitive values may be copied. Tensor entries contain path, shape, dtype, and device descriptions only. Do not evaluate
tensor values, including scalar tensors: no `.item()`, `.tolist()`, `.cpu()`, or array conversion. Do not recursively
walk the model/trainer/optimizer, execute arbitrary properties, stringify unknown objects with arbitrary `repr`, or
enumerate all environment variables. Bound collection length, depth, and final encoded size; report unsupported/cyclic
structures. Stay below OS argv limits with headroom, rather than relying on a giant JSON command to launch successfully.

### Heavy manifest and arrays

By default export tensors actually supplied to that callback, including nested batch/output/loss/checkpoint structures.
A checkpoint can itself be large, so it remains subject to export limits. Callbacks with no tensor arguments produce a
valid empty manifest. Model parameters, gradients, or optimizer tensors need explicit selectors; automatic full-state
capture is excluded. A follow-up selector extension must not change the default payload silently.

Create a private, unique per-invocation temporary directory outside REST-served paths. The manifest contains the event
identity, schema, tensor records, primitive structure description, conversion notes, and unsupported/limit diagnostics.
Assign generated filenames such as `tensor-000001.npy`; never derive filesystem paths from user dictionary keys. Record
nested argument paths separately and preserve enough structure to locate values unambiguously.

Detach and serialize on the training thread, moving tensors to CPU only for an enabled heavy invocation. Ordinary
supported dense NumPy dtypes retain shape and dtype; explicitly record any conversion, with `bfloat16` to `float32` as
the proposed v1 exception. Check the predicted converted byte size before allocating a host copy. Reject unsupported
sparse, quantized, meta, sharded, or custom tensor layouts with a clear failed-capture result instead of implicit
densification or collectives. Treat tensor-conversion validation outcomes as data, not broad exception-driven control
flow.

Use numeric `.npy` arrays with pickling disabled on both save and load. Do not pickle Python objects, models, or
arbitrary callback arguments. NumPy documents why object-array pickling is unsafe for untrusted input.
[NumPy save documentation](https://numpy.org/doc/stable/reference/generated/numpy.save.html)

Array conversion can share CPU tensor storage. Complete all file writes before starting the script and before allowing
training to resume; do not mistake a detached view for an owned immutable snapshot. The completed files are the snapshot
boundary. [PyTorch Tensor.numpy documentation](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.numpy.html)

Use all-or-failed heavy capture in v1: if a requested tensor cannot be represented or a limit is exceeded, log the
precise failure and skip that heavy script. Do not silently pass a partial success manifest. Include how to reduce the
capture. Serialization failures do not suppress the independently executed light event or abort training by default.

Keep files available for the entire supervised subprocess lifetime. Close handles and clean the private directory after
success, nonzero exit, timeout, launch failure, or serialization failure. Use structured resource ownership and
`finally` for cleanup, without suppressing interruption/system exceptions. A script needing persistent output must copy
it to its own destination before returning. Detached descendants and a hard-killed parent cannot receive guaranteed
cleanup; document that limit and never perform broad deletion of other runs' temporary files.

## 7. Subprocess supervision and failure policy

Training waits for each script. Concurrent stdout/stderr draining is allowed as an implementation detail; it is not
asynchronous hook dispatch. Drain both pipes continuously with bounded in-memory buffers, including long lines and
invalid UTF-8, to prevent pipe deadlock or unbounded RAM use.

Stream tagged stdout/stderr chunks through the run's Loguru sink while retaining bounded tails for the final summary.
Preserve output identity and ordering within each stream; do not promise exact interleaving across two OS pipes. No
silent discard: mark decoding substitutions, chunking, or configured output limits. File-sink failure must be reported
through a still-working stderr path without recursive logging. Persisted verbose output can consume disk; document
retention and rotation policy and never promise infinite retention.

For each failure, emit an ERROR record with:

- Hook, modality, absolute live script path, occurrence/invocation/session identity, rank, epoch, and global step.
- Failure phase: validation, capture, launch, execution, timeout, or cleanup.
- Exit code or signal where available, exception type/message for I/O failures, elapsed duration, and timeout setting.
- stdout/stderr tails with clear truncation markers, plus where earlier output was written.

Use `check=False` and inspect exit status as data. Catch only relevant I/O failures explicitly authorized for
resilience. Never broadly catch `BaseException` to turn a training failure into success. Cleanup uses `finally`;
interruption is re-raised after child containment. A failed hook does not auto-disable itself, retry itself, or prevent
the other modality from running. Missing/disabled known scripts are ordinary skips; an actual launch race is a
diagnostic failure.

On POSIX, supervise a separate process group. On timeout or interruption, terminate the group, allow bounded shutdown,
escalate if necessary, and reap it before deleting tensor files. A script deliberately daemonizing outside that group is
outside the guarantee. Test cleanup failure separately: report leaked invocation path without masking the original
error.

Initialization policy differs from invocation policy: invalid explicit configuration or an ambiguous live destination
fails before training starts. Recoverable run-time hook I/O failures log and continue. If the diagnostic file cannot be
opened during enabled-hook initialization, fail with a stderr explanation rather than claiming hook failures are being
persisted when they are not.

Blocking hooks must not request an action that requires the blocked training thread to advance, such as a fresh live
capture or a queued checkpoint followed by waiting for completion. Existing snapshot-only REST reads and exported files
are appropriate. Long hooks may also exhaust distributed process-group watchdogs; the default rank-zero policy does not
remove that risk. Set finite timeouts where needed and test the chosen distributed setup.

## 8. Standard logging into Loguru

**Yes: standard Python logging can be redirected into Loguru.** Use a `logging.Handler` that forwards a resolved
message, level, original source metadata, and `exc_info` into a bound Loguru logger. Loguru documents this interception
pattern.
[Loguru standard logging interoperability](https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging)

The following ownership design is Trainctl policy, not a claim that copying the documentation's root configuration is
sufficient:

1. Introduce an explicit logging-session owner. It holds handler references, original logger settings, run context, and
   IDs of its Loguru sinks. Nothing installs handlers at module import, and no module-level mutable registry owns
   sessions.
2. Default library integration intercepts the selected Lightning namespace and relevant Fabric namespace. Installed
   Lightning adds a stream handler and sets `propagate=False` when root is unconfigured, so root-only interception
   misses those messages. Inspect and test descendants with their own handlers rather than assuming propagation.
3. Temporarily detach the handlers in the explicitly selected namespace scope, retain them for restoration, and install
   exactly one route per record. Avoid both direct interception and root propagation of the same record. Preserve caller
   root handlers and unrelated namespaces. Custom handlers in intercepted namespaces are temporarily replaced by design;
   expose opt-out and make this behavior visible in the API documentation.
4. Offer application-owned root interception for users wanting all standard logging unified. It must explicitly take
   ownership of existing root handlers, preserve them for restoration, and still handle Lightning's non-propagating
   namespaces. Do not use unconditional `basicConfig(force=True)` or `logger.remove()` inside the library.
5. Preserve original record name, pathname, line/function, time, level number/name, and exception information, even if
   some need structured Loguru extras rather than reconstructed call depth. Respect the chosen level policy and test
   custom numeric levels without exception-based lookup logic. Format source attribution using those original fields.
6. Bind hook errors and Trainctl diagnostics to the same session/rank context. Add an owned text sink in the resolved
   run directory, with an explicit filter preventing unrelated sessions from leaking into the file. Keep console
   diagnostics on stderr and never redirect metrics, progress-bar rendering, or arbitrary stdout by pretending they are
   log records.
7. Make the bridge one-way. A Loguru sink that sends records back into intercepted standard loggers creates recursion;
   reject/document that incompatible setup and test a reentrancy guard. Disable diagnostic local-variable dumping for
   exception formatting so tracebacks do not unexpectedly expose tensors or secrets.
8. Install console interception before hook bootstrap; attach the file sink as soon as the resolved directory exists.
   Messages emitted before installation cannot be recovered. Applications needing Trainer-construction logs must install
   the application bridge before constructing the Trainer.
9. Keep ownership through the final hook and runtime cleanup. Flush and remove only owned sinks/handlers, restore only
   settings still owned by the session, and do not erase handlers another library added meanwhile. Exercise normal
   return, early setup failure, exception, repeated Trainer use, and distributed child-process startup.

Standard logger configuration is process-global even when its owner is explicit. Proposed supported v1 operating model:
one active training/logging session per process, with sequential reuse supported and rank processes independently owned.
Detect and reject overlapping installations through the existing owned-handler identity, not an invisible mutable global
registry. Supporting independent concurrent Trainers in one process requires explicit context propagation and a separate
design; do not promise reliable attribution merely by adding a run ID to whichever handler was installed last.

Migrate remaining Trainctl/example diagnostic `print` or standard logger calls only where present; preserve pipeable
data. The existing CIFAR example's logging setup must cooperate with the session owner. Do not patch Lightning sources
and do not replace TensorBoard/CSV/W&B metric loggers with Loguru.

## 9. Responsibility boundaries

Suggested files are implementation candidates, not instructions to create one module per helper:

| Owner | Responsibility and stable boundary |
| -- | -- |
| Hook session (`trainctl/hooks/manager.py`) | Own bootstrap, live catalogue, discovery cadence, rank policy, and dispatch orchestration. Private copy/discovery helpers stay here unless they acquire independent ownership. |
| Payload exporter (`trainctl/hooks/payload.py`) | Own bounded event schema, tensor selection, serialization, manifest, and invocation-scoped file lifetime. No Lightning scheduling or process launch. |
| Shell runner (`trainctl/hooks/runner.py`) | Own argv-based launch, pipe draining, timeout/process-group supervision, and structured execution result. Receives logger/context explicitly. |
| Lightning adapter (`trainctl/hooks/lightning.py`) | Own backend-compatible callback signatures, catalogue binding, and lifecycle ordering. Delegates policy to the session. |
| Logging session (`trainctl/logging.py`) | Own standard-record bridge, explicit handler/sink ownership, run-file routing, and restoration. Does not depend on hook internals. |
| Template resources | Own disabled example content and callback help, packaged and tested as rendered output. |

Dependencies point from adapter/runtime composition to hook session, then to exporter and runner. Logging is shared
infrastructure with explicit ownership, not a backdoor into the runtime. Use structured result types for skip/failure
states, frozen configuration, Python 3.12+ types, and no mutable module globals. Add NumPy as a direct runtime
dependency with `uv add numpy` when implementation starts; it is currently only transitive. No native HTTP dependency is
needed.

## 10. Security and operational acceptance

Boundary: shell scripts are trusted operator code running with the training user's permissions, potentially with network
access and credentials. The baseline source and live directory are executable configuration, not untrusted uploads. The
training operator is the residual-risk owner; the implementing engineer owns the controls and evidence before release.

| Asset / threat | Required trusted-side control | Adversarial verification | Residual risk / response owner |
| -- | -- | -- | -- |
| Training identity: accidental or unauthorized execution | Disabled generated files, master opt-out, strict supported paths, executable-bit revalidation, explicit trusted baseline. Validate ownership/modes and reject world-writable execution roots. | Disabled files never launch; refuse unsafe roots, symlink substitution, and unknown callback paths. | Operator must trust users/groups allowed to edit scripts, including any additional ACL grants. No same-user sandbox. |
| Command boundary: payload becomes shell code | Fixed interpreter plus argv array; JSON passed as one argument; no event interpolation into `-c`. | Quotes, spaces, newlines, dollar expansions, leading dashes, and malicious-looking metadata remain inert data. | Script authors must quote their own arguments; operator owns script behavior. |
| Source/live tree integrity: overwrite or path escape | Validate canonical source/destination relationship; reject links/special files; exclusive bootstrap, staged publication, no overwrite on resume. | Source equals/is inside destination; destination link; concurrent initialization; interrupted publish; readonly files; malicious path components. | Privileged concurrent filesystem mutation is outside v1; stop on detected conflicts. |
| Tensor confidentiality: retained or exposed training data | Minimize exports, private temp directory/files, generated filenames, no pickle, no automatic artifact registration, bounded metadata. | Manifest traversal keys; object arrays; permission checks; cleanup after every handled failure; REST access denied to raw temp files. | Trusted script can copy/exfiltrate data; hard kill may leave private files. Operator owns cleanup and log retention. |
| Training availability: hung script/output flood | Optional timeout, process-group supervision, bounded RAM, continuous pipe draining, explicit output/retention policy. | Infinite loop, ignored termination, full pipes, huge line, invalid bytes, disk-full and cleanup errors. | Unbounded timeout is deliberate debugging risk; detached children/distributed watchdogs remain operator concerns. |
| Diagnostics: missing, duplicated, recursive, or cross-run logs | One-way bridge, owned handlers, filtered run sinks, original record metadata, fail-visible sink initialization. | Root configured/unconfigured; custom handler/level; nested failure; repeated sessions; separate ranks; recursive sink. | Arbitrary simultaneous Trainers per process not supported; fail explicitly rather than misattribute. |
| Supply of examples: code missing or accidentally enabled | Packaged template resources, disabled rendered modes, build verification, no embedded credentials. | Install wheel in clean environment; inspect all generated modes; ShellCheck and execute representative examples. | Template/dependency updates require catalogue and security review by maintainer. |

Do not claim `stat` mode checks validate ACL-based trust. Deployment access policy is the operator's responsibility; ACL
copying/auditing is not part of the requested preservation feature. These are planned controls and tests, not
demonstrated security results.

## 11. Phase A: risk-first discovery before production implementation

P0 is executable only after implementation/probe work is authorized. Its goal is to retire architecture-changing
unknowns before building the adapter and ownership machinery. CPU-only tiny models are sufficient for most probes; use
one worker for multi-process integration and GPU-memory-sensitive checks.

| Probe | Prediction / discriminating check | Strongest alternative and response if falsified |
| -- | -- | -- |
| P0a: attachment and ordering | A model-configured backend-specific callback sees `setup`, all exercised stage hooks, and normal teardown exactly once without changing existing safe points. Trace both namespaces and preserve user callbacks/return values. | If callback composition cannot preserve early coverage, require explicit Trainer callback installation or revise only the attachment seam; do not monkey-patch Lightning internals. |
| P0b: exception and reuse lifecycle | A session can clean up after setup/training failures and then support a later Trainer/stage without leaked sinks, handlers, subprocesses, or tensors. Exercise existing backward hold and checkpoint-load failures. | If model teardown is insufficient, move ownership to a wider explicit session/finalizer boundary. Do not add blanket exception suppression. |
| P0c: log-directory/rank identity | The intended live directory is stable before the first dispatched callback, and distributed workers agree on shared versus local ownership. Test logger absent, both namespace APIs, two ranks, two independent runs, and restart. | If early log resolution or all-rank initialization requires collectives at unsafe points, resolve through the surrounding lifecycle first or defer `all`; never block inside a script on a collective. |
| P0d: logging interception | One standard record yields one intended Loguru record/file entry with correct exception/source data, then original logger behavior returns after cleanup. | If application handler ownership is ambiguous, require explicit logging-session installation for that scope; do not clear host handlers silently. |
| P0e: copy and resource limits | Temporary staging preserves owner/group and execution bits, fills readonly baseline trees correctly, recovers publish interruption, and rejects unsafe layouts. Representative exports establish feasible finite payload/buffer limits. | If ownership cannot be preserved by the training identity or export limits cannot bound allocations, fail validation or resolve the environment/representation with the operator. |

Stop each probe once it produces a repeatable trace/test and either supports the prediction or identifies the smallest
affected redesign. A failed probe can complete discovery but cannot unlock the dependent implementation. Record outcome,
owner, required environment, and next check in this document; preserve unaffected work. Do not repeatedly retry an
unchanged failing hypothesis.

Conceptual algorithm to validate before fixing public interfaces:

```text
on_actual_callback(name, trainer, module, arguments):
    if Trainctl or hooks disabled: return
    ensure_owned_logging_and_initialized_run_copy()
    scan_if_initial_or_new_epoch()
    if rank_not_selected: return
    occurrence = bounded_callback_identity()
    for modality in [light, heavy]:
        candidate = known_path(name, modality)
        if candidate_absent_or_disabled_now: continue
        validate_candidate_and_build_bounded_payload()
        with owned_optional_tensor_files:
            revalidate_candidate()
            execute_blocking_and_record_result()
    if terminal_lifecycle: finalize_owned_resources_at_verified_boundary()
```

The pseudocode omits error branches only for readability. Sections 3, 7, and 8 govern initialization failure, invocation
failure, exception preservation, and delayed final logging cleanup; no exception path may bypass those contracts.

## 12. Phase B: dependency-valid implementation increments

All increments below are provisional until their P0 prerequisites pass and implementation is authorized. Owner: the
implementing engineer/agent. The training operator resolves deployment trust and any requested expansion of the
defaults. Every increment ends with focused behavioral tests and a reviewable atomic commit; do not bundle unrelated
existing work.

### I0 — Owned logging and configuration foundation

- Entry: P0b/P0d support the selected ownership boundary.
- Outcome: a standard Lightning message and a Trainctl diagnostic appear in the same run text log, with no duplicate or
  lost exception, and logging state is restored afterward.
- Build: explicit logging-session owner, namespace bridge, optional application-root installer, validated proposed
  settings, source-path normalization, and safe run sink setup. Keep hook installation independent of unrelated
  services.
- Verify: T6, invalid setting cases, early sink failure, parent handler restoration, custom levels, repeated sessions.
- Stop: ownership/lifecycle tests pass; no global handler clearing or mutable singleton is introduced.

### I1 — Baseline bootstrap and disabled templates

- Entry: P0c/P0e resolve filesystem/identity behavior; I0 can report bootstrap errors.
- Outcome: the log tree faithfully copies a baseline and fills only absent modality/callback slots with disabled
  examples.
- Build: staged initializer, provenance/ownership marker, template renderer/resources, owner/group and mode checks,
  resume/conflict rules.
- Verify: T1, interrupted publication, concurrent initializer, owner/group and modes including readonly directories,
  missing/inaccessible source, source/live containment, and source-unchanged checks. Confirm package resources survive
  wheel installation.
- Stop: repeat initialization changes no live content/modes and never restores deleted scripts; no partial tree
  executes.

### I2 — Discovery, light events, and synchronous execution

- Entry: I0/I1 complete; P0a fixes event identity/cadence.
- Outcome: editing/chmod of a known script affects the next event; newly named scripts appear at the next scan; failures
  contain useful output and do not stop training.
- Build: retained catalogue, epoch token, bounded primitive metadata, argv contract, runner supervision, output
  draining, structured execution results, and skip/error distinction. Supply process/clock/log dependencies explicitly
  for testing.
- Verify: T2/T5 plus metacharacter arguments, execution races, long/binary output, process groups, timeout, and
  interruption.
- Stop: disabled/absent scripts allocate neither tensor files nor tensor host copies; handled failures leave no child.

### I3 — Heavy snapshots and temporary resource ownership

- Entry: I2 stable; P0e establishes initial size/representation limits.
- Outcome: an enabled script can read an independent NumPy snapshot throughout execution; files disappear afterward.
- Build: explicit callback-argument traversal, direct NumPy dependency, manifest schema, dtype rules, preallocation
  limits, and resource-scoped cleanup. Do not add full-state selectors unless separately accepted.
- Verify: T3, CPU and available CUDA tensors, conversion aliases, unsupported layouts/dtypes, nested/cyclic inputs,
  allocation limits, serialization errors, nonzero script exit, and failed cleanup without lost primary error.
- Stop: no unsafe pickle, silent partial captures, disabled-path tensor work, or live references retained after
  dispatch.

### I4 — Complete Lightning lifecycle integration

- Entry: P0a/P0b/P0c pass; I0–I3 establish tested services.
- Outcome: all supported adapter methods obey their contracts, and real fit/validate/test/predict/checkpoint/exception
  runs dispatch at the actual Lightning callback points in both namespaces.
- Build: backend adapter and immutable catalogue, mixin callback composition, stage/epoch scanning, selected-rank
  behavior, and exception/teardown integration without duplicate safe-point notifications.
- Verify: T4/T7, gradient accumulation and manual optimization, callbacks absent from a given training configuration,
  zero-batch stages, checkpoint restore, subclass super composition, and existing inspection/hold regressions.
- Stop: public callback coverage matches the supported API; no fabricated events, changed callback returns, or lost
  cleanup.

### I5 — Examples, operator documentation, and release checks

- Entry: I0–I4 pass; accepted rank/platform scope is documented.
- Outcome: an operator can start a run, find its live scripts, enable one, inspect tensor files during execution, see a
  deliberate failure in the training log, and disable it without restarting.
- Build: relevant example CLI forwarding, diagnostic logging migration, template help, source-versus-live walkthrough,
  `curl` illustration, payload/manifest documentation, timeouts, security limits, and recovery instructions.
- Verify: the walkthrough against an installed wheel; T1–T7; configured type/lint/security checks; relevant Python tests
  through `uv run pytest` with coverage and resource-appropriate worker count; finally `pre-commit run --all`.
- Stop: all applicable checks pass and baseline regressions are distinguished from this change. Do not bypass pre-commit
  or fix unrelated user work to manufacture a clean gate. Hand over test evidence and any explicitly deferred
  capability.

I1 and the isolated runner portion of I2 can proceed independently after their prerequisites. Run resource-heavy Trainer
and distributed tests serially; parallelize small pure metadata/copy tests only when independent. No absolute effort
estimate is justified before P0 resolves the integration risks.

## 13. Acceptance test matrix

Tests must observe behavior and resource boundaries, not incidental helper calls or source text. Callback API catalogue
inspection is a compatibility check, not a substitute for integration tests.

| Suite | Required falsifying cases and evidence |
| -- | -- |
| T1 — Bootstrap | No source creates 74 disabled scripts for this catalogue. Partial source fills only missing slots; original bytes, UID/GID, execution bits, and helper files survive. Ownership-changing failure is explicit; verify modes after ownership operations. Source is unchanged. Live edits/deletions survive repeated setup/resume. Baseline edits do not propagate. Reject wrong path type, unsafe links, recursive source/destination, conflicting session ownership; recover interrupted initialization without overwrite. |
| T2 — Light/discovery | Known chmod/edit/delete/recreate changes take effect as specified. New filenames wait for epoch scan; initial and standalone-stage discovery work. Disabled files never launch. Light events expose no tensor values or host-copy side effects, including scalar tensors and nested checkpoint arguments. Metadata is valid, bounded, and unambiguous. |
| T3 — Heavy/data lifetime | Child observes valid arrays while parent waits; exact values/shapes/dtypes or documented conversions; empty manifest on no-tensor callbacks. Snapshot remains independent of later training mutation. Files persist through execution then disappear after success/error/timeout/launch or serialization failure. Reject pickle/object arrays, unsafe path keys, limits, and unsupported tensor representations. |
| T4 — Lightning contract | Both namespaces: fit with sanity check, validate, test, predict, checkpoint save/load, exception, setup failure, teardown, and sequential reuse. Preserve user callbacks, return values, ordering, native metrics, and existing Trainctl safe points. Catalogue covers all 37 current lifecycle signatures; configuration-dependent absent hooks are not fabricated. |
| T5 — Runner/failure | Success, nonzero exit with both streams, signal, timeout, launch race, unreadable enabled script, disk-full logging, child ignoring termination, large/binary output, interruption, and cleanup failure. Assert ERROR context/tails, no pipe deadlock, bounded buffers, contained child group, and original training error preserved. |
| T6 — Logging | Root configured and unconfigured; Lightning-installed and custom namespace handlers; custom numeric levels; original source/exception data; no duplicate routing; one-way/reentrancy protection; stderr vs stdout; owned cleanup; startup failure; repeated sessions; separate rank files; overlapping-session rejection. Metrics remain on native loggers. |
| T7 — End-to-end/regression | Master-off produces no hook work. Default examples never execute. Default rank-zero does not duplicate notifications; optional all-rank mode is tested only if P0 unlocks it. Two independent processes cannot collide by numeric run ID. Existing REST/FUSE, inspection artifacts, live-capture scheduling, and backward-exception hold retain behavior. Installed-wheel walkthrough and repository gates pass. |

Performance evidence is behavioral first: zero tensor conversion/temp allocation with no enabled heavy handler, bounded
metadata/output memory, no directory walks at batch frequency, and no retention of completed callback tensors. Record
dispatch overhead with all scripts disabled and representative heavy export costs as measurements, not universal
promises.

## 14. Rollout, recovery, and handover gate

Roll out with the facility enabled but all generated examples disabled. A provided executable baseline is explicit code
activation, so log the active directory and enabled subscriptions at startup. Document that baseline changes require a
new run or editing its live copy, not a hidden reload. `chmod -x` disables the next invocation; `hooks_enabled=False` is
the next-start master opt-out. Neither undoes a completed script's external side effects.

On failure, preserve run logs and the live hook tree for diagnosis. Remove only owned temporary/staging resources. Never
recursively delete the run log root or overwrite the source as a recovery action. A partial initializer names its
staging path and resume check; an unknown existing destination requires the operator to select a fresh run directory or
resolve ownership. No automatic retry of side-effecting scripts.

Handover review boundary: ready for the risk-first probes after authorization, **not release approval**.

| Gate | Evidence in this plan | Verdict and limits |
| -- | -- | -- |
| Scope/frame | Sections 1 and 3 incorporate the latest baseline-copy rule; section 1 defers HTTP and asynchronous work. | Ready for planning handover; API names/defaults beyond user requirements remain explicit proposals. |
| Plan quality | R1–R10 map to I0–I5/P0 and T1–T7; sections 11–12 order prerequisites, tests, owners, and stop conditions. | Discovery plan actionable; production increments conditional on P0 and implementation authorization. |
| Architecture | Sections 2, 4, 8, and 9 assign callback, runtime, file, process, and logging ownership. | Coherent candidate design; early lifecycle/rank/logging integration still requires executable evidence. |
| Security | Sections 3, 6, 7, and 10 trace execution/data threats to boundary controls and adversarial checks. | Design controls specified; effectiveness unverified. Deployment trust and any expanded metadata/link policy require operator ownership. |
| Communication/reasoning | Accepted requirements, proposed policies, observed source facts, alternatives/falsifiers, and unrun checks are distinguished. | Standalone handover; do not reinterpret planned tests as completed tests or infer implementation permission. |

Use the structured-thinking plan, architecture, security, reasoning, and communication rubrics when reviewing the
eventual implementation handoff. Keep their verdicts separate; an aggregate score cannot waive missing behavioral
evidence or a security floor. This document's qualitative review is not a numerical score of an unimplemented feature.

Next authorized action after plan review: approve implementation/probes, then execute P0 and update only the decisions
its evidence changes. Keep the existing working tree intact; the repository contains substantial user-owned staged and
unstaged work unrelated to this handover.
