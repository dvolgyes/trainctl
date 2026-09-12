"use strict";

const JSON_HEADERS = { "Content-Type": "application/json" };

// ---- generic helpers -------------------------------------------------------

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${await res.text()}`);
  return res.json();
}

async function postJSON(url, body) {
  return fetchJSON(url, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body ?? {}),
  });
}

/** Reads a virtual file via GET /files/<path>; JSON-parses when the response says so. */
async function fetchFile(path) {
  const res = await fetch(`/files${path}`);
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  const contentType = res.headers.get("content-type") || "";
  return contentType.includes("application/json") ? res.json() : res.text();
}

async function submitCommand(kind, args) {
  const command = await postJSON("/commands", { kind, args: args ?? {} });
  return pollCommand(command.id);
}

async function pollCommand(id, timeoutMs = 60000, intervalMs = 300) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const command = await fetchJSON(`/commands/${id}`);
    if (command.status !== "queued" && command.status !== "executing")
      return command;
    if (Date.now() > deadline)
      throw new Error(`command ${id} did not finish within ${timeoutMs}ms`);
    await sleep(intervalMs);
  }
}

/** Polls GET /artifacts until a debug entry of `kind` appears that isn't in `excludeIds`. */
async function pollForArtifact(
  kind,
  excludeIds,
  timeoutMs = 120000,
  intervalMs = 500,
) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const { debug } = await fetchJSON("/artifacts");
    for (const id of debug) {
      if (excludeIds.has(id)) continue;
      excludeIds.add(id);
      const manifest = await fetchJSON(`/artifacts/${id}`);
      if (manifest.kind === kind) return manifest;
    }
    if (Date.now() > deadline)
      throw new Error(`no ${kind} capture appeared within ${timeoutMs}ms`);
    await sleep(intervalMs);
  }
}

// ---- shared poll scheduler --------------------------------------------------
//
// Every panel's periodic refresh runs off one shared, user-configurable
// interval (see #poll-interval-input) instead of its own setInterval, so
// "how chatty is this page" is one setting rather than six hardcoded ones.

const DEFAULT_POLL_SECONDS = 15;
const MAX_POLL_SECONDS = 120;

const _pollFns = [];
let _pollTimerId = null;

/** Registers `fn` to run on every poll tick and via the manual refresh button. */
function registerPoll(fn) {
  _pollFns.push(fn);
}

function runPollsNow() {
  for (const fn of _pollFns) fn().catch((err) => console.error(err));
}

/** (Re)configures the shared poll timer. `seconds` of 0 disables auto-polling --
 * only the manual refresh button (or the initial load) then updates the page. */
function setPollInterval(seconds) {
  if (_pollTimerId !== null) {
    clearInterval(_pollTimerId);
    _pollTimerId = null;
  }
  if (seconds > 0) {
    _pollTimerId = setInterval(runPollsNow, seconds * 1000);
  }
}

function setupPollingControls() {
  const input = byId("poll-interval-input");
  input.value = DEFAULT_POLL_SECONDS;
  const applyFromInput = () => {
    const seconds = Math.min(
      MAX_POLL_SECONDS,
      Math.max(0, Math.round(Number(input.value) || 0)),
    );
    input.value = seconds;
    setPollInterval(seconds);
  };
  input.addEventListener("change", applyFromInput);
  byId("refresh-now-btn").addEventListener("click", runPollsNow);
  setPollInterval(DEFAULT_POLL_SECONDS);
}

function el(tag, props, children) {
  const node = document.createElement(tag);
  Object.assign(node, props ?? {});
  for (const child of children ?? []) node.append(child);
  return node;
}

function setText(id, text) {
  const node = document.getElementById(id);
  if (node) node.textContent = text;
}

function byId(id) {
  return document.getElementById(id);
}

// ---- run / status panel ----------------------------------------------------

/** Renders "<hostname> · <artifacts_path's parent dir>/", with the parent
 * dir's own last segment (e.g. "version_3") highlighted -- that segment is
 * what actually distinguishes concurrent runs sharing a host and log root. */
function renderRunLocation(hostname, artifactsPath) {
  const node = byId("run-location");
  if (!node) return;
  node.replaceChildren();
  if (!hostname) return;
  if (!artifactsPath) {
    node.append(`${hostname} · (no artifact dir)`);
    return;
  }
  const parentDir = artifactsPath.replace(/\/[^/]+\/?$/, "");
  const segments = parentDir.split("/").filter(Boolean);
  const tail = segments.pop() ?? "";
  const prefix = segments.length ? `/${segments.join("/")}/` : "/";
  node.append(`${hostname} · ${prefix}`);
  node.append(el("span", { className: "path-tail", textContent: tail }));
  node.append("/");
}

async function refreshRunStatus() {
  const run = await fetchJSON("/run");
  setText("status-badge", `${run.status} · ${run.phase}`);
  byId("status-badge").className = `badge status-${run.status}`;
  setText("run-epoch", run.epoch);
  setText("run-step", run.global_step);
  setText("run-batch-idx", run.batch_idx ?? "-");
  setText("hold-state", run.hold_state);
  setText("pending-commands", run.pending_commands);
  renderRunLocation(run.ranks?.[0]?.hostname, run.artifacts_path);
  byId("resume-btn").hidden = run.hold_state !== "held";
  byId("pause-btn").hidden = run.hold_state === "held";
}

function setupRunControls() {
  byId("pause-btn").addEventListener("click", () =>
    runAction("/control/pause", "hold requested"),
  );
  byId("resume-btn").addEventListener("click", () =>
    runAction("/control/continue", "resumed"),
  );
  byId("checkpoint-btn").addEventListener("click", () =>
    runAction(
      () => postJSON("/checkpoints", { label: "manual" }),
      "checkpoint requested",
    ),
  );
  byId("snapshot-btn").addEventListener("click", () =>
    runAction(
      () => postJSON("/snapshots", { kind: "runtime" }),
      "snapshot requested",
    ),
  );
  byId("stop-btn").addEventListener("click", () => {
    if (!confirm("Stop training? This cannot be undone.")) return;
    runAction("/control/stop", "stop requested");
  });
}

async function runAction(action, message) {
  try {
    if (typeof action === "string") await postJSON(action);
    else await action();
    setText("action-status", message);
  } catch (err) {
    setText("action-status", `error: ${err.message}`);
  }
  await refreshRunStatus();
}

// ---- model overview ---------------------------------------------------------

async function loadOverview() {
  const [cls, hparams, summary, summaryJson, availability, tunable] =
    await Promise.all([
      fetchFile("/model/class"),
      fetchFile("/model/hparams.json"),
      fetchFile("/model/summary.txt"),
      fetchFile("/model/summary.json"),
      fetchFile("/model/inspectors.json"),
      fetchFile("/model/tunable-hparams.json"),
    ]);
  setText("model-class", cls.trim() || "(unknown)");
  setText("model-summary", summary);
  setText(
    "model-param-counts",
    `${summaryJson.total_params ?? "?"} total, ${summaryJson.trainable_params ?? "?"} trainable`,
  );
  renderReadonlyHparams(hparams, tunable);
  renderInspectorAvailability(availability);
  buildInspectorButtons(availability);

  const surgeryEnabled =
    (await fetchFile("/meta/optimizer-surgery-enabled")).trim() === "true";
  byId("reset-momentum-btn").hidden = !surgeryEnabled;
}

// Every hparam NOT registered via `trainctl_tunable_hparams` (e.g. network depth)
// stays here, marked read-only -- this is the whole "structural vs. tunable" split.
function renderReadonlyHparams(hparams, tunable) {
  const container = byId("readonly-hparams-list");
  container.replaceChildren();
  const tunableNames = new Set(Object.keys(tunable ?? {}));
  const names = Object.keys(hparams).filter((name) => !tunableNames.has(name));
  if (names.length === 0) {
    container.append(
      el("p", { className: "empty", textContent: "none captured" }),
    );
    return;
  }
  for (const name of names) {
    container.append(
      el("div", { className: "kv-row" }, [
        el("span", { className: "kv-name", textContent: name }),
        el("span", {
          className: "kv-value",
          textContent: JSON.stringify(hparams[name]),
        }),
        el("span", { className: "badge badge-off", textContent: "structural" }),
      ]),
    );
  }
}

function renderInspectorAvailability(availability) {
  const container = byId("inspector-availability");
  container.replaceChildren();
  for (const [name, info] of Object.entries(availability)) {
    container.append(
      el("span", {
        className: `badge ${info.available ? "badge-ok" : "badge-off"}`,
        textContent: name,
      }),
    );
  }
}

// ---- controls: learning rate / momentum / finite-check / tunable hparams ---

function setupControls() {
  byId("lr-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const value = Number(byId("lr-input").value);
    setText("lr-status", "setting…");
    try {
      const result = await submitCommand("set_learning_rate", { value });
      setText(
        "lr-status",
        result.status === "succeeded" ? "updated" : `failed: ${result.error}`,
      );
    } catch (err) {
      setText("lr-status", `error: ${err.message}`);
    }
  });

  byId("reset-momentum-btn").addEventListener("click", async () => {
    const result = await submitCommand("reset_momentum", {});
    if (result.status !== "succeeded")
      alert(`reset_momentum failed: ${result.error}`);
  });

  byId("finite-check-toggle").addEventListener("change", (event) => {
    submitCommand("set_dataloader_finite_check", {
      enabled: event.target.checked,
    });
  });

  byId("tunable-hparams-set-all-btn").addEventListener("click", async () => {
    const rows = byId("tunable-hparams-list").querySelectorAll(
      "[data-hparam-name]",
    );
    const overallStatus = byId("tunable-hparams-set-all-status");
    overallStatus.textContent = "setting…";
    const results = await Promise.all(
      Array.from(rows).map((row) => setTunableHparamFromRow(row)),
    );
    const succeeded = results.filter(Boolean).length;
    overallStatus.textContent = results.length
      ? `${succeeded}/${results.length} updated`
      : "";
  });
}

/** Submits one tunable-hparam row's current input value; used by the shared
 * "Set all" button so every field can be edited before any of them is sent. */
async function setTunableHparamFromRow(row) {
  const name = row.dataset.hparamName;
  const input = row.querySelector("input, textarea");
  const status = row.querySelector(".status-line");
  let value;
  try {
    value = readInputValue(input, row.dataset.hparamType);
  } catch (err) {
    status.textContent = `invalid value: ${err.message}`;
    return false;
  }
  status.textContent = "setting…";
  try {
    const result = await submitCommand("set_hparam", { name, value });
    if (result.status === "succeeded") {
      status.textContent = "updated";
      delete input.dataset.dirty;
      return true;
    }
    status.textContent = `failed: ${result.error}`;
    return false;
  } catch (err) {
    status.textContent = `error: ${err.message}`;
    return false;
  }
}

async function refreshControlsState() {
  const [tunable, finiteCheck, hparams] = await Promise.all([
    fetchFile("/model/tunable-hparams.json"),
    fetchFile("/state/dataloader/finite-check.json"),
    fetchFile("/model/hparams.json"),
  ]);
  renderTunableHparams(tunable);
  renderReadonlyHparams(hparams, tunable);
  byId("finite-check-toggle").checked = finiteCheck.enabled;
}

// A row's input is rebuilt from scratch on every poll (the container is fully
// replaced), so an in-progress edit not yet sent via "Set all" would otherwise
// be silently overwritten by the next poll's server value -- `dirty` (set on
// the `input`/`change` events) marks a row whose edit must survive rebuilding.
function renderTunableHparams(tunable) {
  const container = byId("tunable-hparams-list");
  const pending = new Map();
  for (const row of container.querySelectorAll("[data-hparam-name]")) {
    const input = row.querySelector("input, textarea");
    if (input?.dataset.dirty === "1") {
      pending.set(row.dataset.hparamName, {
        value: input.value,
        checked: input.checked,
      });
    }
  }
  container.replaceChildren();
  const names = Object.keys(tunable);
  if (names.length === 0) {
    container.append(
      el("p", {
        className: "empty",
        textContent: "No live-tunable hyperparameters registered.",
      }),
    );
    return;
  }
  for (const name of names)
    container.append(
      renderTunableHparamRow(name, tunable[name], pending.get(name)),
    );
}

function renderTunableHparamRow(name, info, pending) {
  const input = buildInputForType(info);
  if (pending) {
    if (info.type === "bool") input.checked = pending.checked;
    else input.value = pending.value;
    input.dataset.dirty = "1";
  }
  const markDirty = () => {
    input.dataset.dirty = "1";
  };
  input.addEventListener("input", markDirty);
  input.addEventListener("change", markDirty);
  const status = el("span", { className: "status-line" });
  const row = el("div", { className: "tunable-row" }, [
    el("div", { className: "tunable-row-header" }, [
      el("span", {
        className: "kv-name",
        textContent: `${info.label ?? name} (${info.type})`,
        title: name,
      }),
    ]),
    el("div", { className: "tunable-row-body" }, [input, status]),
  ]);
  row.dataset.hparamName = name;
  row.dataset.hparamType = info.type;
  return row;
}

function buildInputForType(info) {
  if (info.type === "bool")
    return el("input", { type: "checkbox", checked: Boolean(info.value) });
  if (info.type === "int" || info.type === "float") {
    const attrs = {
      type: "number",
      value: info.value,
      step: info.type === "int" ? 1 : "any",
    };
    if (info.min !== undefined && info.min !== null) attrs.min = info.min;
    if (info.max !== undefined && info.max !== null) attrs.max = info.max;
    return el("input", attrs);
  }
  if (info.type === "str")
    return el("input", { type: "text", value: info.value ?? "" });
  const textarea = el("textarea", {
    value: JSON.stringify(info.value ?? []),
    rows: 2,
    title: info.shape
      ? `shape ${JSON.stringify(info.shape)}, dtype ${info.dtype}, device ${info.device}`
      : "",
  });
  return textarea;
}

function readInputValue(input, type) {
  if (type === "bool") return input.checked;
  if (type === "int") return Number.parseInt(input.value, 10);
  if (type === "float") return Number.parseFloat(input.value);
  if (type === "str") return input.value;
  return JSON.parse(input.value); // tensor: a JSON array
}

// ---- metrics -----------------------------------------------------------------

const METRIC_HISTORY = new Map();
const METRIC_HISTORY_LIMIT = 50;

async function refreshMetrics() {
  const metrics = await fetchJSON("/metrics");
  const container = byId("metrics-list");
  container.replaceChildren();
  const names = Object.keys(metrics);
  if (names.length === 0) {
    container.append(
      el("p", { className: "empty", textContent: "no metrics logged yet" }),
    );
    return;
  }
  for (const [name, metric] of Object.entries(metrics)) {
    const history = METRIC_HISTORY.get(name) ?? [];
    if (
      history.length === 0 ||
      history[history.length - 1].step !== metric.step
    ) {
      history.push({ step: metric.step, value: metric.value });
      if (history.length > METRIC_HISTORY_LIMIT) history.shift();
    }
    METRIC_HISTORY.set(name, history);
    container.append(
      el("div", { className: "kv-row" }, [
        el("span", { className: "kv-name", textContent: name }),
        el("span", {
          className: "kv-value",
          textContent: metric.value.toFixed(4),
        }),
        renderSparkline(history),
      ]),
    );
  }
}

function renderSparkline(history) {
  const width = 100;
  const height = 24;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("class", "sparkline");
  if (history.length < 2) return svg;
  const values = history.map((point) => point.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = history
    .map((point, index) => {
      const x = (index / (history.length - 1)) * width;
      const y = height - ((point.value - min) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const polyline = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "polyline",
  );
  polyline.setAttribute("points", points);
  svg.append(polyline);
  return svg;
}

// ---- pipeline pressure --------------------------------------------------------

async function refreshPipeline() {
  const pipeline = await fetchJSON("/telemetry/pipeline");
  setText("pipeline-summary", JSON.stringify(pipeline, null, 2));
}

// ---- DOT graph rendering (Viz.js, loaded via <script> in index.html) --------

let VIZ_INSTANCE = null;

function vizInstance() {
  VIZ_INSTANCE ??= Viz.instance();
  return VIZ_INSTANCE;
}

/** Renders `dotSource` as an SVG graph into `container` (replacing its content). */
async function renderDot(dotSource, container) {
  try {
    const viz = await vizInstance();
    const svg = viz.renderSVGElement(dotSource);
    svg.classList.add("dot-graph");
    container.replaceChildren(svg);
  } catch (err) {
    container.replaceChildren(
      el("pre", {
        textContent: `(graph render failed: ${err.message})\n\n${dotSource}`,
      }),
    );
  }
}

// ---- inspectors / profiling / cuda memory --------------------------------------

const CAPTURE_KINDS = [
  ["torchinfo_capture", "torchinfo"],
  ["torchview_capture", "torchview"],
  ["torchviz_capture", "torchviz"],
  ["torchlens_capture", "torchlens"],
];

function buildInspectorButtons(availability) {
  const container = byId("capture-buttons");
  container.replaceChildren();
  for (const [kind, pkg] of CAPTURE_KINDS) {
    const resultBox = el("div", { className: "capture-result" });
    const btn = el("button", {
      type: "button",
      textContent: `Capture (${pkg})`,
      disabled: !availability[pkg]?.available,
    });
    btn.addEventListener("click", () => runCaptureCommand(kind, resultBox));
    container.append(
      el("div", { className: "capture-item" }, [btn, resultBox]),
    );
  }
}

async function runCaptureCommand(kind, resultBox) {
  resultBox.textContent = "running…";
  try {
    const command = await submitCommand(kind, {});
    if (command.status !== "succeeded") {
      resultBox.textContent = `failed: ${command.error}`;
      return;
    }
    const manifest = await fetchJSON(`/artifacts/${command.result.id}`);
    await renderArtifactManifest(resultBox, manifest);
  } catch (err) {
    resultBox.textContent = `error: ${err.message}`;
  }
}

async function renderArtifactManifest(container, manifest) {
  container.replaceChildren();
  container.append(
    el("p", {
      className: "empty",
      textContent: `${manifest.kind} · ${manifest.id}`,
    }),
  );
  // Debug captures list their files explicitly; a snapshot manifest doesn't (it
  // always writes exactly one `<kind>.json` payload file alongside it).
  const filenames =
    manifest.files ?? (manifest.kind ? [`${manifest.kind}.json`] : []);
  // A .dot file with a same-stem .svg sibling is already the Graphviz-rendered
  // picture below -- skip re-rendering it client-side and just keep it as source.
  const svgStems = new Set(
    filenames
      .filter((name) => name.endsWith(".svg"))
      .map((name) => name.slice(0, -4)),
  );
  for (const filename of filenames) {
    const url = `${manifest.files_url}/${filename}`;
    if (filename.endsWith(".svg")) {
      container.append(
        el("img", { src: url, alt: filename, className: "artifact-preview" }),
      );
    } else if (filename === "trace.json") {
      // Chrome-trace-format profiler output -- meant for chrome://tracing/Perfetto,
      // not inline display, and can be large; reference it instead of dumping it.
      container.append(
        el("div", { textContent: `${filename} (chrome trace) — ${url}` }),
      );
    } else if (
      filename.endsWith(".dot") &&
      !svgStems.has(filename.slice(0, -4))
    ) {
      const text = await (await fetch(url)).text();
      const graphContainer = el("div", { className: "dot-container" });
      container.append(graphContainer);
      await renderDot(text, graphContainer);
    } else if (/\.(json|txt|dot)$/.test(filename)) {
      const text = await (await fetch(url)).text();
      container.append(el("pre", { textContent: `${filename}:\n${text}` }));
    } else {
      container.append(
        el("div", { textContent: `${filename} (binary) — ${url}` }),
      );
    }
  }
}

function setupInspectorControls() {
  byId("live-summary-btn").addEventListener("click", async () => {
    const text = await fetchFile("/model/summary-live.txt");
    byId("live-view-output").replaceChildren(el("pre", { textContent: text }));
  });
  byId("live-graph-btn").addEventListener("click", async () => {
    const dot = await fetchFile("/model/graph.dot");
    await renderDot(dot, byId("live-view-output"));
  });

  byId("profile-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const args = {
      warmup_steps: Number(byId("profile-warmup").value || 0),
      active_steps: Number(byId("profile-active").value || 1),
      level: byId("profile-level").value,
      then_hold: byId("profile-then-hold").checked,
    };
    const resultBox = byId("profile-result");
    resultBox.textContent = "arming…";
    try {
      // "profile" only arms the profiler and returns immediately -- it writes its
      // actual summary/trace as a debug capture several batches later, off the
      // per-batch training hook, not in this command's own result.
      const before = new Set((await fetchJSON("/artifacts")).debug);
      const command = await submitCommand("profile", args);
      if (command.status !== "succeeded") {
        resultBox.textContent = `failed: ${command.error}`;
        return;
      }
      resultBox.textContent = `armed -- waiting for ${args.warmup_steps + args.active_steps} batch(es)…`;
      const manifest = await pollForArtifact("profiler", before);
      await renderArtifactManifest(resultBox, manifest);
    } catch (err) {
      resultBox.textContent = `error: ${err.message}`;
    }
  });

  byId("lightning-profiler-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const level = byId("lightning-profiler-level").value;
    const resultBox = byId("lightning-profiler-result");
    resultBox.textContent = "setting…";
    try {
      const result = await submitCommand("set_lightning_profiler", { level });
      resultBox.textContent =
        result.status === "succeeded"
          ? `level: ${level}`
          : `failed: ${result.error}`;
    } catch (err) {
      resultBox.textContent = `error: ${err.message}`;
    }
  });

  byId("lightning-profiler-view-btn").addEventListener("click", async () => {
    const resultBox = byId("lightning-profiler-result");
    resultBox.textContent = "loading…";
    try {
      const text = await fetchFile("/model/lightning-profiler-summary.txt");
      resultBox.replaceChildren(el("pre", { textContent: text }));
    } catch (err) {
      resultBox.textContent = `error: ${err.message}`;
    }
  });

  byId("cuda-snapshot-btn").addEventListener("click", async () => {
    const resultBox = byId("cuda-result");
    resultBox.textContent = "capturing…";
    try {
      const manifest = await fetchJSON("/debug/cuda-memory/snapshot");
      await renderArtifactManifest(resultBox, manifest);
    } catch (err) {
      resultBox.textContent = `error: ${err.message}`;
    }
  });
}

async function refreshInspectorState() {
  setText(
    "profiler-state",
    JSON.stringify(await fetchFile("/state/profiler.json"), null, 2),
  );
  setText(
    "cuda-memory-state",
    JSON.stringify(await fetchFile("/state/cuda-memory.json"), null, 2),
  );
  const lightningProfiler = await fetchFile("/state/lightning-profiler.json");
  // Deliberately does not write to lightning-profiler-level's `.value`: that
  // select is the user's pending choice for the *next* "Set" submission, and
  // overwriting it on every poll would clobber a selection made between polls.
  setText("lightning-profiler-current", lightningProfiler.level);
}

// ---- artifacts browser ----------------------------------------------------------

async function refreshArtifacts() {
  const artifacts = await fetchJSON("/artifacts");
  const container = byId("artifacts-list");
  container.replaceChildren();
  for (const category of ["checkpoints", "snapshots", "debug"]) {
    for (const name of artifacts[category] ?? []) {
      const item = el("li", { textContent: `${category}/${name}` });
      item.addEventListener("click", () => showArtifact(category, name));
      container.append(item);
    }
  }
}

async function showArtifact(category, name) {
  const detail = byId("artifact-detail");
  if (category === "checkpoints") {
    detail.replaceChildren(
      el("p", {
        textContent: `checkpoint file: ${name} — /artifact-files/checkpoints/${name}`,
      }),
    );
    return;
  }
  const manifest = await fetchJSON(`/artifacts/${name}`);
  await renderArtifactManifest(detail, manifest);
}

// ---- boot -------------------------------------------------------------------

function main() {
  setupRunControls();
  setupControls();
  setupInspectorControls();
  loadOverview().catch((err) => console.error("overview load failed", err));

  registerPoll(refreshRunStatus);
  registerPoll(refreshMetrics);
  registerPoll(refreshPipeline);
  registerPoll(refreshControlsState);
  registerPoll(refreshInspectorState);
  registerPoll(refreshArtifacts);

  runPollsNow();
  setupPollingControls();
}

document.addEventListener("DOMContentLoaded", main);
