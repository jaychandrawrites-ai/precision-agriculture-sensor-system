"use strict";

const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="jupiter-token"]').content;
const labels = {
  inspect: "Schema inspection", prepare: "AZMET preparation", train: "Model training",
  forecast: "CSV forecast", field: "Field forecast", replay: "Packet replay",
  spatial: "Spatial map", live: "Live acquisition", health: "Sensor quality",
  benchmark: "Forecast benchmark", search: "Architecture search",
  direct_train: "Direct model training", joint: "Synthetic joint experiment",
  adaptive_spatial: "Adaptive reconstruction", ensemble_spatial: "Forecast ensemble map"
};
const features = {
  air_temperature_c: "Air temperature °C", relative_humidity_pct: "Relative humidity %",
  lwi: "Leaf wetness index", eto_mm_interval: "Reference ET mm per interval",
  eta_mm_interval: "Actual ET mm per interval", cotton_heat_stress: "Cotton heat stress index",
  soil_moisture_pct: "Soil moisture %", soil_temperature_c: "Soil temperature °C",
  soil_ec_us_cm: "Soil conductivity µS/cm", soil_ph: "Soil pH", leaf_mv: "Leaf wetness mV"
};
const groups = {
  datasets: ["inspect", "prepare"], experiments: ["train", "benchmark", "search", "direct_train", "joint"],
  forecasts: ["forecast", "field"], fieldwork: ["replay", "live", "health", "spatial", "adaptive_spatial", "ensemble_spatial"]
};
const colors = ["#087e6b", "#9f4b16", "#355b9c", "#8b3278", "#6c681d", "#573ab2", "#374557"];
const datasetKinds = groups.experiments.filter((kind) => kind !== "joint");
const configLabels = {
  prepare: "AZMET mapping JSON", replay: "Packet schema JSON",
  field: "Field calibration and settings JSON", spatial: "Grid and reconstruction JSON",
  adaptive_spatial: "Adaptive reconstruction JSON", ensemble_spatial: "Forecast map JSON"
};
const mapLabels = {
  variational: "Variational field", idw: "Inverse distance weighted field", kriging: "Kriging field",
  mean: "Conditional mean", std: "Conditional standard deviation", lower: "Conditional lower bound",
  upper: "Conditional upper bound", split_rhat: "Split R-hat", ess: "Effective sample size"
};
const metricLabels = {
  rmse: "RMSE", mae: "MAE", bias: "Bias", r2: "R squared",
  persistence_skill: "Persistence skill", seasonal_skill: "Seasonal persistence skill",
  coverage: "Marginal interval coverage", mean_width: "Mean interval width"
};

function controls() {
  const kind = $("kind").value;
  const useSavedDataset = datasetKinds.includes(kind) && $("dataset").value;
  const needUpload = !["live", "health", "field", "joint"].includes(kind) && !useSavedDataset;
  $("source-name").textContent = kind === "replay" ? "Raw capture JSONL" : "Input data CSV";
  $("config-name").textContent = configLabels[kind] || "Configuration JSON";
  const visibility = {
    data: needUpload, config: Boolean(configLabels[kind]), dataset: datasetKinds.includes(kind),
    port: kind === "live", baud: kind === "live", epoch: datasetKinds.includes(kind) || kind === "joint",
    model: ["forecast", "field"].includes(kind), step: ["forecast", "field"].includes(kind),
    session: ["field", "health"].includes(kind), inspection: kind === "inspect", header: kind === "inspect",
    optimizer: kind === "train", loss: kind === "train", horizon: ["benchmark", "direct_train"].includes(kind),
    architecture: kind === "direct_train", width: kind === "direct_train", "joint-seed": kind === "joint",
    seeds: kind === "benchmark", synthetic: kind === "benchmark", population: kind === "search",
    generations: kind === "search", "candidate-epochs": kind === "search"
  };
  for (const [field, show] of Object.entries(visibility)) $(field + "-label").hidden = !show;
  $("data").required = Boolean(needUpload);
  $("config").required = Boolean(configLabels[kind]);
  $("model").required = visibility.model;
  $("session").required = visibility.session;
  $("epochs").max = kind === "train" ? 300 : 50;
}

async function api(url, payload) {
  const options = payload === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Jupiter-Token": token},
    body: JSON.stringify(payload)
  };
  const response = await fetch(url, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || "Request failed");
  return value;
}

function encode(file) {
  if (!file || file.size > 12 * 1024 * 1024) throw new Error("Choose a file of at most 12 MiB");
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Unable to read the selected file"));
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.readAsDataURL(file);
  });
}

function legend(series) {
  const items = series.map((name, index) => {
    const item = document.createElement("li");
    const marker = document.createElement("b");
    marker.classList.add("series-" + index % colors.length);
    item.append(marker, name);
    return item;
  });
  $("legend").replaceChildren(...items);
}

function short(value) {
  return typeof value === "number" ? Number(value.toPrecision(5)).toString() : (value ?? "—");
}

function table(columns, rows) {
  const element = document.createElement("table");
  const head = element.createTHead().insertRow();
  for (const label of columns) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    head.append(cell);
  }
  const body = element.createTBody();
  for (const row of rows) {
    const line = body.insertRow();
    for (const value of row) line.insertCell().textContent = short(value);
  }
  $("table").replaceChildren(element);
}

function chart(series, title, xAxis) {
  const canvas = $("plot");
  const values = Object.values(series).flat().filter(Number.isFinite);
  canvas.hidden = values.length === 0;
  legend(canvas.hidden ? [] : Object.keys(series));
  if (canvas.hidden) return;
  canvas.setAttribute("aria-label", title);
  const context = canvas.getContext("2d");
  const min = Math.min(...values), max = Math.max(...values);
  context.clearRect(0, 0, canvas.width, canvas.height);
  const left = 92, right = canvas.width - 22, top = 35, bottom = canvas.height - 42;
  context.strokeStyle = "#a7b9ca";
  context.beginPath();
  context.moveTo(left, top);
  context.lineTo(left, bottom);
  context.lineTo(right, bottom);
  context.stroke();
  context.fillStyle = "#17243a";
  context.font = "16px system-ui";
  context.fillText(title, left, 23);
  context.fillText(short(max), 6, top + 5);
  context.fillText(short(min), 6, bottom);
  context.fillText(xAxis, left, canvas.height - 11);
  Object.values(series).forEach((line, index) => {
    context.strokeStyle = context.fillStyle = colors[index % colors.length];
    context.lineWidth = 2;
    let started = false;
    context.beginPath();
    line.forEach((value, offset) => {
      if (!Number.isFinite(value)) { started = false; return; }
      const x = left + (right - left) * offset / Math.max(1, line.length - 1);
      const y = min === max ? (bottom + top) / 2 : bottom - (bottom - top) * (value - min) / (max - min);
      if (started) context.lineTo(x, y); else context.moveTo(x, y);
      if (line.length === 1) context.fillRect(x - 2, y - 2, 4, 4);
      started = true;
    });
    context.stroke();
  });
}

function heatmap(grid, label) {
  const canvas = $("plot");
  canvReset();
  const values = grid.flat().filter(Number.isFinite);
  canvas.hidden = !values.length;
  if (!values.length) return;
  canvas.setAttribute("aria-label", label);
  const lo = Math.min(...values), hi = Math.max(...values);
  const context = canvas.getContext("2d");
  const cellWidth = canvas.width / grid[0].length;
  const cellHeight = (canvas.height - 50) / grid.length;
  grid.forEach((row, i) => row.forEach((value, j) => {
    const normalized = (value - lo) / (hi - lo || 1);
    context.fillStyle = Number.isFinite(value) ? "hsl(" + (215 - normalized * 180) + " 64% 45%)" : "#d9e0e8";
    context.fillRect(j * cellWidth, 24 + (grid.length - 1 - i) * cellHeight, cellWidth + 1, cellHeight + 1);
  }));
  context.fillStyle = "#17243a";
  context.font = "16px system-ui";
  context.fillText(label, 5, 18);
  context.fillText(short(lo) + " to " + short(hi) + " · north up", 5, canvas.height - 6);
}

function canvReset() {
  const canvas = $("plot");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  legend([]);
}

function report(data) {
  $("report").hidden = data === null;
  $("report").textContent = data === null ? "" : (typeof data === "string" ? data : JSON.stringify(data, null, 2));
}

function select(id, values) {
  const item = $(id);
  const existing = item.value;
  const options = Object.entries(values);
  item.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
  if (Object.hasOwn(values, existing)) item.value = existing;
  $(id + "-label").hidden = !options.length;
}

function featureOptions(values) {
  return Object.fromEntries(values.map((feature) => [feature, features[feature] || feature]));
}

let selectedResult = null;
let activeLive = null;
let resultRequest = 0;

function showResult(job, mode, data, live = false) {
  if (!live) activeLive = null;
  selectedResult = {job, mode, data};
  $("details").hidden = false;
  $("detail-title").textContent = labels[job.kind];
  if (["forecast", "sensor"].includes(mode)) select("feature", featureOptions(Object.keys(data.series)));
  else if (mode === "dataset") select("feature", featureOptions(Object.keys(data.distributions)));
  else if (mode === "research") select("feature", featureOptions([...new Set(data.metrics.map((row) => row.feature))]));
  else select("feature", {});
  if (mode === "dataset") select("view", {distribution: "Distribution", seasonal: "Monthly profile", quality: "Quality"});
  else if (mode === "research") {
    const metrics = Object.entries(metricLabels).filter(([key]) => data.metrics.some((row) => row[key] !== null));
    select("view", {...Object.fromEntries(metrics), ...(data.search ? {search: "Search diagnostics"} : {})});
  } else if (mode === "map") select("view", Object.fromEntries(Object.keys(data.layers || data).map((key) => [key, mapLabels[key] || key])));
  else select("view", {});
  renderResult();
  if (!live) $("details").scrollIntoView({block: "nearest"});
}

function renderResult() {
  if (!selectedResult) return;
  const {job, mode, data} = selectedResult;
  const feature = $("feature").value;
  const view = $("view").value;
  canvReset();
  $("plot").hidden = true;
  $("table").replaceChildren();
  report(null);
  if (mode === "map") {
    const layers = data.layers || data;
    const grid = layers[view];
    if (grid) heatmap(grid, mapLabels[view] || view);
  } else if (mode === "forecast" || mode === "sensor") {
    const values = data.series[feature] || [];
    chart({[features[feature]]: values}, features[feature], mode === "forecast" ? "Forecast hour" : "Recent records");
    if (mode === "forecast") table(["Time UTC", features[feature]], data.timestamps.map((time, index) => [time, values[index]]));
    else if (!values.length) report("Waiting for sensor readings");
  } else if (mode === "dataset") {
    if (view === "quality") report({quality: data.quality, provenance: data.provenance});
    if (view === "distribution" && data.distributions[feature]) {
      const histogram = data.distributions[feature];
      chart({"Frequency": histogram.bins}, features[feature], "Equal-width intervals");
      table(["Lower edge", "Upper edge", "Count"], histogram.bins.map((count, i) => [histogram.edges[i], histogram.edges[i + 1], count]));
      report({missing_hours: histogram.missing});
    }
    if (view === "seasonal") {
      const rows = Object.entries(data.seasonal.months)
        .map(([month, samples]) => [month, samples.features[feature]])
        .filter(([, sample]) => sample);
      chart({"Mean": rows.map(([, item]) => item.mean),
        "95th percentile": rows.map(([, item]) => item.p95)}, features[feature], "Calendar month");
      table(["Month", "Mean", "5th percentile", "95th percentile", "Missing"],
        rows.map(([month, item]) => [month, item.mean, item.p05, item.p95, item.missing]));
    }
  } else if (mode === "research") {
    if (view === "search") {
      const rows = Object.entries(data.search);
      table(["Method", "Parameters", "Candidates", "Sample steps", "Parameter sample steps", "Winning bits"],
        rows.map(([name, item]) => [name, item.parameters, item.candidate_count, item.candidate_model_sample_steps,
          item.candidate_parameter_sample_steps, item.winning_bits?.join("")]));
      const traces = Object.fromEntries(rows.filter(([, item]) => item.generations?.length)
        .map(([name, item]) => [name, item.generations.map((record) => record.best_selection_score)]));
      chart(traces, "Best validation selection score", "Generation");
      report({protocol: data.protocol, pareto_fronts: Object.fromEntries(rows.map(([name, item]) => [name, item.validation_pareto_front]))});
    } else {
      const metrics = data.metrics.filter((row) => row.feature === feature && row[view] !== null);
      const series = {};
      for (const row of metrics) {
        (series[row.label] ??= [])[row.lead_hours - 1] = row[view];
      }
      chart(series, metricLabels[view] + " · " + features[feature], "Lead hour");
      table(["Method", "Lead hours", metricLabels[view]],
        metrics.map((row) => [row.label, row.lead_hours, row[view]]));
      if (data.manifest) report({manifest: data.manifest, windows: data.windows});
    }
  } else {
    report(data);
    if (job.kind === "train" && data.fit_diagnostics) {
      chart({"Validation": data.fit_diagnostics.validation_loss || [],
        "Training": data.fit_diagnostics.training_loss || []}, "Epoch loss", "Epoch");
    }
  }
}

async function inspectArtifact(job, artifact) {
  activeLive = null;
  const request = ++resultRequest;
  const response = await fetch("/api/artifacts/" + job.id + "/" + artifact);
  if (!response.ok) throw new Error("Result is no longer available");
  const data = artifact.endsWith(".json") ? await response.json() : (await response.text()).slice(0, 12000);
  if (request === resultRequest) showResult(job, "artifact", data);
}

async function inspectView(job, route, mode) {
  activeLive = null;
  const request = ++resultRequest;
  const data = await api("/api/" + route + "/" + job.id);
  if (request === resultRequest) showResult(job, mode, data);
}

async function viewLive(job) {
  activeLive = job.id;
  const data = await api("/api/live/" + job.id);
  if (activeLive === job.id) showResult(job, "sensor", data, true);
}

function button(label, run) {
  const node = document.createElement("button");
  node.type = "button";
  node.textContent = label;
  node.addEventListener("click", () => run().catch((error) => { $("notice").textContent = error.message; }));
  return node;
}

function link(label, href) {
  const node = document.createElement("a");
  node.href = href;
  node.textContent = label;
  return node;
}

function availableSelections(jobs) {
  const sets = {
    dataset: ["Upload prepared CSV", "prepare", "prepared/hourly.csv"],
    session: ["Choose a session", ["replay", "live"], "session/readings.jsonl"],
    model: ["Choose a model", "train", "model.json"]
  };
  for (const [id, [placeholder, kinds, artifact]] of Object.entries(sets)) {
    const item = $(id), current = item.value;
    const entries = jobs.filter((job) => [kinds].flat().includes(job.kind) && job.artifacts.includes(artifact));
    item.replaceChildren(new Option(placeholder, ""), ...entries.map((job) =>
      new Option(labels[job.kind] + " · " + new Date(job.created_at).toLocaleString(), job.id)));
    if (entries.some((job) => job.id === current)) item.value = current;
  }
  controls();
}

function jobRow(job) {
  const row = document.createElement("li");
  const status = document.createElement("span");
  status.textContent = labels[job.kind] + " · " + job.status;
  row.append(status);
  if (job.progress?.stage === "epoch") {
    const progress = document.createElement("progress");
    progress.max = job.progress.total;
    progress.value = job.progress.completed;
    progress.setAttribute("aria-label", "Epoch " + progress.value + " of " + progress.max);
    row.append(progress);
  } else if (job.progress?.stage === "search") {
    const summary = document.createElement("span");
    summary.textContent = Object.entries(job.progress.generations)
      .map(([name, generation]) => name + " " + generation + " / " + job.progress.total).join(" · ");
    row.append(summary);
  }
  if (["queued", "running"].includes(job.status)) {
    if (job.kind === "live") row.append(button("Live chart", () => viewLive(job)));
    row.append(button("Cancel", async () => { await api("/api/cancel/" + job.id, {}); await refresh(); }));
  }
  if (job.can_resume) row.append(button("Resume", async () => { await api("/api/resume/" + job.id, {}); await refresh(); }));
  if (job.artifacts.includes("prepared/hourly.csv")) row.append(button("Dataset profile", () => inspectView(job, "dataset", "dataset")));
  if (job.artifacts.includes("forecast.csv") || job.artifacts.includes("inference/forecast.csv")) {
    row.append(button("Forecast chart", () => inspectView(job, "forecast", "forecast")));
  }
  if (job.artifacts.includes("session/readings.jsonl")) row.append(button("Reading chart", () => inspectView(job, "live", "sensor")));
  if (job.artifacts.some((name) => ["map/field.npz", "map/forecast-field.npz"].includes(name))) row.append(button("Heatmaps", () => inspectView(job, "maps", "map")));
  if (job.can_export) {
    row.append(button("Metrics and results", () => inspectView(job, "research", "research")));
    row.append(link("Metrics CSV", "/api/metrics/" + job.id));
  }
  for (const artifact of job.artifacts) {
    row.append(link(artifact, "/api/artifacts/" + job.id + "/" + artifact));
    if (artifact.endsWith("report.json") || artifact.endsWith("failure.log")) {
      row.append(button(artifact.endsWith("failure.log") ? "See error" : "View report", () => inspectArtifact(job, artifact)));
    }
  }
  return row;
}

let previousJobs = "";
async function refresh() {
  const jobs = await api("/api/jobs");
  const live = jobs.find((job) => job.id === activeLive);
  if (live) await viewLive(live);
  const kind = $("job-filter").value;
  const signature = JSON.stringify([jobs, kind]);
  if (signature === previousJobs) return;
  previousJobs = signature;
  availableSelections(jobs);
  const visible = kind ? jobs.filter((job) => groups[kind].includes(job.kind)) : jobs;
  $("jobs").replaceChildren(...visible.map(jobRow));
  if (!visible.length) $("jobs").textContent = "No matching jobs";
}

function options() {
  return {
    model: $("model").value, dataset: $("dataset").value, session: $("session").value,
    architecture: $("architecture").value, width: Number($("width").value),
    seed: Number($("joint-seed").value), epochs: Number($("epochs").value), steps: Number($("steps").value),
    baud: Number($("baud").value), port: $("port").value, optimizer: $("optimizer").value,
    loss: $("loss").value, delimiter: $("delimiter").value === "tab" ? "\t" : $("delimiter").value,
    headerless: $("headerless").checked, horizon: Number($("horizon").value),
    seeds: $("seeds").value.split(",").filter((seed) => seed.trim()).map(Number),
    synthetic: $("synthetic").checked, population: Number($("population").value),
    generations: Number($("generations").value), candidate_epochs: Number($("candidate-epochs").value)
  };
}

document.querySelectorAll("[data-operation]").forEach((link) => {
  link.addEventListener("click", () => {
    $("kind").value = link.dataset.operation;
    controls();
  });
});
$("kind").addEventListener("change", controls);
$("dataset").addEventListener("change", controls);
$("view").addEventListener("change", renderResult);
$("feature").addEventListener("change", renderResult);
$("job-filter").addEventListener("change", () => refresh().catch((error) => { $("notice").textContent = error.message; }));
$("job").addEventListener("submit", async (event) => {
  event.preventDefault();
  const submit = event.submitter;
  submit.disabled = true;
  try {
    const kind = $("kind").value, files = {};
    if (!$("data-label").hidden) files.data = await encode($("data").files[0]);
    if (configLabels[kind]) files.config = await encode($("config").files[0]);
    await api("/api/jobs", {kind, files, options: options()});
    $("notice").textContent = "Job queued";
    await refresh();
  } catch (error) { $("notice").textContent = error.message; }
  finally { submit.disabled = false; }
});
controls();
function poll() {
  refresh().catch((error) => { $("notice").textContent = error.message; })
    .finally(() => setTimeout(poll, 1250));
}
poll();

api("/api/ports").then((ports) => {
  $("live-option").hidden = !ports.length;
  for (const port of ports) $("port").add(new Option(port, port));
}).catch((error) => { $("notice").textContent = error.message; });
