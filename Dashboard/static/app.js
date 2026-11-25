"use strict";
const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="jupiter-token"]').content;
const labels = {inspect: "Schema", train: "Training", forecast: "Forecast", replay: "Replay", spatial: "Spatial map", live: "Live acquisition"};

function controls() {
  const kind = $("kind").value;
  $("source-name").textContent = kind === "replay" ? "Raw capture JSONL" : "Input data CSV";
  $("data-label").hidden = kind === "live";
  $("data").required = kind !== "live";
  $("port-label").hidden = kind !== "live";
  $("baud-label").hidden = kind !== "live";
  $("config-label").hidden = !["replay", "spatial"].includes(kind);
  $("epoch-label").hidden = kind !== "train";
  $("step-label").hidden = kind !== "forecast";
  $("model-label").hidden = kind !== "forecast";
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

function chart(series, axisLabel, secondary = []) {
  const canvas = $("plot");
  const filtered = [...series, ...secondary].filter(Number.isFinite);
  canvas.hidden = filtered.length < 2;
  if (canvas.hidden) return;
  const context = canvas.getContext("2d");
  const min = Math.min(...filtered);
  const max = Math.max(...filtered);
  context.clearRect(0, 0, canvas.width, canvas.height);
  const left = 70, right = canvas.width - 22, top = 30, bottom = canvas.height - 40;
  context.strokeStyle = "#a7b9ca";
  context.beginPath();
  context.moveTo(left, top);
  context.lineTo(left, bottom);
  context.lineTo(right, bottom);
  context.stroke();
  context.fillStyle = "#17243a";
  context.font = "16px system-ui";
  context.fillText(axisLabel, left, 20);
  context.fillText(max.toFixed(2), 8, top + 7);
  context.fillText(min.toFixed(2), 8, bottom);
  function line(values, color) {
    if (values.length < 2) return;
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    let started = false;
    values.forEach((value, index) => {
      if (!Number.isFinite(value)) { started = false; return; }
      const x = left + (right - left) * index / (values.length - 1);
      const y = bottom - (bottom - top) * (value - min) / (max - min || 1);
      if (started) context.lineTo(x, y); else context.moveTo(x, y);
      started = true;
    });
    context.stroke();
  }
  line(series, "#087e6b");
  line(secondary, "#9f4b16");

}

async function view(job, artifact, url) {
  activeLive = null;
  const response = await fetch(url);
  $("details").hidden = false;
  $("detail-title").textContent = labels[job.kind] + " · " + artifact;
  $("plot").hidden = true;
  $("uncertainty").hidden = true;
  if (!response.ok) throw new Error("Result is no longer available");
  if (artifact.endsWith(".json")) {
    const data = await response.json();
    $("report").textContent = JSON.stringify(data, null, 2);
    if (artifact === "model.json") chart(data.training_result?.history || [], "Validation loss by epoch");
    if (job.kind === "train" && artifact === "report.json" && data.fit_diagnostics) {
      chart(data.fit_diagnostics.validation_loss, "Epoch loss · green validation · orange training",
            data.fit_diagnostics.training_loss);
    }
  } else {
    const text = await response.text();
    $("report").textContent = text.slice(0, 12000);
    if (artifact === "forecast.csv") {
      chart(text.trim().split("\n").slice(1).map((line) => Number(line.split(",")[1])), "Air temperature °C");
    }
  }
  $("details").scrollIntoView({block: "nearest"});
}

function heatmap(canvas, grid, label) {
  canvas.hidden = !grid;
  if (!grid) return;
  const values = grid.flat();
  const lo = Math.min(...values), hi = Math.max(...values);
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  const cellWidth = canvas.width / grid[0].length;
  const cellHeight = (canvas.height - 50) / grid.length;
  grid.forEach((row, i) => row.forEach((value, j) => {
    const normalized = (value - lo) / (hi - lo || 1);
    context.fillStyle = "hsl(" + (215 - normalized * 180) + " 64% 45%)";
    context.fillRect(j * cellWidth, 24 + (grid.length - 1 - i) * cellHeight, cellWidth + 1, cellHeight + 1);
  }));
  context.fillStyle = "#17243a";
  context.font = "16px system-ui";
  context.fillText(label, 5, 18);
  context.fillText(lo.toFixed(3) + " to " + hi.toFixed(3) + " · north up", 5, canvas.height - 6);
}

async function viewMap(job) {
  const maps = await api("/api/maps/" + job.id);
  activeLive = null;
  $("details").hidden = false;
  $("detail-title").textContent = "Spatial field";
  $("report").textContent = "";
  heatmap($("plot"), maps.variational, "Variational field");
  heatmap($("uncertainty"), maps.std, "Conditional standard deviation");
}

let activeLive = null;
async function viewLive(job) {
  activeLive = job.id;
  const report = await api("/api/live/" + job.id);
  if (activeLive !== job.id) return;
  $("details").hidden = false;
  $("detail-title").textContent = "Live air temperature";
  $("report").textContent = report.temperature_c.length ? "" : "Waiting for sensor readings";
  $("uncertainty").hidden = true;
  chart(report.temperature_c, "Air temperature °C");
}

function button(label, run) {
  const node = document.createElement("button");
  node.type = "button";
  node.textContent = label;
  node.addEventListener("click", () => run().catch((err) => { $("notice").textContent = err.message; }));
  return node;
}

let previousJobs = "";
async function refresh() {
  const jobs = await api("/api/jobs");
  const live = jobs.find((job) => job.id === activeLive);
  if (live) await viewLive(live);
  const signature = JSON.stringify(jobs);
  if (signature === previousJobs) return;
  previousJobs = signature;
  const current = $("model").value;
  $("model").replaceChildren();
  $("jobs").replaceChildren();
  for (const job of jobs) {
    if (job.kind === "train" && job.status === "complete") {
      $("model").add(new Option("Model " + job.id.slice(0, 8), job.id));
    }
    const row = document.createElement("li");
    const text = document.createElement("span");
    const progress = job.progress ? " · epoch " + job.progress.epoch + " / " + job.progress.epochs : "";
    text.textContent = labels[job.kind] + " · " + job.status + progress;
    row.append(text);
    if (["queued", "running"].includes(job.status)) {
      if (job.kind === "live") row.append(button("Live chart", () => viewLive(job)));
      row.append(button("Cancel", async () => { await api("/api/cancel/" + job.id, {}); }));
    }
    for (const artifact of job.artifacts) {
      const url = "/api/artifacts/" + job.id + "/" + artifact;
      const link = document.createElement("a");
      link.href = url;
      link.textContent = artifact === "model.json" ? "Download model.json" : artifact;
      row.append(link);
      if (artifact.endsWith(".npz")) {
        row.append(button("Heatmaps", () => viewMap(job)));
      } else {
        row.append(button(artifact === "model.json" ? "Loss chart" : "View", () => view(job, artifact, url)));
      }
    }
    $("jobs").append(row);
  }
  if ([...$("model").options].some((item) => item.value === current)) $("model").value = current;
  if (!jobs.length) $("jobs").textContent = "No jobs yet";
}

$("kind").addEventListener("change", controls);
$("job").addEventListener("submit", async (event) => {
  event.preventDefault();
  const submit = event.submitter;
  submit.disabled = true;
  try {
    const kind = $("kind").value;
    const files = kind === "live" ? {} : {data: await encode($("data").files[0])};
    if (["replay", "spatial"].includes(kind)) files.config = await encode($("config").files[0]);
    await api("/api/jobs", {kind, files, options: {model: $("model").value,
      epochs: Number($("epochs").value), steps: Number($("steps").value),
      baud: Number($("baud").value), port: $("port").value}});
    $("notice").textContent = "Job queued";
    await refresh();
  } catch (error) { $("notice").textContent = error.message; }
  finally { submit.disabled = false; }
});
controls();
const poll = () => refresh().catch((error) => { $("notice").textContent = error.message; });
poll();
setInterval(poll, 1250);

api("/api/ports").then((ports) => {
  $("live-option").hidden = !ports.length;
  for (const port of ports) $("port").add(new Option(port, port));
}).catch((error) => { $("notice").textContent = error.message; });
