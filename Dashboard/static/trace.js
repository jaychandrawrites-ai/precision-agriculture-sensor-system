"use strict";

const trace = {
  token: document.querySelector('[name="jupiter-token"]').content,
  id: "",
  survey: null,
  receiver: null,
  busy: false,
  readiness: null,
};
const el = id => document.getElementById(id);
const path = (id, action = "") => "/api/trace/" + encodeURIComponent(id) + (action ? "/" + action : "");
const measurements = {
  air_temperature_c: "Air temperature (°C)", relative_humidity_pct: "Relative humidity (%)",
  soil_temperature_c: "Soil temperature (°C)", soil_moisture_pct: "Soil moisture (%)",
  soil_ec_us_cm: "Soil conductivity (µS/cm)", soil_ph: "Soil pH", leaf_mv: "Leaf voltage (mV)",
  gps_hdop: "GPS HDOP", gps_satellites: "GPS satellites",
};

async function request(url, payload) {
  const headers = {"X-Jupiter-Token": trace.token};
  const options = {headers};
  if (payload !== undefined) {
    options.method = "POST";
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(payload);
  }
  const result = await fetch(url, options);
  const content = await result.json();
  if (!result.ok) throw new Error(content.error || "Trace request failed");
  return content;
}

async function jsonFile(input, required = false) {
  const file = input.files[0];
  if (!file) {
    if (required) throw new Error("Select a JSON file.");
    return undefined;
  }
  if (file.size > 24 * 1024 * 1024) throw new Error("Selected file exceeds the 24 MB upload limit.");
  return JSON.parse(await file.text());
}

async function base64File(file) {
  if (!file || file.size > 24 * 1024 * 1024) throw new Error("Select a raw archive under 24 MB.");
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (let index = 0; index < bytes.length; index += 16384) {
    binary += String.fromCharCode(...bytes.subarray(index, index + 16384));
  }
  return btoa(binary);
}

function option(value, label) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  return item;
}
function text(tag, content) {
  const node = document.createElement(tag);
  node.textContent = content;
  return node;
}
function setOptions(target, options, emptyLabel = "Select one") {
  const selected = target.value;
  target.replaceChildren(option("", emptyLabel), ...options.map(([value, label]) => option(value, label)));
  if ([...target.options].some(item => item.value === selected)) target.value = selected;
}
function readout(target, entries) {
  target.replaceChildren(...entries.flatMap(([label, value]) => [text("dt", label), text("dd", String(value))]));
}
function setDisabled(id, disabled) {
  for (const control of el(id).querySelectorAll("input, select, button")) control.disabled = disabled;
}
function updateLink(id, query) {
  if (trace.id) el(id).href = path(trace.id, "export") + "?" + query;
  else el(id).removeAttribute("href");
}
function download(query, label) {
  const link = text("a", label);
  link.href = path(trace.id, "export") + "?" + query;
  return link;
}
function shortTime(value) {
  return value ? new Date(value).toLocaleString() : "in progress";
}

function render() {
  const active = trace.survey?.active_segment_id;
  const closed = Boolean(trace.survey?.finished_at);
  const open = Boolean(trace.survey);
  const receiver = trace.receiver || {state: "disconnected", bytes_received: 0, packets_received: 0};
  const latest = receiver.latest;
  const connected = ["connected", "connecting", "reconnecting"].includes(receiver.state) || receiver.native_readiness?.ready === false;
  const blockStart = ["connecting", "reconnecting", "failed"].includes(receiver.state);
  const entries = [["Status", receiver.state], ["Port", receiver.port || "Not connected"],
    ["Baud rate", receiver.baud ?? "—"], ["Received bytes", receiver.bytes_received],
    ["Raw frames", receiver.packets_received], ["Buffered bytes", receiver.pending_frame_bytes ?? 0],
    ["Successful parses", receiver.successful_parses ?? 0], ["Parse failures", receiver.parse_failures ?? 0],
    ["Duplicate packets", receiver.duplicate_packets ?? 0],
    ["GPS", latest?.gps_validity?.replace("_", " ") || "Not reported"],
    ["Valid GPS fixes", receiver.valid_gps_fixes ?? 0], ["Invalid GPS reports", receiver.invalid_gps_reports ?? 0],
    ["Native readiness", receiver.native_readiness ? (receiver.native_readiness.ready ? "Verified" : "Waiting for fresh sensors and stable GPS") : "—"],
    ["Stationary GPS samples", receiver.native_readiness?.gps.valid_samples ?? "—"],
    ["Device restarts", receiver.restart_count ?? 0],
    ["Last frame received", receiver.last_receipt ? shortTime(receiver.last_receipt) : "—"],
    ["Latest measurement time", latest?.observed_at ? shortTime(latest.observed_at) : "Not supplied"],
    ["Quality", latest ? latest.flags.join(", ") || "No parser flags" : "Awaiting parsed measurements"]];
  if (receiver.error) entries.push(["Current receiver message", receiver.error]);
  if (receiver.most_recent_error) {
    const error = receiver.most_recent_error;
    entries.push(["Most recent error", error.message],
      ["Error time", shortTime(error.at)], ["Recovery", error.recoverable ? "Reconnectable" : "Reconnect manually"]);
  }
  const report = trace.readiness;
  el("trace-readiness").textContent = report
    ? report.ready_locations + " of " + report.locations.length + " locations ready in the last check."
    : "Forecast readiness has not been checked for these recordings.";
  readout(el("trace-diagnostics"), entries);
  readout(el("trace-measurements"), Object.entries(latest?.measurements || {}).map(
    ([name, value]) => [measurements[name] || name, value]));
  readout(el("trace-devices"), [
    ...Object.entries(receiver.device_packet_counts || {}).map(([key, count]) => ["Device " + key, count + " packets"]),
    ...Object.entries(receiver.missing_measurements || {}).map(([key, count]) => ["Missing " + (measurements[key] || key), count]),
    ...(receiver.recent_restart_boundaries || []).map(item => ["Restart", [item.device_id || "Unknown device",
      item.boot_id || "Unknown boot", shortTime(item.at)].join(" · ")]),
  ]);
  readout(el("trace-units"), Object.entries(receiver.configured_units || {}).map(
    ([name, units]) => [measurements[name] || name, units.source + " received · " + units.normalized + " stored"]));
  readout(el("trace-calibration"), Object.entries(receiver.calibration || {}).flatMap(([name, calibration]) => {
    const label = name.startsWith("leaf") ? "Leaf wetness calibration" : "Soil moisture calibration";
    const details = [[label, calibration.state.replaceAll("_", " ")]];
    if (calibration.raw_field) details.push(["Raw " + (measurements[calibration.raw_field] || calibration.raw_field), calibration.raw_value]);
    if ("calibrated_value" in calibration) {
      const calibrated = calibration.calibrated_field === "lwi" ? "Calibrated leaf wetness index" : "Calibrated soil water fraction";
      details.push([calibrated, calibration.calibrated_value]);
    }
    return details;
  }));
  const recording = (trace.survey?.locations || []).flatMap(location =>
    location.segments.map(segment => ({label: location.label, ...segment}))).find(segment => segment.id === active);
  const elapsed = recording ? Math.max(0, (Date.now() - Date.parse(recording.started_at)) / 1000) : 0;
  el("trace-recording").textContent = recording
    ? `${recording.label} · ${Math.floor(elapsed)} s · ${recording.sample_count} parsed samples · ${recording.raw_count} raw frames`
    : closed ? "Survey finished." : "No active recording.";
  setDisabled("trace-start", !open || closed || Boolean(active) || blockStart || trace.busy);
  setDisabled("trace-revisit", !open || closed || Boolean(active) || blockStart || !trace.survey?.locations.length || trace.busy);
  setDisabled("trace-replay", !open || closed || Boolean(active) || connected || trace.busy);
  setDisabled("trace-forecast-form", !open || trace.busy);
  setDisabled("trace-ensemble-form", !open || trace.busy);
  setDisabled("trace-map-form", !open || trace.busy);
  el("trace-connect").querySelector("button").disabled =
    !open || closed || Boolean(active) || connected || el("trace-ports").options.length < 2 || trace.busy;
  el("trace-disconnect").disabled = !open || (!connected && receiver.state !== "failed") || trace.busy;
  el("trace-stop").disabled = !open || !active || trace.busy;
  el("trace-finish").disabled = !open || closed || trace.busy;
  el("trace-start-button").textContent = trace.survey?.locations.length ? "Record New Location" : "Start Recording";
  updateLink("trace-export", "format=json");
  updateLink("trace-export-csv", "format=csv");
  updateLink("trace-export-raw", "format=raw");
  const locations = trace.survey?.locations || [];
  setOptions(el("trace-location"), locations.map(value => [value.id, value.label]), "Select a location");
  setOptions(el("trace-replay-location"), locations.map(value => [value.id, value.label]), "Select a location");
  const list = locations.map(location => {
    const article = document.createElement("article");
    article.append(text("h3", location.label),
      text("p", location.manual_coordinates_available ? "Manual coordinates saved" : "GPS provenance is recorded per sample"));
    const body = document.createElement("ul");
    for (const segment of location.segments) {
      const item = text("li", `${shortTime(segment.started_at)} – ${shortTime(segment.ended_at)} · ${segment.status} · ${segment.sample_count} samples `);
      item.append(download("format=csv&segment=" + encodeURIComponent(segment.id), "Segment CSV"));
      body.append(item);
    }
    article.append(download("format=csv&location=" + encodeURIComponent(location.id), "Location CSV"), body);
    return article;
  });
  el("trace-history").replaceChildren(...(list.length ? list : [text("p", "No locations recorded.")]));
}
async function listSurveys() {
  const surveys = await request("/api/trace");
  setOptions(el("trace-surveys"), surveys.map(item => [item.id, item.label]), "Select a survey");
  el("trace-surveys").value = trace.id;
}
async function refresh() {
  if (!trace.id) { trace.survey = trace.receiver = null; render(); return; }
  const id = trace.id;
  const answer = await request(path(id));
  if (id !== trace.id) return;
  if (trace.survey?.active_segment_id !== answer.survey.active_segment_id) trace.readiness = null;
  trace.survey = answer.survey;
  trace.receiver = answer.receiver;
  render();
}
async function perform(action, payload = {}) {
  const result = await request(path(trace.id, action), payload);
  if (["start", "stop", "finish", "replay"].includes(action)) trace.readiness = null;
  return result;
}
function operation(fn, success = "") {
  return async event => {
    event?.preventDefault();
    if (trace.busy) return;
    trace.busy = true; render();
    el("trace-alert").textContent = "";
    try {
      await fn(event);
      await refresh();
      el("trace-alert").textContent = success;
    } catch (error) {
      el("trace-alert").textContent = error.message;
    } finally {
      trace.busy = false;
      render();
    }
  };
}
function submit(id, fn, success) { el(id).addEventListener("submit", operation(fn, success)); }
function press(id, fn, success) { el(id).addEventListener("click", operation(fn, success)); }
function showResult(result) {
  el("trace-result").hidden = false;
  el("trace-result").textContent = JSON.stringify(result, null, 2);
}
function drawMap(values, id = "trace-map", label = "Observational map") {
  const canvas = el(id);
  canvas.hidden = !Array.isArray(values) || !values.length;
  if (canvas.hidden) return;
  const ctx = canvas.getContext("2d");
  const finite = values.flat().filter(Number.isFinite);
  const min = Math.min(...finite), max = Math.max(...finite);
  const rows = values.length, cols = values[0].length;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  values.forEach((row, y) => row.forEach((v, x) => {
    const t = max > min ? (v - min) / (max - min) : .5;
    ctx.fillStyle = Number.isFinite(v) ? `hsl(${215 - 175 * t}, 65%, 50%)` : "#b8c1c5";
    const width = canvas.width / cols, height = canvas.height / rows;
    ctx.fillRect(x * width, (rows - 1 - y) * height, width + .6, height + .6);
  }));
  canvas.setAttribute("aria-label", label + ", values from " + min.toFixed(2) + " to " + max.toFixed(2));
}
function drawAdaptive(cells, field, id, label) {
  const canvas = el(id);
  const values = cells[field].filter((value, i) => cells.inside[i] && Number.isFinite(value));
  canvas.hidden = !values.length;
  if (canvas.hidden) return;
  const ctx = canvas.getContext("2d");
  const min = Math.min(...values), max = Math.max(...values);
  const west = Math.min(...cells.bounds_m.map(cell => cell[0]));
  const south = Math.min(...cells.bounds_m.map(cell => cell[1]));
  const east = Math.max(...cells.bounds_m.map(cell => cell[2]));
  const north = Math.max(...cells.bounds_m.map(cell => cell[3]));
  const sx = canvas.width / (east - west), sy = canvas.height / (north - south);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  cells.bounds_m.forEach(([left, bottom, right, top], index) => {
    const value = cells[field][index];
    if (!cells.inside[index] || !Number.isFinite(value)) return;
    const t = max > min ? (value - min) / (max - min) : .5;
    ctx.fillStyle = "hsl(" + (215 - 175 * t) + ", 65%, 50%)";
    ctx.fillRect((left - west) * sx, (north - top) * sy, (right - left) * sx + .6, (top - bottom) * sy + .6);
  });
  canvas.setAttribute("aria-label", label + ", values from " + min.toFixed(2) + " to " + max.toFixed(2));
}
submit("trace-create", async event => {
  const survey = await request("/api/trace", {label: event.target.elements.label.value});
  trace.id = survey.id;
  trace.readiness = null;
  event.target.reset();
  await listSurveys();
}, "Survey created.");
submit("trace-restore", async event => {
  const archive_bytes = await base64File(event.target.elements.archive.files[0]);
  const survey = await request("/api/trace/restore", {archive_bytes});
  trace.id = survey.id;
  trace.readiness = null;
  await listSurveys();
}, "Survey imported.");
submit("trace-import-batch", async event => {
  const fields = event.target.elements;
  const selected = [...fields.files.files];
  const labels = fields.locations.value.split(",").map(value => value.trim());
  if (fields.locations.value.trim() && labels.length !== selected.length) {
    throw new Error("Enter one location label per file.");
  }
  const files = await Promise.all(selected.map(async (file, index) => ({
    name: file.name, data: await base64File(file), location: labels[index] || null
  })));
  const preview = event.submitter.value === "preview";
  const result = await request("/api/trace/import-csv-batch", {
    label: fields.label.value, timezone: fields.timezone.value.trim(), files, preview
  });
  if (preview) {
    showResult(result);
  } else {
    trace.id = result.id;
    await listSurveys();
    await refresh();
  }
}, "CSV batch processed.");
submit("trace-import-csv", async event => {
  const fields = event.target.elements;
  const survey = await request("/api/trace/import-csv", {label: fields.label.value,
    csv: await base64File(fields.csv.files[0]), mapping: await jsonFile(fields.mapping, true)});
  trace.id = survey.id;
  trace.readiness = null;
  await listSurveys();
}, "Measurements imported.");
submit("trace-connect", async event => perform("connect", {port: event.target.elements.port.value,
  baud: Number(event.target.elements.baud.value), mapping: el("trace-protocol").value || await jsonFile(el("trace-mapping"))}),
"Receiver connection requested.");
for (const action of ["start-all", "stop-all"]) {
  press("trace-" + action, async () => {
    showResult(await request("/api/trace/" + action, {}));
    if (trace.id) await refresh();
  }, "Receiver group results available.");
}
press("trace-aligned-export", () => {
  const seconds = el("trace-alignment");
  if (!seconds.value || !seconds.checkValidity()) throw new Error("Choose a valid alignment interval.");
  window.location.assign("/api/trace/aligned-export?seconds=" + encodeURIComponent(seconds.value));
});
press("trace-disconnect", () => perform("disconnect"), "Receiver disconnected.");
submit("trace-start", async event => {
  const fields = event.target.elements;
  if (Boolean(fields.latitude.value) !== Boolean(fields.longitude.value)) throw new Error("Supply both coordinates or leave both blank.");
  await perform("start", {label: fields.label.value,
    coordinates: fields.latitude.value ? [Number(fields.latitude.value), Number(fields.longitude.value)] : undefined,
    calibration: await jsonFile(fields.calibration)});
  event.target.reset();
}, "Location recording started.");
submit("trace-revisit", event => perform("start", {location_id: event.target.elements.location.value}),
"Revisit recording started.");
press("trace-stop", () => perform("stop"), "Location saved. You can record another location.");
press("trace-finish", () => perform("finish"), "Survey finalized.");
submit("trace-replay", async event => {
  const fields = event.target.elements;
  if (Boolean(fields.location.value) === Boolean(fields.label.value.trim())) {
    throw new Error("Enter a new location name or select one location to revisit.");
  }
  await perform("replay", {archive: await base64File(fields.archive.files[0]),
    label: fields.label.value || undefined, location_id: fields.location.value || undefined,
    mapping: el("trace-protocol").value || await jsonFile(el("trace-mapping"))});
}, "Replay saved as its own recording segment.");
submit("trace-forecast-form", async event => {
  const action = event.submitter.value;
  const fields = event.target.elements;
  if (action === "forecast" && !fields.model.value) throw new Error("Select a saved model.");
  const report = await perform(action, {configuration: await jsonFile(fields.configuration, true),
    model: fields.model.value || undefined, steps: Number(fields.steps.value)});
  trace.readiness = report;
  showResult(report);
}, "Forecast report available.");
function mapDomain() {
  const fields = ["map-latitude", "map-longitude", "map-radius"].map(id => el(id));
  const resolution = Number(el("map-resolution").value);
  if (!Number.isInteger(resolution) || resolution < 2 || resolution > 100) {
    throw new Error("Choose a map resolution from 2 to 100.");
  }
  if (fields.every(field => !field.value)) return {resolution};
  if (fields.some(field => !field.value || !field.checkValidity())) {
    throw new Error("Enter a valid latitude, longitude and radius.");
  }
  const [latitude, longitude, radius_m] = fields.map(field => Number(field.value));
  return {resolution, domain: {center_lat_lon: [latitude, longitude], radius_m}};
}

submit("trace-ensemble-form", async event => {
  const fields = event.target.elements;
  const lead = Number(fields.lead.value);
  const members = [...fields.members.selectedOptions].map(item => item.value).filter(Boolean);
  if (members.includes(fields.model.value)) throw new Error("Choose distinct ensemble models.");
  if (!members.length) {
    const forecast = await perform("diffusion-forecast", {
      ...mapDomain(), model: fields.model.value, origin: new Date(fields.origin.value).toISOString(),
      configuration: await jsonFile(fields.configuration, true), lead, feature: fields.feature.value,
      diffusion: {iterations: Number(el("map-diffusion-iterations").value)}
    });
    drawMap(forecast.estimate, "trace-map", "Diffusion forecast map");
    el("trace-spread").hidden = true;
    showResult({lead_hours: lead, feature: forecast.feature, diffusion: forecast.diffusion});
    return;
  }
  const result = await perform("ensemble-map", {configuration: await jsonFile(fields.configuration, true),
    model: fields.model.value, members, origin: new Date(fields.origin.value).toISOString(),
    ...mapDomain(), steps: lead, lead, feature: fields.feature.value, adaptive: await jsonFile(fields.adaptive),
    predictive: await jsonFile(fields.predictive)});
  if (result.predictive) {
    if (result.adaptive) {
      const cells = {...result.adaptive, ...result.predictive};
      drawAdaptive(cells, "mean", "trace-map", "Adaptive predictive scenario mean");
      drawAdaptive(cells, "width", "trace-spread", "Adaptive predictive scenario interval width");
    } else {
      drawMap(result.predictive.mean, "trace-map", "Predictive scenario mean");
      drawMap(result.predictive.width, "trace-spread", "Predictive scenario interval width");
    }
  } else if (result.adaptive) {
    drawAdaptive(result.adaptive, "mean", "trace-map", "Adaptive ensemble forecast mean");
    drawAdaptive(result.adaptive, "spread", "trace-spread", "Adaptive ensemble model spread");
  } else {
    drawMap(result.layers.mean, "trace-map", "Ensemble forecast mean");
    drawMap(result.layers.spread, "trace-spread", "Ensemble model spread");
  }
  showResult({feature: result.feature, lead_hours: result.lead_hours, ...result.metadata});
}, "Ensemble forecast map available.");
submit("trace-map-form", async event => {
  const fields = event.target.elements;
  const composite = event.submitter.value === "composite";
  const clock = composite ? {} : {timestamp: new Date(fields.timestamp.value).toISOString(),
    window_minutes: Number(fields.window_minutes.value)};
  const result = await perform(composite ? "session-map" : "map", {
    ...mapDomain(), ...clock, feature: fields.feature.value,
    diffusion: {iterations: Number(el("map-diffusion-iterations").value)}
  });
  drawMap(result.estimate, "trace-map", composite ? "Non-simultaneous session composite" : "Observational map");
  el("trace-spread").hidden = true;
  const summary = {...result};
  delete summary.estimate;
  showResult(summary);
}, "Observational map available.");
el("trace-surveys").addEventListener("change", operation(async event => {
  trace.id = event.target.value;
  trace.survey = trace.receiver = null;
  trace.readiness = null;
  el("trace-result").hidden = el("trace-map").hidden = el("trace-spread").hidden = true;
}));
async function setup() {
  const [ports, jobs] = await Promise.all([request("/api/ports"), request("/api/jobs"), listSurveys()]);
  setOptions(el("trace-ports"), ports.map(value => [value, value]), "Select an enabled port");
  const models = jobs.filter(job => job.status === "complete" &&
    ["model.json", "direct-model.json", "quantile-model.json", "blend.json", "models/classical_ga.json"]
      .some(name => job.artifacts.includes(name)))
    .map(job => [job.id, job.label || job.id]);
  setOptions(el("trace-models"), models, "Check data without a model");
  setOptions(el("trace-ensemble-model"), models, "Select a saved model");
  el("trace-ensemble-members").replaceChildren(...models.map(([value, name]) => option(value, name)));
  render();
}
el("trace-protocol").addEventListener("change", event => {
  if (event.target.value === "aquamap_rx_csv") el("trace-connect").elements.baud.value = "115200";
});
setup().catch(error => { el("trace-alert").textContent = error.message; });
setInterval(() => { if (!trace.busy && trace.id) refresh().catch(() => {}); }, 1200);
