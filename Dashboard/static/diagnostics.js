"use strict";

const selection = document.getElementById("diagnostic-run");
const status = document.getElementById("diagnostic-status");
const results = document.getElementById("diagnostic-results");
let runs = [];

function render() {
  results.replaceChildren();
  const run = runs.find(item => item.id === selection.value);
  if (!run) return;
  status.textContent = "Held-out results and saved model diagnostics";
  for (const artifact of run.artifacts.filter(name => name.startsWith("diagnostics/"))) {
    const name = artifact.split("/")[1];
    const url = "/api/artifacts/" + encodeURIComponent(run.id) + "/diagnostics/" + encodeURIComponent(name);
    const link = document.createElement("a");
    link.href = url;
    link.textContent = "Download " + name;
    if (name.endsWith(".png")) {
      const figure = document.createElement("figure");
      const caption = document.createElement("figcaption");
      const image = document.createElement("img");
      image.src = url;
      image.alt = name.replace(".png", "").replaceAll("-", " ");
      caption.append(link);
      figure.append(image, caption);
      results.append(figure);
    } else {
      const item = document.createElement("p");
      item.append(link);
      results.append(item);
    }
  }
}
selection.addEventListener("change", render);
fetch("/api/jobs").then(response => {
  if (!response.ok) throw new Error("Cannot load saved jobs");
  return response.json();
}).then(jobs => {
  runs = jobs.filter(job => job.artifacts.includes("diagnostics/metrics.json"));
  selection.replaceChildren(...runs.map(run => new Option(new Date(run.created_at).toLocaleString(), run.id)));
  if (runs.length) render();
  else status.textContent = "Complete a Model training job to generate diagnostics.";
}).catch(error => { status.textContent = error.message; });
