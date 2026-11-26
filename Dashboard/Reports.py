"""Read dashboard views from saved scientific artifacts."""
import csv
import io
import json
import math

import numpy as np
import pandas as pd

from Data.StationAnalysis import seasonal_profile
from Utils.Serialization import read_json
from config import FEATURES


SENSORS = (*FEATURES, "soil_moisture_pct", "soil_temperature_c", "soil_ec_us_cm",
           "soil_ph", "leaf_mv")
METHODS = {
    "persistence": "Persistence",
    "seasonal_persistence": "Seasonal persistence",
    "recursive": "Recursive",
    "direct": "Direct",
    "hybrid": "Hybrid",
    "fixed": "Fixed architecture",
    "classical_ga": "Classical GA",
    "quantum_inspired": "Classical amplitude search",
    "random": "Random search",
}


def _optional(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def quantile_rows(report):
    """Keep probabilities and marginal calibration aligned with their original leads."""
    levels, test = report["levels"], report["test"]
    rows = []
    for lead, lead_scores in enumerate(test["pinball"]):
        for index, feature in enumerate(FEATURES):
            rows.append({"method": "quantile_interval", "label": "Outermost quantile interval",
                         "lead_hours": lead + 1, "feature": feature, "quantile_level": None,
                         "coverage": _optional(test["coverage"][lead][index]),
                         "mean_width": _optional(test["mean_width"][lead][index])})
            for level_index, level in enumerate(levels):
                rows.append({"method": f"quantile_{level}", "label": f"Quantile {level:g}",
                             "lead_hours": lead + 1, "feature": feature, "quantile_level": level,
                             "pinball": _optional(lead_scores[index][level_index]),
                             "fraction_below": _optional(test["fraction_below"][lead][index][level_index])})
    return rows


def metric_rows(report, kind):
    """Flatten saved metrics without modifying scores or evaluation groups."""
    if kind == "quantile_train":
        return quantile_rows(report)
    if kind == "benchmark":
        methods = report["metrics"]
    elif kind == "search":
        methods = {key: item["test_rollout"] for key, item in report.items()
                   if isinstance(item, dict) and "test_rollout" in item}
        methods.update({name: item for name, item in report.get("baselines", {}).items()
                        if name in {"persistence", "seasonal_persistence"}})
    else:
        methods = {"recursive": report.get("test_rollout", [])}
    records = []
    for method, scores in methods.items():
        intervals = report.get("intervals", {}).get(method, {})
        for offset, lead in enumerate(scores):
            for index, feature in enumerate(FEATURES):
                if feature not in lead:
                    continue
                result = lead[feature]
                row = {"method": method, "label": METHODS.get(method, method),
                       "lead_hours": offset + 1, "feature": feature,
                       **{name: _optional(result.get(name)) for name in
                          ("rmse", "mae", "bias", "r2", "persistence_skill", "seasonal_skill")}}
                for name in ("coverage", "mean_width"):
                    if name in intervals:
                        row[name] = _optional(intervals[name][offset][index])
                    else:
                        row[name] = None
                records.append(row)
    return records


def metrics_csv(path, kind):
    rows = metric_rows(read_json(path), kind)
    output = io.StringIO(newline="")
    fields = ("method", "label", "lead_hours", "feature", "rmse", "mae", "bias", "r2",
              "persistence_skill", "seasonal_skill", "coverage", "mean_width", "quantile_level",
              "pinball", "fraction_below")
    writer = csv.DictWriter(output, fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def research_report(path, kind):
    report = read_json(path)
    result = {"metrics": metric_rows(report, kind)}
    if kind == "quantile_train":
        result["calibration"] = {
            "levels": report["levels"],
            "validation_pinball": report["validation_pinball"],
            "selection_epoch": report["selection_epoch"],
            "test_origins": report["test"]["origins"],
            "nominal_interval_mass": report["test"]["nominal_interval_mass"],
            "simultaneous_coverage": report["test"]["simultaneous_coverage"],
            "outcomes": "Held-out diagnostic outcomes; no post-training coverage calibration was applied.",
        }
    if kind == "benchmark":
        result.update({"manifest": report.get("manifest", {}),
                       "windows": report.get("windows", {}),
                       "reliability": report.get("reliability", {})})
    if kind == "search":
        methods = {method: details for method, details in report.items()
                   if isinstance(details, dict) and "candidate_count" in details}
        result["search"] = {name: {key: details[key] for key in
                                  ("parameters", "candidate_count", "candidate_model_sample_steps",
                                   "candidate_parameter_sample_steps", "generations",
                                   "validation_pareto_front", "winning_bits", "hidden", "skips")
                                  if key in details}
                            for name, details in methods.items()}
        result["protocol"] = report.get("protocol", {})
    return result


def dataset_report(hourly, report):
    frame = pd.read_csv(hourly)
    frame.index = pd.to_datetime(frame.pop("timestamp"), utc=True, format="ISO8601")
    profile = seasonal_profile(frame)
    bounds = {}
    for feature in FEATURES:
        values = frame[feature].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            counts, edges = np.histogram(values, bins=16)
            bounds[feature] = {"bins": counts.tolist(), "edges": edges.tolist(),
                               "missing": len(frame) - len(values)}
    data = read_json(report)
    return {"quality": data["quality"], "provenance": data["provenance"], "seasonal": profile,
            "distributions": bounds}


def forecast_series(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        rows = list(reader)
    point = {name: [float(row[name]) for row in rows] for name in FEATURES}
    quantiles = {}
    for name in FEATURES:
        prefix = name + "_q"
        columns = sorted((float(key[len(prefix):]), key) for key in header if key.startswith(prefix))
        if columns:
            quantiles[name] = [{"level": level, "value": [float(row[key]) for row in rows]}
                               for level, key in columns]
    return {"timestamps": [row["timestamp"] for row in rows], "series": point, "quantiles": quantiles}


def sensor_series(path):
    """Preview a bounded packet tail without device names, GPS, or raw payloads."""
    rows = []
    if path.is_file():
        with path.open("rb") as stream:
            size = path.stat().st_size
            stream.seek(max(0, size - 65536))
            if size > 65536:
                stream.readline()
            for line in stream:
                try:
                    row = json.loads(line)
                    measurements = row["measurements"]
                    if isinstance(measurements, dict):
                        rows.append(measurements)
                except (ValueError, TypeError, KeyError):
                    continue
    series = {name: [_optional(row.get(name)) for row in rows[-120:]] for name in SENSORS}
    series = {name: values for name, values in series.items() if any(x is not None for x in values)}
    return {"series": series, "temperature_c": series.get("air_temperature_c", [])}
