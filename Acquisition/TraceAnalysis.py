"""Segment-isolated forecasting and bounded-time observational survey maps."""
from copy import deepcopy
from datetime import timedelta
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from Acquisition.Hourly import _aware
from Acquisition.MappedPackets import FIELDS
from config import FEATURES
from Pipeline import check_session, forecast_prepared_session
from Spatial.Interpolation import idw
from Spatial.SpatialGrid import SpatialGrid, local_coordinates
from Utils.ModelLoader import load_forecaster


EXCLUDED_MEASUREMENTS = {"latitude_deg", "longitude_deg", "gps_hdop", "gps_satellites"}
HISTORY_BOUNDARIES = {"device_restart", "receiver_reconnected", "receiver_disconnected",
                      "previous_boot_packet"}


def _write_readings(path, records):
    with path.open("w", encoding="utf-8") as output:
        for row in records:
            output.write(json.dumps(row, allow_nan=False) + "\n")


def _prepare_segment(records, configuration, started_at, ended_at):
    """Discard readings outside the closed recording and refuse mixed device histories."""
    config = deepcopy(configuration)
    hourly = config.setdefault("hourly", {})
    requested = hourly.get("as_of")
    start = _aware(started_at)
    cutoff = _aware(ended_at)
    if requested is not None:
        cutoff = min(cutoff, _aware(requested))
    hourly["as_of"] = cutoff.isoformat()
    clock = hourly.get("clock", "received_at")
    device = hourly.get("device_id")
    selected = []
    identities = set()
    boots = set()
    for row in records:
        if not start <= _aware(row["received_at"]) <= cutoff:
            continue
        if device is not None and row.get("device_id") != device:
            continue
        if HISTORY_BOUNDARIES.intersection(row["flags"]):
            raise ValueError("Segment includes a restart or receiver boundary; record a new segment")
        if not row["measurements"] or "duplicate_packet" in row["flags"]:
            continue
        if clock not in row or not start <= _aware(row[clock]) <= cutoff:
            continue
        identities.add(row.get("device_id"))
        if row.get("boot_id") is not None:
            boots.add(row["boot_id"])
        selected.append(row)
    if len(identities) > 1 or len(boots) > 1:
        raise ValueError("Segment contains separate device histories; select a single compatible device")
    return selected, config


def survey_forecasts(store, survey_id, configuration, checkpoint=None, steps=None):
    """Return eligible closed segments without merging locations, revisits or device histories.

    By default this only checks readiness. A checkpoint and a positive horizon
    request predictions for eligible segments. Times remain independent across
    locations, so the result is not a spatial forecast at a common valid time.
    """
    if not isinstance(configuration, dict):
        raise ValueError("A field configuration object is required")
    if steps is not None and (checkpoint is None or type(steps) is not int or not 1 <= steps <= 168):
        raise ValueError("Forecasting requires a saved checkpoint and 1–168 hourly steps")
    forecaster = load_forecaster(checkpoint) if steps is not None else None
    survey = store.status(survey_id)
    report = {"survey_id": survey_id, "scope": "independent closed segments",
              "locations": [], "forecast_steps": steps}
    with TemporaryDirectory(prefix="jupiter-trace-") as directory:
        path = Path(directory) / "readings.jsonl"
        for location in survey["locations"]:
            segments = []
            for segment in location["segments"]:
                detail = {"id": segment["id"], "ready": False,
                          "started_at": segment["started_at"], "ended_at": segment["ended_at"]}
                segments.append(detail)
                if segment["ended_at"] is None:
                    detail["reason"] = "Stop recording before checking segment readiness"
                    continue
                try:
                    records, config = _prepare_segment(
                        store.segment_readings(survey_id, segment["id"]), configuration,
                        segment["started_at"], segment["ended_at"],
                    )
                    if not records:
                        raise ValueError("No usable readings in the closed segment")
                    _write_readings(path, records)
                    detail.update(check_session(path, config, checkpoint))
                    if forecaster is not None and detail["ready"]:
                        forecast, diagnostics, _, _ = forecast_prepared_session(path, config, forecaster, steps)
                        detail["forecast"] = [
                            {"timestamp": timestamp.isoformat(), **dict(zip(FEATURES, map(float, values)))}
                            for timestamp, values in forecast.iterrows()
                        ]
                        detail["checkpoint_sha256"] = diagnostics["checkpoint_sha256"]
                except (ValueError, TypeError, KeyError) as error:
                    detail.update(ready=False, reason=str(error))
            report["locations"].append({"id": location["id"], "label": location["label"],
                                        "ready": any(item["ready"] for item in segments),
                                        "segments": segments})
    report["ready_locations"] = sum(item["ready"] for item in report["locations"])
    return report


def _measurement(value, field):
    if type(value) not in (int, float) or not math.isfinite(value):
        return False
    low, high, _ = FIELDS[field]
    return low <= value <= high


def _coordinates(values, location, flags):
    invalid = any(flag.startswith(("invalid:gps", "incomplete:gps")) for flag in flags)
    if not invalid and all(_measurement(values.get(key), key) for key in ("latitude_deg", "longitude_deg")):
        return values["latitude_deg"], values["longitude_deg"], "recorded_gps"
    if location.get("coordinate_source") == "manual" and all(
        _measurement(location.get(key), key) for key in ("latitude_deg", "longitude_deg")
    ):
        return location["latitude_deg"], location["longitude_deg"], "manual_location"
    return None


def observational_map(store, survey_id, feature, timestamp, window_minutes=60,
                      clock="received_at", grid_size=25):
    """Interpolate recent measured values with one eligible sample per location.

    The map is a bounded-window composite, not a simultaneous measurement.
    Manual coordinates are accepted for a location only with explicit
    provenance. Precise coordinates are not returned in this summary.
    """
    if (not isinstance(feature, str) or feature not in FIELDS or feature in EXCLUDED_MEASUREMENTS
            or clock not in ("received_at", "observed_at")
            or type(window_minutes) not in (int, float) or not 0 < window_minutes <= 1440
            or type(grid_size) is not int or not 2 <= grid_size <= 100):
        raise ValueError("Choose a measured sensor feature, a valid clock, and bounded map settings")
    end = _aware(timestamp)
    start = end - timedelta(minutes=window_minutes)
    locations = store.status(survey_id)["locations"]
    sites, samples = [], []
    for location in locations:
        best = None
        for segment in location["segments"]:
            for record in store.segment_readings(survey_id, segment["id"]):
                flags = record["flags"]
                if clock not in record or "duplicate_packet" in flags or "previous_boot_packet" in flags:
                    continue
                observed, received = _aware(record[clock]), _aware(record["received_at"])
                if not start < observed <= end or received > end:
                    continue
                values = record["measurements"]
                if not _measurement(values.get(feature), feature):
                    continue
                coordinates = _coordinates(values, location, flags)
                if coordinates is None:
                    continue
                candidate = (observed, received)
                if best is None or candidate > best[0]:
                    best = (candidate, coordinates, values[feature], segment["id"])
        if best is None:
            continue
        (observed, received), (lat, lon, source), value, segment_id = best
        sites.append((lat, lon))
        samples.append({"location_id": location["id"], "location_label": location["label"],
                        "segment_id": segment_id, "sample_at": observed.isoformat(),
                        "received_at": received.isoformat(), "coordinate_source": source,
                        "value": value})
    if len(set(sites)) < 2:
        raise ValueError("At least two distinct locations with measurements and coordinates are needed in the window")
    points = local_coordinates(sites, sites[0])
    limits = points.min(axis=0), points.max(axis=0)
    padding = max(1.0, float(np.max(limits[1] - limits[0])) * 0.05)
    east, north = [np.linspace(low - padding, high + padding, grid_size) for low, high in zip(*limits)]
    grid = SpatialGrid(east, north)
    estimate = idw(points, [sample["value"] for sample in samples], grid.points()).reshape(grid.shape)
    return {"survey_id": survey_id, "feature": feature, "provenance": "measured_window_composite",
            "unit": next(iter(FIELDS[feature][2])), "interpolation": "IDW", "clock": clock,
            "window_start_exclusive": start.isoformat(),
            "window_end_inclusive": end.isoformat(), "excluded_locations": len(locations) - len(sites),
            "samples": samples, "east_m": east.tolist(), "north_m": north.tolist(),
            "estimate": estimate.tolist()}
