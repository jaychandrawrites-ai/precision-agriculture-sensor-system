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
from Pipeline import check_prepared_session, forecast_field, prepare_session
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
        if row.get(clock) is None or not start <= _aware(row[clock]) <= cutoff:
            continue
        identities.add(row.get("device_id"))
        if row.get("boot_id") is not None:
            boots.add(row["boot_id"])
        selected.append(row)
    if len(identities) > 1 or len(boots) > 1:
        raise ValueError("Segment contains separate device histories; select a single compatible device")
    return selected, config


def survey_forecasts(store, survey_id, configuration, checkpoint=None, steps=None, *, ensemble=()):
    """Return eligible closed segments without merging locations, revisits or device histories.

    By default this only checks readiness. A checkpoint and a positive horizon
    request predictions for eligible segments. Times remain independent across
    locations, so the result is not a spatial forecast at a common valid time.
    """
    if not isinstance(configuration, dict):
        raise ValueError("A field configuration object is required")
    if steps is not None and (checkpoint is None or type(steps) is not int or not 1 <= steps <= 168):
        raise ValueError("Forecasting requires a saved checkpoint and 1–168 hourly steps")
    forecaster = load_forecaster(checkpoint) if checkpoint is not None else None
    members = (forecaster, *(load_forecaster(path) for path in ensemble)) if forecaster is not None else ()
    hashes = [member.checkpoint_sha256 for member in members]
    if ensemble and (steps is None or len(set(hashes)) != len(members)):
        raise ValueError("An ensemble needs forecasting steps and distinct checkpoints")
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
                    prepared = prepare_session(path, config)
                    detail.update(check_prepared_session(prepared, forecaster))
                    if steps is not None and detail["ready"]:
                        forecast, diagnostics, _, _ = forecast_field(prepared, forecaster, steps)
                        detail["forecast"] = [
                            {"timestamp": timestamp.isoformat(), **dict(zip(FEATURES, map(float, values)))}
                            for timestamp, values in forecast.iterrows()
                        ]
                        detail["checkpoint_sha256"] = diagnostics["checkpoint_sha256"]
                        if "quantiles" in diagnostics:
                            detail["quantiles"] = diagnostics["quantiles"]
                        if ensemble:
                            # Evaluate the same ordered checkpoints at every location and lead.
                            draws = [forecast.to_numpy(dtype=float)]
                            for member in members[1:]:
                                prediction, _, _, _ = forecast_field(prepared, member, steps)
                                draws.append(prediction.to_numpy(dtype=float))
                            detail["member_forecasts"] = np.stack(draws).tolist()
                except (ValueError, TypeError, KeyError) as error:
                    detail.pop("forecast", None)
                    detail.update(ready=False, reason=str(error))
            report["locations"].append({"id": location["id"], "label": location["label"],
                                        "ready": any(item["ready"] for item in segments),
                                        "segments": segments})
    if ensemble:
        report["member_checkpoint_sha256"] = hashes
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
        source = "manual_csv" if "manual_csv" in flags else "recorded_gps"
        return values["latitude_deg"], values["longitude_deg"], source
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
                if record.get(clock) is None or "duplicate_packet" in flags or "previous_boot_packet" in flags:
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
    raster = _raster(sites, samples, grid_size)
    return {"survey_id": survey_id, "feature": feature, "provenance": "measured_window_composite",
            "unit": next(iter(FIELDS[feature][2])), "interpolation": "IDW", "clock": clock,
            "window_start_exclusive": start.isoformat(),
            "window_end_inclusive": end.isoformat(), "excluded_locations": len(locations) - len(sites),
            "samples": samples, **raster}

def _raster(sites, samples, grid_size):
    if len(set(sites)) < 2:
        raise ValueError("At least two distinct locations with values and coordinates are required")
    grid, points = _forecast_grid(sites, grid_size)
    estimate = idw(points, [sample["value"] for sample in samples], grid.points()).reshape(grid.shape)
    return {"east_m": grid.east.tolist(), "north_m": grid.north.tolist(), "estimate": estimate.tolist()}


def _forecast_grid(sites, grid_size):
    points = local_coordinates(sites, sites[0])
    limits = points.min(axis=0), points.max(axis=0)
    padding = max(1.0, float(np.max(limits[1] - limits[0])) * 0.05)
    east, north = [np.linspace(low - padding, high + padding, grid_size) for low, high in zip(*limits)]
    return SpatialGrid(east, north), points


def forecast_map(store, survey_id, report, feature, valid_at, origin, *, gps_minutes=60, grid_size=25):
    sites, samples, issued, valid, lead, excluded = _forecast_sites(
        store, survey_id, report, feature, valid_at, origin, gps_minutes, grid_size
    )
    return {"survey_id": survey_id, "feature": feature, "forecast_origin": issued.isoformat(),
            "valid_at": valid.isoformat(), "lead_hours": int(lead), "interpolation": "IDW",
            "provenance": "point_model_forecasts", "uncertainty": "not estimated",
            "excluded_locations": excluded, "samples": samples,
            **_raster(sites, samples, grid_size)}


def _forecast_sites(store, survey_id, report, feature, valid_at, origin, gps_minutes, grid_size):
    """Map point forecasts at one issued and valid time using GPS known at issue."""
    if (report.get("survey_id") != survey_id or feature not in FEATURES
            or type(gps_minutes) not in (int, float) or not 0 < gps_minutes <= 1440
            or type(grid_size) is not int or not 2 <= grid_size <= 100):
        raise ValueError("Choose a single survey, forecast feature and valid map settings")
    issued, valid = _aware(origin), _aware(valid_at)
    lead = (valid - issued).total_seconds() / 3600
    if lead < 1 or not lead.is_integer():
        raise ValueError("Forecast validity must be a positive whole-hour lead")
    locations = {item["id"]: item for item in store.status(survey_id)["locations"]}
    points, samples = [], []
    for details in report["locations"]:
        location = locations[details["id"]]
        candidates = []
        for segment in details["segments"]:
            forecasts = segment.get("forecast", [])
            if not forecasts or _aware(forecasts[0]["timestamp"]) != issued + timedelta(hours=1):
                continue
            supported = {item["id"] for item in location["segments"] if item["ended_at"] is not None}
            if segment["id"] not in supported:
                continue
            for prediction in forecasts:
                if _aware(prediction["timestamp"]) == valid:
                    candidates.append((segment, prediction))
        if not candidates:
            continue
        if len(candidates) > 1:
            raise ValueError("Multiple visits have the same issue time at a location; resolve the duplicate forecasts")
        segment, forecast = candidates[0]
        coordinate = None
        received_at = None
        for row in store.segment_readings(survey_id, segment["id"]):
            if {"duplicate_packet", "previous_boot_packet"}.intersection(row["flags"]):
                continue
            received = _aware(row["received_at"])
            observed = _aware(row["observed_at"]) if row.get("observed_at") else received
            if received > issued or not issued - timedelta(minutes=gps_minutes) < observed <= issued:
                continue
            fix = _coordinates(row["measurements"], location, row["flags"])
            if fix and fix[2] != "manual_location" and (received_at is None or received > received_at):
                coordinate, received_at = fix, received
        if coordinate is None:
            coordinate = _coordinates({}, location, ())
        value = forecast.get(feature)
        if coordinate is None or type(value) not in (int, float) or not math.isfinite(value):
            continue
        points.append(coordinate[:2])
        samples.append({"location_id": location["id"], "segment_id": segment["id"],
                        "coordinate_source": coordinate[2], "value": value,
                        "checkpoint_sha256": segment.get("checkpoint_sha256")})
    return points, samples, issued, valid, lead, len(locations) - len(samples)


def ensemble_forecast_maps(store, survey_id, report, origin, *, steps, probability=.9, grid_size=25, adaptive=None, predictive=None, physical_sources=None, draw_path=None):
    """Propagate shared model draws, keeping site, lead and feature dependencies."""
    from spatial_forecast import ensemble_maps, interpolation_weights

    hashes = report.get("member_checkpoint_sha256", [])
    if (len(hashes) < 2 or len(set(hashes)) != len(hashes)
            or type(steps) is not int or not 1 <= steps <= 72):
        raise ValueError("Provide at least two distinct ensemble members and 1–72 leads")
    issued = _aware(origin)
    sites, samples, *_ = _forecast_sites(store, survey_id, report, FEATURES[0],
                                         issued + timedelta(hours=1), issued, 60, grid_size)
    by_segment = {segment["id"]: segment for location in report["locations"]
                  for segment in location["segments"]}
    retained, points, ids, segments = [], [], [], []
    for site, sample in zip(sites, samples):
        segment = by_segment[sample["segment_id"]]
        values = np.asarray(segment.get("member_forecasts"), dtype=float)
        timestamps = [_aware(row["timestamp"]) for row in segment["forecast"][:steps]]
        if (values.ndim != 3 or values.shape[0] != len(hashes) or values.shape[1] < steps
                or values.shape[2] != len(FEATURES) or not np.isfinite(values).all()
                or timestamps != [issued + timedelta(hours=i) for i in range(1, steps + 1)]):
            continue
        points.append(site)
        ids.append(sample["location_id"])
        retained.append(values[:, :steps])
        segments.append(segment)
    if len(set(points)) != len(points) or len(points) < 2:
        raise ValueError("Forecast maps require distinct coordinates and at least two complete ensemble sites")
    grid, xy = _forecast_grid(points, grid_size)
    members = np.stack(retained, axis=1)  # member × site × lead × feature
    arrays = ensemble_maps(grid, xy, members.reshape(len(hashes), len(points), -1),
                           probability=probability)
    result = {name: value.reshape(*grid.shape, steps, len(FEATURES)).transpose(2, 0, 1, 3)
              for name, value in arrays.items()}
    result.update(east_m=grid.east, north_m=grid.north, site_member_forecasts=members,
                  interpolation_weights=interpolation_weights(xy, grid.points(), 2).reshape(*grid.shape, len(points)))
    metadata = {
        "survey_id": survey_id, "issued_at": issued.isoformat(), "features": FEATURES,
        "valid_at": [(issued + timedelta(hours=i)).isoformat() for i in range(1, steps + 1)],
        "sites": ids, "checkpoint_sha256": hashes, "central_mass": probability,
        "axes": ["lead", "north", "east", "feature"],
        "member_axes": ["member", "site", "lead", "feature"],
        "weight_axes": ["north", "east", "site"],
        "interpretation": "propagated empirical model disagreement; field coverage is unverified",
    }
    default_bounds = [grid.east[0], grid.north[0], grid.east[-1], grid.north[-1]]
    if adaptive is not None:
        from Spatial.EnsembleRefinement import adaptive_ensemble
        arrays, metadata["adaptive"] = adaptive_ensemble(
            xy, members, adaptive.get("bounds_m", default_bounds), adaptive, probability)
        result.update(arrays)
    if adaptive is not None and "variational" in adaptive:
        from Spatial.SurveyUncertainty import variational_trajectories

        options = {**adaptive, **adaptive["variational"]}
        options["refinement"] = adaptive["variational"].get("refinement", {})
        arrays, metadata["variational"] = variational_trajectories(
            xy, members, options.get("bounds_m", default_bounds), options, probability)
        result.update(arrays)
    if predictive is not None:
        from Spatial.SurveyUncertainty import predictive_fields

        settings = {key: adaptive[key] for key in ("field_polygon_m", "field_holes_m")
                    if adaptive is not None and key in adaptive}
        settings.update(predictive)
        provenance = [{"checkpoint_sha256": sha, "physical_scenario_id": None} for sha in hashes]
        physical = settings.pop("physical", None)
        if physical_sources is not None:
            if physical is not None:
                raise ValueError("Choose either saved or live-computed physical scenarios")
            physical = _survey_physical_scenarios(store, survey_id, ids, segments, issued, steps, physical_sources)
        if physical is not None:
            if (_aware(physical["issue_at"]) != issued or physical["sites"] != ids
                    or list(physical["features"]) != list(FEATURES)
                    or physical["checkpoint_sha256"] not in hashes):
                raise ValueError("Physical forecasts must match the survey site order, issue, features and checkpoint")
            forecasts = np.asarray(physical["forecasts"], dtype=float)
            draw_ids = physical["draw_ids"]
            if (forecasts.ndim != 4 or forecasts.shape[1:] != members.shape[1:]
                    or not len(forecasts) or len(draw_ids) != len(forecasts) or len(set(draw_ids)) != len(draw_ids)):
                raise ValueError("Invalid aligned physical forecast scenarios")
            settings["member_probabilities"] = [1 / len(hashes)] * len(hashes)
            settings["member_probabilities"][hashes.index(physical["checkpoint_sha256"])] = 0.
            settings["member_probabilities"].extend([1 / (len(hashes) * len(forecasts))] * len(forecasts))
            members = np.concatenate((members, forecasts))
            provenance.extend({"checkpoint_sha256": physical["checkpoint_sha256"],
                               "physical_scenario_id": key} for key in draw_ids)
        if "residuals" in settings:
            origins = [_aware(value) for value in settings["residual_origins"]]
            if (settings.get("residual_sites") != ids or len(origins) < 2
                    or any(b - a < timedelta(hours=steps) for a, b in zip(origins, origins[1:]))
                    or origins[-1] + timedelta(hours=steps) > issued):
                raise ValueError("Residual calibration must match sites and use nonoverlapping pre-issue trajectories")
        geometry = result.get("adaptive_centers_m", grid.points())
        operator = None
        if settings.get("method") in {"variational", "adaptive_variational"}:
            if "variational_weights" not in result:
                raise ValueError("Configure adaptive.variational to select a variational prediction")
            if settings["method"] == "variational":
                mesh = metadata["variational"]
                if mesh["adaptive"]:
                    raise ValueError("Fixed variational prediction requires equal initial_depth and max_depth")
            geometry = result["variational_centers_m"]
            operator = result["variational_weights"]
        arrays, manifest = predictive_fields(xy, members, geometry, settings, probability, operator=operator, draw_path=draw_path)
        result.update(arrays)
        result["predictive_source_trajectories"] = members
        result["predictive_site_xy_m"] = xy
        metadata["predictive"] = {**manifest, "members": provenance,
                                  "physical_provenance": physical.get("provenance") if physical else None}
    return result, metadata


def _survey_physical_scenarios(store, survey_id, ids, segments, issued, steps, sources):
    from Physics.ForecastScenarios import load_site, scenario_forecasts, utc_hour
    from physical_ensemble import ParameterRange

    issue = utc_hour(issued)
    forecaster = load_forecaster(sources["checkpoint"])
    spec = sources["scenarios"]
    config = deepcopy(sources["field_configuration"])
    config.setdefault("hourly", {})["as_of"] = issue.isoformat()
    parameters = [ParameterRange(**item) for item in spec["ranges"]]
    sites = {}
    with TemporaryDirectory(prefix="jupiter-trace-scenarios-") as directory:
        path = Path(directory) / "readings.jsonl"
        for location, segment in zip(ids, segments):
            records, field = _prepare_segment(store.segment_readings(survey_id, segment["id"]),
                                               config, segment["started_at"], segment["ended_at"])
            _write_readings(path, records)
            sites[location] = load_site(path, field, "field", issue)
        draws, _, report = scenario_forecasts(
            sites, forecaster, parameters, issue, steps=steps, draws=spec.get("draws", 16),
            seed=spec.get("seed", 42), latent_correlation=spec.get("latent_correlation"), perturb=False)
    return {"issue_at": report["issue_at"], "sites": ids, "features": report["features"],
            "checkpoint_sha256": forecaster.checkpoint_sha256, "draw_ids": report["draw_ids"],
            "forecasts": draws, "provenance": {key: report[key] for key in (
                "parameter_scenarios", "parameter_draws", "latent_correlation_assumption")}}
