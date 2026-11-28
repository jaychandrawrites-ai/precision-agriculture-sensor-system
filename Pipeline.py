"""Independent commands for historical training, field forecasting and measured-site mapping."""
import argparse
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from Acquisition.Calibration import LeafCalibration, SoilCalibration
from Acquisition.FieldFeatures import field_features
from Acquisition.Hourly import session_hourly
from Acquisition.Health import session_health
from Acquisition.SpatialObservations import spatial_observations
from assumptions import Assumptions
from azmet import prepare
from config import FEATURES, SpatialConfig, TrainingConfig
from Data.ForecastHistory import inspect_history, require_history
from Data.QualityReport import quality_report
from Data.TimeSeries import calendar_cutoffs
from Data.WeatherPhysics import ReferenceET
from Framework.Backend import create_backend
from Spatial.MonteCarlo import SamplingConfig
from Spatial.SpatialGrid import SpatialGrid, local_coordinates
from Spatial.Validation import score_snapshot
from SpatiotemporalModel import SpatiotemporalModel
from Temporal.Quantile import quantile_columns
from train import train_values
from Utils.Diagnostics import forecast_diagnostics
from Utils.ModelLoader import load_forecaster
from Utils.Serialization import read_json, write_arrays, write_json


def train_historical(source, configuration, checkpoint, settings=TrainingConfig()):
    features, flags, report = prepare(source, configuration)
    timestamps = features.index.to_numpy(dtype="datetime64[ns]")
    cutoffs = calendar_cutoffs(timestamps, settings.train_fraction, settings.validation_fraction)
    quality = quality_report(features, flags, settings, cutoffs=cutoffs)
    provenance = {**report["provenance"], "chronological_split": {
        "method": "calendar_before_target_filtering",
        "cutoffs_utc": [str(cutoff) + "Z" for cutoff in cutoffs],
    }}
    result = train_values(features.to_numpy(), timestamps, checkpoint, settings,
                          provenance=provenance, cutoffs=cutoffs)
    return {"training": result, "quality": quality, "provenance": provenance}


def _short_session_context(configuration, weather, site, reference):
    """Accept explicit, adjacent hourly context with matched physics metadata."""
    from dataclasses import asdict

    context = configuration.get("historical_context")
    if context is None:
        return None
    source = context.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Historical context requires a source label")
    if context.get("assumptions") != site.metadata() or context.get("reference_et") != asdict(reference):
        raise ValueError("Historical context physics differs from the field configuration")
    rows = context.get("rows", [])
    if not rows or len(rows) > 720:
        raise ValueError("Supply 1–720 historical hourly context rows")
    times = [pd.Timestamp(row["timestamp"]) for row in rows]
    if any(pd.isna(time) or time.tzinfo is None for time in times):
        raise ValueError("Historical context timestamps must include UTC offsets")
    index = pd.DatetimeIndex([time.tz_convert("UTC") for time in times], name="timestamp")
    expected = pd.date_range(end=weather.values.index[0] - pd.offsets.Hour(),
                             periods=len(index), freq="h", name="timestamp")
    if not index.equals(expected):
        raise ValueError("Historical context must be consecutive and end one hour before the first field bin")
    values = np.asarray([[row.get(feature, np.nan) for feature in FEATURES] for row in rows], dtype=float)
    if not np.isfinite(values[:, 0]).all():
        raise ValueError("Historical context needs finite air temperatures")
    return pd.DataFrame(values, index=index, columns=FEATURES)


def prepare_session(readings, configuration):
    site = Assumptions.from_dict(configuration.get("assumptions", {}))
    reference = ReferenceET(site.site.latitude_deg, **configuration["reference_et"])
    leaf = LeafCalibration(**configuration["leaf_calibration"])
    soil = SoilCalibration(**configuration["soil_calibration"])
    weather, gps, coverage = session_hourly(readings, leaf, soil, **configuration.get("hourly", {}))
    context = _short_session_context(configuration, weather, site, reference)
    prior = () if context is None else context["air_temperature_c"].items()
    features, flags, provenance = field_features(weather, site, reference, prior_temperature=prior)
    if context is not None:
        features = pd.concat([context, features])
        flags = flags.reindex(features.index, fill_value=False)
        flags["historical_context"] = features.index.isin(context.index)
        provenance["historical_context"] = {
            "source": configuration["historical_context"]["source"], "hours": len(context),
            "end_utc": context.index[-1].isoformat(),
            "interpretation": "Explicit adjacent historical features followed by locally aggregated field readings",
        }
    provenance["model_input_interval_hours"] = 1
    return features, flags, provenance, gps, coverage


def forecast_session(readings, configuration, checkpoint, steps, *, device="cpu"):
    return forecast_prepared_session(readings, configuration, load_forecaster(checkpoint, device), steps)


def forecast_prepared_session(readings, configuration, forecaster, steps):
    return forecast_field(prepare_session(readings, configuration), forecaster, steps)


def forecast_field(prepared, forecaster, steps):
    """Forecast from one prepared field history without reopening its readings."""
    features, flags, provenance, gps, coverage = prepared
    if not provenance["complete_hour_bins_only"]:
        raise ValueError("Forecasting requires closed hourly bins")
    verified = forecaster.require_compatible_field(provenance)
    hours = forecaster.history_hours
    history = require_history(features.to_numpy(), features.index.to_numpy(dtype="datetime64[ns]"),
                              f"Latest {hours} hours need all six valid features and sufficient ET history",
                              hours=hours)
    prediction, quantiles = forecaster.forecast_bundle(history, steps)
    future = pd.date_range(features.index[-1] + pd.offsets.Hour(), periods=steps, freq="h",
                           name="timestamp")
    forecast = pd.DataFrame(prediction, index=future, columns=FEATURES)
    report = {"checkpoint_sha256": forecaster.checkpoint_sha256,
              "checkpoint_physics_checked": verified,
              "field_provenance": provenance,
              "quality": quality_report(features, flags),
              "forecast": forecast_diagnostics(history, prediction, forecaster.targets)}
    if quantiles is not None:
        report["quantiles"] = {
            "levels": forecaster.quantile.levels.tolist(),
            "features": list(FEATURES),
            "values": quantiles.tolist(),
            "axes": ["lead_hour", "feature", "quantile"],
            "interpretation": "trained quantiles without post-training calibration",
        }
    return forecast, report, gps, coverage


def check_session(readings, configuration, checkpoint=None):
    """Check a recorded session without training or emitting forecasts or GPS."""
    prepared = prepare_session(readings, configuration)
    forecaster = None if checkpoint is None else load_forecaster(checkpoint)
    return check_prepared_session(prepared, forecaster)


def check_prepared_session(prepared, forecaster=None):
    """Check an already prepared field history against an optional loaded model."""
    features, flags, provenance, _, _ = prepared
    hours = forecaster.history_hours if forecaster is not None else 3
    result = inspect_history(features.to_numpy(), features.index.to_numpy(dtype="datetime64[ns]"), hours=hours)
    history_hours = provenance["reference_et"]["history_hours"]
    result.update({
        "closed_hour_bins": bool(provenance["complete_hour_bins_only"]),
        "ready": result["ready"] and bool(provenance["complete_hour_bins_only"]),
        "minimum_temperature_history_hours": history_hours + result["required_hours"] - 1,
        "historical_context": provenance.get("historical_context"),
        "latest_hour_utc": features.index[-1].isoformat() if len(features) else None,
        "reference_et_unavailable_hours": int(flags["reference_et_unavailable"].sum()),
    })
    if forecaster is not None:
        result["checkpoint_sha256"] = forecaster.checkpoint_sha256
        try:
            verified = forecaster.require_compatible_field(provenance)
        except ValueError as error:
            result.update(ready=False, checkpoint_physics="incompatible", checkpoint_reason=str(error))
        else:
            result["checkpoint_physics"] = "matched" if verified else "metadata unavailable"
    else:
        result["checkpoint_physics"] = "unchecked"
    return result


def map_measurements(source, configuration, output):
    """Read one explicitly timestamped field snapshot, including only measured labels."""
    field_name = configuration["feature"]
    timestamp = pd.Timestamp(configuration["timestamp"])
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Spatial snapshot needs an explicit UTC offset")
    if field_name in {"timestamp", "observed_at", "latitude_deg", "longitude_deg", "device_id"}:
        raise ValueError("Spatial feature must be a measured quantity, separate from coordinates")
    payload = Path(source).read_bytes()
    frame = pd.read_csv(BytesIO(payload))
    source_times = frame["timestamp"].astype(str)
    if not source_times.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$").all():
        raise ValueError("Field measurement timestamps need an explicit UTC offset")
    mask = pd.to_datetime(source_times, format="ISO8601", utc=True) == timestamp
    snapshot = frame.loc[mask]
    predicted_source = False
    if "value_origin" in snapshot:
        predicted_rows = snapshot["value_origin"].eq("model_forecast")
        if predicted_rows.any():
            if not configuration.get("allow_forecast_snapshot", False) or not predicted_rows.all():
                raise ValueError("Enable forecast-snapshot mapping explicitly and keep it separate from observations")
            predicted_source = True
    held_out = None
    if role := configuration.get("split_column"):
        if predicted_source:
            raise ValueError("Held-out spatial validation requires independent measurements")
        labels = snapshot[role]
        if not labels.isin(("fit", "test")).all() or not labels.eq("test").any():
            raise ValueError("A spatial split requires fit and test labels with held-out rows")
        held_out = snapshot.loc[labels.eq("test")]
        snapshot = snapshot.loc[labels.eq("fit")]
    grid = SpatialGrid(configuration["east_m"], configuration["north_m"])
    backend = create_backend(dtype="float64")
    settings = SpatialConfig(**configuration.get("solver", {}))
    sampling = SamplingConfig(**configuration["sampling"]) if "sampling" in configuration else None
    result = SpatiotemporalModel().reconstruct(
        grid, snapshot[["latitude_deg", "longitude_deg"]].to_numpy(),
        snapshot[field_name].to_numpy(dtype=float), configuration["origin_lat_lon"], backend,
        settings, kriging_range_m=configuration.get("kriging_range_m"), sampling=sampling,
        hard_seeds=configuration.get("hard_seeds", False),
        boundary_values=configuration.get("boundary_values"),
        boundary_weight=configuration.get("boundary_weight"),
        boundary=configuration.get("laplacian_boundary", "neumann"),
        algorithm=configuration.get("algorithm", "descent"),
        robust=configuration.get("robust"),
    )
    arrays = {"east_m": grid.east, "north_m": grid.north, "variational": result.field,
              "idw": result.idw}
    if result.kriging is not None:
        arrays["kriging"] = result.kriging
    report = {"measurement_count": 0 if predicted_source else len(snapshot),
              "forecast_seed_count": len(snapshot) if predicted_source else 0, "feature": field_name,
              "source_sha256": sha256(payload).hexdigest(),
              "algorithm": configuration.get("algorithm", "descent"),
              "timestamp": timestamp.isoformat(), "converged": result.solver.converged,
              "iterations": result.solver.iterations, "gradient_rms": result.solver.gradient_rms,
              "initial_energy": result.solver.energies[0], "final_energy": result.solver.energies[-1]}
    if result.uncertainty is not None:
        for name in ("mean", "std", "lower", "upper", "split_rhat", "ess"):
            arrays[name] = getattr(result.uncertainty, name)
        report["sampling"] = {"acceptance": result.sampling.acceptance.tolist(),
                              "burn_in_acceptance": result.sampling.burn_in_acceptance.tolist(),
                              "parameters": configuration["sampling"],
                              "proposal_scales": result.sampling.proposal_scales.tolist(),
                              "conditional_on_energy_model": True}
    if held_out is not None:
        coordinate_columns = ["latitude_deg", "longitude_deg"]
        origin = configuration["origin_lat_lon"]
        sites = local_coordinates(snapshot[coordinate_columns].to_numpy(), origin)
        queries = local_coordinates(held_out[coordinate_columns].to_numpy(), origin)
        report["held_out_count"] = len(held_out)
        report["held_out"] = score_snapshot(
            grid, result.field, sites, snapshot[field_name].to_numpy(dtype=float),
            queries, held_out[field_name].to_numpy(dtype=float),
            configuration.get("kriging_range_m"), result.sampling,
        )
    adaptive = configuration.get("adaptive")
    if adaptive is not None:
        from adaptive_map import adaptive_kriging

        if not isinstance(adaptive, dict):
            raise ValueError("Adaptive map settings must be an object")
        adaptive_settings = {"bounds_m": [grid.east[0], grid.north[0], grid.east[-1], grid.north[-1]],
                             **adaptive}
        sites = local_coordinates(snapshot[["latitude_deg", "longitude_deg"]].to_numpy(),
                                  configuration["origin_lat_lon"])
        variance_key = adaptive.get("measurement_variance_column")
        refined = adaptive_kriging(sites, snapshot[field_name], adaptive_settings,
                                   variance=snapshot[variance_key] if variance_key else None)
        report["adaptive"] = {
            "artifact": "adaptive.json", "training_sites": len(sites), "leaf_count": refined["leaf_count"],
            "source_kind": "model forecasts" if predicted_source else "measurements",
        }
    output = Path(output)
    output.mkdir()
    write_arrays(output / "field.npz", **arrays)
    if adaptive is not None:
        write_json(output / "adaptive.json", refined)
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    health = commands.add_parser("session-health")
    health.add_argument("source", type=Path)
    health.add_argument("output", type=Path)
    for name in ("train-history", "prepare-session", "forecast-session", "check-session", "extract-spatial", "map"):
        sub = commands.add_parser(name)
        sub.add_argument("source", type=Path)
        sub.add_argument("configuration", type=Path)
        sub.add_argument("output", type=Path)
        if name == "extract-spatial":
            sub.add_argument("--also", type=Path, action="append", default=[])
        if name == "train-history":
            sub.add_argument("--epochs", type=int, default=TrainingConfig().epochs)
        if name == "check-session":
            sub.add_argument("--checkpoint", type=Path)
        if name == "forecast-session":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    if args.command == "session-health":
        write_json(args.output, session_health(args.source))
        return
    configuration = read_json(args.configuration)
    if args.command == "check-session":
        write_json(args.output, check_session(args.source, configuration, args.checkpoint))
    elif args.command == "train-history":
        report = train_historical(args.source, configuration, args.output,
                                  replace(TrainingConfig(), epochs=args.epochs))
        write_json(args.output.with_suffix(args.output.suffix + ".report.json"), report)
    elif args.command == "extract-spatial":
        frame, report = spatial_observations([args.source, *args.also], **configuration)
        args.output.mkdir()
        frame.to_csv(args.output / "measurements.csv", index=False)
        write_json(args.output / "report.json", report)
    elif args.command == "map":
        map_measurements(args.source, configuration, args.output)
    else:
        if args.command == "prepare-session":
            frame, flags, provenance, gps, coverage = prepare_session(args.source, configuration)
            report = {"provenance": provenance, "quality": quality_report(frame, flags)}
            name = "hourly.csv"
        else:
            frame, report, gps, coverage = forecast_session(args.source, configuration, args.checkpoint,
                                                            args.steps)
            if quantiles := report.get("quantiles"):
                tail = pd.DataFrame(
                    [sum(lead, []) for lead in quantiles["values"]],
                    index=frame.index, columns=quantile_columns(quantiles["levels"]),
                )
                frame = pd.concat((frame, tail), axis=1)
            name = "forecast.csv"
        args.output.mkdir()
        frame.to_csv(args.output / name)
        gps.to_csv(args.output / "gps.csv", index=False)
        coverage.to_csv(args.output / "coverage.csv")
        if name == "hourly.csv":
            flags.to_csv(args.output / "flags.csv")
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
