"""Independent commands for historical training, field forecasting and measured-site mapping."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from Acquisition.Calibration import LeafCalibration, SoilCalibration
from Acquisition.FieldFeatures import field_features
from Acquisition.Hourly import session_hourly
from assumptions import Assumptions
from azmet import prepare
from config import FEATURES, SpatialConfig, TrainingConfig
from Data.DataQuality import invalid_rows
from Data.QualityReport import quality_report
from Data.WeatherPhysics import ReferenceET
from Framework.Backend import create_backend
from Spatial.MonteCarlo import SamplingConfig
from Spatial.SpatialGrid import SpatialGrid
from SpatiotemporalModel import SpatiotemporalModel
from train import train_values
from Utils.Diagnostics import forecast_diagnostics
from Utils.ModelLoader import load_forecaster
from Utils.Serialization import read_json, write_arrays, write_json


def train_historical(source, configuration, checkpoint, settings=TrainingConfig()):
    features, flags, report = prepare(source, configuration)
    quality = quality_report(features, flags, settings)
    result = train_values(features.to_numpy(), features.index.to_numpy(dtype="datetime64[ns]"),
                          checkpoint, settings, provenance=report["provenance"])
    return {"training": result, "quality": quality, "provenance": report["provenance"]}


def prepare_session(readings, configuration):
    site = Assumptions.from_dict(configuration.get("assumptions", {}))
    reference = ReferenceET(site.site.latitude_deg, **configuration["reference_et"])
    leaf = LeafCalibration(**configuration["leaf_calibration"])
    soil = SoilCalibration(**configuration["soil_calibration"])
    weather, gps, coverage = session_hourly(readings, leaf, soil, **configuration.get("hourly", {}))
    features, flags, provenance = field_features(weather, site, reference)
    return features, flags, provenance, gps, coverage


def forecast_session(readings, configuration, checkpoint, steps, *, device="cpu"):
    features, flags, provenance, gps, coverage = prepare_session(readings, configuration)
    forecaster = load_forecaster(checkpoint, device)
    if not provenance["complete_hour_bins_only"]:
        raise ValueError("Forecasting requires closed hourly bins")
    verified = forecaster.require_compatible_field(provenance)
    history = features.iloc[-3:].to_numpy()
    if len(history) < 3 or invalid_rows(history).any():
        raise ValueError("Latest three hours need all six valid features and sufficient ET history")
    prediction = SpatiotemporalModel(forecaster).forecast(history, steps)
    future = pd.date_range(features.index[-1] + pd.offsets.Hour(), periods=steps, freq="h",
                           name="timestamp")
    forecast = pd.DataFrame(prediction, index=future, columns=FEATURES)
    report = {"checkpoint_sha256": forecaster.checkpoint_sha256,
              "checkpoint_physics_checked": verified,
              "field_provenance": provenance,
              "quality": quality_report(features, flags),
              "forecast": forecast_diagnostics(history, prediction, forecaster.targets)}
    return forecast, report, gps, coverage


def map_measurements(source, configuration, output):
    """Read one explicitly timestamped field snapshot, including only measured labels."""
    field_name = configuration["feature"]
    timestamp = pd.Timestamp(configuration["timestamp"])
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Spatial snapshot needs an explicit UTC offset")
    frame = pd.read_csv(source)
    source_times = frame["timestamp"].astype(str)
    if not source_times.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$").all():
        raise ValueError("Field measurement timestamps need an explicit UTC offset")
    mask = pd.to_datetime(source_times, format="ISO8601", utc=True) == timestamp
    snapshot = frame.loc[mask]
    grid = SpatialGrid(configuration["east_m"], configuration["north_m"])
    backend = create_backend(dtype="float64")
    settings = SpatialConfig(**configuration.get("solver", {}))
    sampling = SamplingConfig(**configuration["sampling"]) if "sampling" in configuration else None
    result = SpatiotemporalModel().reconstruct(
        grid, snapshot[["latitude_deg", "longitude_deg"]].to_numpy(),
        snapshot[field_name].to_numpy(dtype=float), configuration["origin_lat_lon"], backend,
        settings, kriging_range_m=configuration.get("kriging_range_m"), sampling=sampling,
        hard_seeds=configuration.get("hard_seeds", False),
    )
    arrays = {"east_m": grid.east, "north_m": grid.north, "variational": result.field,
              "idw": result.idw}
    if result.kriging is not None:
        arrays["kriging"] = result.kriging
    report = {"measurement_count": len(snapshot), "feature": field_name,
              "timestamp": timestamp.isoformat(), "converged": result.solver.converged,
              "iterations": result.solver.iterations, "gradient_rms": result.solver.gradient_rms,
              "initial_energy": result.solver.energies[0], "final_energy": result.solver.energies[-1]}
    if result.uncertainty is not None:
        for name in ("mean", "std", "lower", "upper", "split_rhat"):
            arrays[name] = getattr(result.uncertainty, name)
        report["sampling"] = {"acceptance": result.sampling.acceptance.tolist(),
                              "burn_in_acceptance": result.sampling.burn_in_acceptance.tolist(),
                              "parameters": configuration["sampling"],
                              "conditional_on_energy_model": True}
    output = Path(output)
    output.mkdir()
    write_arrays(output / "field.npz", **arrays)
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train-history", "prepare-session", "forecast-session", "map"):
        sub = commands.add_parser(name)
        sub.add_argument("source", type=Path)
        sub.add_argument("configuration", type=Path)
        sub.add_argument("output", type=Path)
        if name == "train-history":
            sub.add_argument("--epochs", type=int, default=TrainingConfig().epochs)
        if name == "forecast-session":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    configuration = read_json(args.configuration)
    if args.command == "train-history":
        report = train_historical(args.source, configuration, args.output,
                                  replace(TrainingConfig(), epochs=args.epochs))
        write_json(args.output.with_suffix(args.output.suffix + ".report.json"), report)
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
