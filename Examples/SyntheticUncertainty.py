"""Reproduce a survey forecast with joint uncertainty from synthetic readings."""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from Acquisition.AITrace import TraceStore
from Acquisition.TraceAnalysis import ensemble_forecast_maps
from Examples.SyntheticTrace import run as synthetic_trace
from Utils.Serialization import read_json, write_arrays, write_json


def run(output, seed=42):
    output = Path(output)
    synthetic_trace(output, ensemble=True)
    manifest = read_json(output / "ensemble-maps.json")
    forecast = read_json(output / "forecasts.json")
    configuration = read_json(output / "field-settings.json")
    issue = datetime.fromisoformat(manifest["issued_at"])
    scales = [.25, .8, .01, .005, .005, .015]
    residuals = np.random.default_rng(seed).normal(size=(4, 1, 1, 1)) * np.array(scales)
    residuals = np.broadcast_to(residuals, (4, len(manifest["sites"]), 6, 6))
    adaptive = {"covariance": {"range_m": 50.}, "feature_scales": scales,
                "refinement": {"max_depth": 2, "observation_radius_m": 10.}}
    predictive = {
        "feature_scales": scales, "draws": 32, "seed": seed,
        "residual_sites": manifest["sites"], "residuals": residuals,
        "residual_origins": [(issue - timedelta(days=i)).isoformat() for i in (4, 3, 2, 1)],
        "covariance_scenarios": [
            {"id": "short", "weight": .5, "covariance": {"range_m": 24., "shape": "matern32"}},
            {"id": "anisotropic", "weight": .5, "covariance": {
                "range_m": 85., "minor_ratio": .5, "angle_deg": 35., "nugget": .02}},
        ],
    }
    physical = {"checkpoint": output / "synthetic-model.json", "field_configuration": configuration,
                "scenarios": {"draws": 4, "seed": seed,
                              "ranges": [{"path": "crop.critical_temperature_c", "lower": 25., "upper": 42.}]}}
    arrays, metadata = ensemble_forecast_maps(
        TraceStore(output / "trace", recover=False), manifest["survey_id"], forecast,
        manifest["issued_at"], steps=6, adaptive=adaptive, predictive=predictive, physical_sources=physical)
    metadata["synthetic"] = True
    metadata["residual_fixture"] = "Independent synthetic scores, shared across leads and locations for this example"
    write_arrays(output / "predictive-fields.npz", **arrays)
    write_json(output / "predictive-fields.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.output, args.seed)
