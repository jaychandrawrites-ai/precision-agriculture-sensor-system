"""Saved seasonal residual ensemble with explicit local history requirements."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Data.DataQuality import invalid_rows
from Temporal.DirectCheckpoint import load_direct, save_direct
from Temporal.Seasonality import ConvexCorrection, SeasonalLookup
from Utils.Serialization import read_json, write_json


def save_member(directory, seed, model, width):
    filename = f"seasonal-seed-{seed}.json"
    save_direct(directory / filename, model, width,
                provenance={"forecast_anchor": "causal_daily_seasonal_lookup", "seed": seed})
    return filename


def save_ensemble(directory, members, blend, intervals, horizon):
    write_json(directory / "seasonal-ensemble.json", {
        "format": "jupiter-seasonal-residual-v1", "members": members, "horizon": horizon,
        "period_hours": 24, "blend": blend.weight, "empirical_marginal_radius": intervals.marginal_radius,
        "empirical_trajectory_radius": intervals.trajectory_radius,
        "calibration_origins": intervals.calibration_count, "nominal_mass": intervals.mass,
        "calibration_spacing_hours": intervals.spacing_hours,
    })


@dataclass(frozen=True)
class SavedSeasonalEnsemble:
    models: tuple
    correction: ConvexCorrection
    intervals: object
    horizon: int

    def predict(self, values, timestamps, origins, histories):
        daily, fallback = SeasonalLookup(values, timestamps).predict(origins, histories, self.horizon)
        total = np.zeros_like(daily)
        for model in self.models:
            total += model.predict(histories, anchor=daily)
        return self.correction.predict(daily, total / len(self.models)), daily, fallback


def load_seasonal(checkpoint, device="cpu"):
    from Temporal.BlockIntervals import BlockIntervals

    checkpoint = Path(checkpoint)
    record = read_json(checkpoint)
    if record.get("format") != "jupiter-seasonal-residual-v1" or record.get("period_hours") != 24:
        raise ValueError("Unsupported seasonal checkpoint")
    horizon = record["horizon"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Invalid seasonal checkpoint horizon")
    blend = np.asarray(record["blend"], dtype=float)
    radius = np.asarray(record["empirical_marginal_radius"], dtype=float)
    joint = np.asarray(record["empirical_trajectory_radius"], dtype=float)
    if (blend.shape != (horizon, 6) or radius.shape != blend.shape or joint.shape != (6,)
            or not all(np.isfinite(x).all() for x in (blend, radius, joint))
            or (blend < 0).any() or (blend > 1).any() or (radius < 0).any() or (joint < 0).any()):
        raise ValueError("Invalid blend or empirical interval parameters")
    models = []
    members = record["members"]
    if (not isinstance(members, list) or not members or not all(isinstance(name, str) for name in members)
            or len(set(members)) != len(members)):
        raise ValueError("Missing or duplicate seasonal members")
    for filename in members:
        if (not isinstance(filename, str) or Path(filename).name != filename
                or not filename.startswith("seasonal-seed-") or not filename.endswith(".json")):
            raise ValueError("Invalid seasonal checkpoint member name")
        model = load_direct(checkpoint.parent / filename, device)
        if model.model.horizon != horizon:
            raise ValueError("Seasonal member horizon differs from the ensemble checkpoint")
        models.append(model)
    if (type(record["calibration_origins"]) is not int or record["calibration_origins"] < 1
            or type(record["calibration_spacing_hours"]) is not int
            or record["calibration_spacing_hours"] < horizon or not 0 < record["nominal_mass"] < 1):
        raise ValueError("Invalid calibration metadata")
    intervals = BlockIntervals(radius, joint, record["calibration_origins"],
                               record["calibration_spacing_hours"], record["nominal_mass"])
    return SavedSeasonalEnsemble(tuple(models), ConvexCorrection(blend), intervals, horizon)


def forecast_saved(checkpoint, values, timestamps, device="cpu"):
    if (len(values) < 3 or invalid_rows(values[-3:]).any()
            or not (np.diff(timestamps[-3:]) == np.timedelta64(1, "h")).all()):
        raise ValueError("Need three latest consecutive, physically valid hourly observations")
    ensemble = load_seasonal(checkpoint, device)
    history = np.asarray(values[-3:], dtype=float)[None]
    point, daily, fallback = ensemble.predict(values, timestamps, timestamps[-1:], history)
    point, radius, joint = point[0], ensemble.intervals.marginal_radius, ensemble.intervals.trajectory_radius
    return {"point": point, "daily": daily[0], "used_current_state": fallback[0],
            "lower": point - radius, "upper": point + radius,
            "trajectory_lower": point - joint, "trajectory_upper": point + joint,
            "nominal_mass": ensemble.intervals.mass, "calibration_origins": ensemble.intervals.calibration_count}
