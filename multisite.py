"""Forecast independent field sessions from one checkpoint and align spatial snapshots."""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd

from config import FEATURES
from Pipeline import forecast_prepared_session
from Utils.ModelLoader import load_forecaster
from Utils.Serialization import read_json, write_json


def forecast_multiple(locations, checkpoint, steps, *, device="cpu"):
    """Each location keeps its own observed history and sensor calibration."""
    if not isinstance(locations, dict) or not locations:
        raise ValueError("Supply a nonempty mapping of named field sessions")
    if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name)
           for name in locations):
        raise ValueError("Use short alphanumeric location identifiers")
    forecaster = load_forecaster(checkpoint, device)
    forecast, reports, positions = {}, {}, {}
    for name, location in locations.items():
        frame, report, gps, _ = forecast_prepared_session(
            location["readings"], location["configuration"], forecaster, steps
        )
        forecast[name], reports[name], positions[name] = frame, report, gps
    return forecast, reports, positions


def aligned_snapshot(forecasts, positions, *, feature, valid_at, max_gps_age_minutes=60):
    """Return model predictions paired only with recent positions known as of a shared origin."""
    target = pd.Timestamp(valid_at)
    if (feature not in FEATURES or pd.isna(target) or target.tzinfo is None
            or not np.isfinite(max_gps_age_minutes) or max_gps_age_minutes <= 0):
        raise ValueError("Use a known feature, aware target time and positive GPS freshness limit")
    target = target.tz_convert("UTC")
    rows, excluded = [], {}
    origin = None
    for name, frame in forecasts.items():
        if target not in frame.index:
            excluded[name] = "outside forecast period"
            continue
        current_origin = frame.index[0] - pd.offsets.Hour()
        fixes = positions[name]
        times = pd.to_datetime(fixes["timestamp"], utc=True)
        recent = fixes.loc[(times <= current_origin) &
                           (times > current_origin - pd.offsets.Minute(max_gps_age_minutes))]
        if recent.empty:
            excluded[name] = "no recent GPS fix known at the forecast origin"
            continue
        if origin is not None and current_origin != origin:
            raise ValueError("Spatial snapshots require a common forecast origin")
        origin = current_origin
        last = recent.sort_values("timestamp").iloc[-1]
        rows.append({"timestamp": target, "forecast_origin": origin,
                     "location": name, "value_origin": "model_forecast", "latitude_deg": last.latitude_deg,
                     "longitude_deg": last.longitude_deg, feature: frame.loc[target, feature]})
    result = pd.DataFrame(rows)
    if len(result) < 2 or len(result[["latitude_deg", "longitude_deg"]].drop_duplicates()) < 2:
        raise ValueError("Need forecasts and fresh GPS fixes from at least two distinct sites")
    return result, {"feature": feature, "valid_at": target.isoformat(), "origin": origin.isoformat(),
                    "lead_hours": (target - origin).total_seconds() / 3600,
                    "value_origin": "model_forecasts_for_mapping_only",
                    "used_locations": len(result), "excluded_locations": excluded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("locations", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--snapshot-at")
    parser.add_argument("--feature", choices=FEATURES, default=FEATURES[0])
    args = parser.parse_args()
    config = read_json(args.locations)
    forecasts, reports, positions = forecast_multiple(config, args.checkpoint, args.steps)
    args.output.mkdir()
    for name, frame in forecasts.items():
        frame.to_csv(args.output / f"{name}.csv")
    result = {"locations": reports}
    if args.snapshot_at:
        snapshot, report = aligned_snapshot(forecasts, positions, feature=args.feature,
                                            valid_at=args.snapshot_at)
        snapshot.to_csv(args.output / "forecast-snapshot.csv", index=False)
        result["snapshot"] = report
    write_json(args.output / "report.json", result)


if __name__ == "__main__":
    main()
