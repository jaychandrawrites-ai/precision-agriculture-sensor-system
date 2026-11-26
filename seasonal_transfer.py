"""Evaluate a frozen seasonal ensemble on a separately prepared station."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from Data.Preprocessing import read_hourly
from Data.RolloutWindows import evaluation_windows
from Temporal.SeasonalCheckpoint import load_seasonal
from Utils.Bootstrap import seasonal_temperature_comparison
from Utils.Metrics import horizon_metrics
from Utils.Serialization import write_json


def evaluate_transfer(ensemble, values, times):
    windows = evaluation_windows(values, times, ensemble.horizon)
    if not len(windows.starts):
        raise ValueError("No complete transfer forecast windows")
    origins = times[windows.starts - 1]
    estimate, daily, fallback = ensemble.predict(values, times, origins, windows.histories)
    persistence = np.broadcast_to(windows.histories[:, -1:, :], windows.targets.shape)
    predictions = {"seasonal_blend": estimate, "daily": daily, "persistence": persistence}
    return {"evaluation_origins": len(origins), "first_origin": str(origins[0]) + "Z",
            "last_origin": str(origins[-1]) + "Z", "horizon": ensemble.horizon,
            "seasonal_fallbacks_by_lead": fallback.sum(axis=0),
            "scores": {key: horizon_metrics(value, windows.targets, daily)
                       for key, value in predictions.items()},
            "transferred_interval_diagnostics": ensemble.intervals.score(estimate, windows.targets, origins),
            "adapted_on_target_station": False,
            "paired_calendar_comparison": seasonal_temperature_comparison(estimate, daily, windows.targets, origins),
            "interval_calibration": "inherited from source station, without a transfer coverage guarantee"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--station", required=True)
    args = parser.parse_args()
    report = evaluate_transfer(load_seasonal(args.checkpoint), *read_hourly(args.source))
    report["manifest"] = {"target_station": args.station,
                          "source_sha256": sha256(args.source.read_bytes()).hexdigest(),
                          "checkpoint_sha256": sha256(args.checkpoint.read_bytes()).hexdigest()}
    write_json(args.output, report)


if __name__ == "__main__":
    main()
