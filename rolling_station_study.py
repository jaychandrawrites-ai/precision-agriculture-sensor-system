"""Compare transparent station baselines over reusable chronological folds."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from Data.Preprocessing import read_hourly
from Data.RollingOrigin import rolling_folds
from seasonal_study import assemble
from Temporal.Seasonality import DailyClimatology, HOUR, HarmonicRegression, SeasonalLookup
from Utils.Metrics import horizon_metrics
from Utils.Serialization import write_json


def _timestamp(hour):
    return str(np.datetime64(hour, "h")) + ":00:00Z"


def evaluate(values, times, *, initial_hours, test_hours, horizon=24, stride=None,
             max_training_hours=None):
    recipe = dict(initial_hours=initial_hours, test_hours=test_hours, horizon=horizon,
                  stride=stride, max_training_hours=max_training_hours)
    lookup = SeasonalLookup(values, times)
    leads = np.arange(horizon)
    folds = []
    aggregate = {}
    for fold in rolling_folds(values, times, **recipe):
        train = fold.train_rows
        start, end = fold.test_boundary_hours
        record = {"training_hours": train.stop - train.start,
                  "training_first": str(times[train.start]) + "Z" if train.start < train.stop else None,
                  "training_last": str(times[train.stop - 1]) + "Z" if train.start < train.stop else None,
                  "test_start": _timestamp(start), "test_end_exclusive": _timestamp(end),
                  "eligible_origins": len(fold.starts), "potential_origins": fold.potential}
        folds.append(record)
        if not len(fold.starts):
            record["status"] = "no_complete_windows"
            continue
        climate = DailyClimatology.fit(values[train], times[train])
        harmonic = HarmonicRegression.fit(values[train], times[train])
        starts = fold.starts
        histories = values[starts[:, None] + np.arange(-3, 0)]
        observed = values[starts[:, None] + leads]
        origins = times[starts - 1]
        targets = origins[:, None] + (leads + 1) * HOUR
        day, day_missing = lookup.predict(origins, histories, horizon, 24)
        week, week_missing = lookup.predict(origins, histories, horizon, 168)
        scores = {"persistence": np.broadcast_to(histories[:, -1:, :], observed.shape),
                  "daily": day, "weekly": week, "daily_climatology": climate.predict(targets),
                  "harmonic": harmonic.predict(targets)}
        record["scores_vs_daily"] = {key: horizon_metrics(predicted, observed, day)
                                    for key, predicted in scores.items()}
        record["fallback_hours_per_lead"] = {
            "daily": day_missing.sum(axis=0), "weekly": week_missing.sum(axis=0)
        }
        for key, predicted in scores.items():
            statistics = aggregate.setdefault(key, {"count": 0, "sum_sq": np.zeros((horizon, 6)),
                                                     "sum_abs": np.zeros((horizon, 6))})
            error = predicted - observed
            statistics["count"] += len(error)
            statistics["sum_sq"] += np.square(error).sum(axis=0)
            statistics["sum_abs"] += np.abs(error).sum(axis=0)
    if not aggregate:
        raise ValueError("The predeclared schedule produced no complete evaluation windows")
    return {"recipe": recipe, "folds": folds, "pool": {
        key: {"origins": stats["count"], "rmse": np.sqrt(stats["sum_sq"] / stats["count"]),
              "mae": stats["sum_abs"] / stats["count"]}
        for key, stats in aggregate.items()
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--initial-hours", type=int, required=True)
    parser.add_argument("--test-hours", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-training-hours", type=int)
    args = parser.parse_args()
    values, times = read_hourly(args.sources[0]) if len(args.sources) == 1 else assemble(args.sources)
    result = evaluate(values, times, initial_hours=args.initial_hours, test_hours=args.test_hours,
                      horizon=args.horizon, stride=args.stride, max_training_hours=args.max_training_hours)
    result["sources"] = [{"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}
                         for path in args.sources]
    write_json(args.output, result)


if __name__ == "__main__":
    main()
