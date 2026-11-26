"""Calibrate a frozen quantile checkpoint on explicitly separate time periods."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from Data.Preprocessing import read_hourly
from Data.RolloutWindows import evaluation_windows
from Temporal.Quantile import QuantileForecaster
from Temporal.QuantileCalibration import QuantileCalibration
from Utils.Serialization import write_json


def evaluate(source, checkpoint, calibration_start, test_start, *, stride=None):
    from Temporal.Seasonality import clock_hours

    model = QuantileForecaster.load(checkpoint)
    values, times = read_hourly(source)
    dates = np.asarray([calibration_start, test_start], dtype="datetime64[ns]")
    clock_hours(dates)
    cal, test = np.searchsorted(times, dates)
    if not 3 <= cal < test < len(times):
        raise ValueError("Provide separate ordered calibration and evaluation periods")
    slices = tuple(evaluation_windows(values, times, model.horizon, np.arange(start, stop))
                   for start, stop in ((cal, test), (test, len(times))))
    if any(not len(batch.starts) for batch in slices):
        raise ValueError("Each period needs at least one complete valid trajectory")
    arguments = []
    for batch in slices:
        q = model.predict(batch.histories)
        arguments.append((q[..., 0], q[..., -1], batch.targets, times[batch.starts - 1]))
    scales = np.broadcast_to(model.scale, (model.horizon, len(model.scale)))
    result = QuantileCalibration.fit(*arguments[0], scales, stride=stride,
                                     mass=float(model.levels[-1] - model.levels[0]))
    return {"source_sha256": sha256(Path(source).read_bytes()).hexdigest(),
            "checkpoint_sha256": sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "calibration_start_utc": str(dates[0]), "test_start_utc": str(dates[1]),
            "evaluation": result.score(*arguments[1])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, epilog="Choose calibration dates after all training and model selection.")
    parser.add_argument("hourly_csv")
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--calibration-start", required=True, help="Naive ISO date interpreted as UTC")
    parser.add_argument("--test-start", required=True, help="Naive ISO date interpreted as UTC")
    parser.add_argument("--stride", type=int)
    args = parser.parse_args()
    write_json(args.output, evaluate(args.hourly_csv, args.checkpoint, args.calibration_start,
                                    args.test_start, stride=args.stride))
