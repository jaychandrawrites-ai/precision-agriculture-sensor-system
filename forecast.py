"""Load a checkpoint and forecast from the latest consecutive hourly samples."""
import argparse

import numpy as np
import pandas as pd

from config import FEATURES
from Data.ForecastHistory import require_history
from Data.Preprocessing import read_hourly
from Temporal.Quantile import quantile_columns
from Utils.ModelLoader import load_forecaster


def forecast_csv(source, checkpoint, steps, device="cpu"):
    values, timestamps = read_hourly(source)
    model = load_forecaster(checkpoint, device)
    history = require_history(values, timestamps, hours=model.history_hours)
    predictions, quantiles = model.forecast_bundle(history, steps)
    start = pd.Timestamp(timestamps[-1] + np.timedelta64(1, "h"), tz="UTC")
    frame = pd.DataFrame(predictions, columns=FEATURES)
    if quantiles is not None:
        tail = pd.DataFrame(quantiles.reshape(steps, -1), columns=quantile_columns(model.quantile.levels))
        frame = pd.concat((frame, tail), axis=1)
    frame.insert(0, "timestamp", pd.date_range(start, periods=steps, freq="h"))
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("checkpoint")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(forecast_csv(args.source, args.checkpoint, args.steps, args.device).to_csv(index=False, lineterminator="\n"),
          end="")


if __name__ == "__main__":
    main()
