"""Train or use an isolated experimental direct multi-horizon checkpoint."""
import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURES, TrainingConfig
from Data.DataQuality import invalid_rows
from Data.Dataset import build_dataset
from Data.Preprocessing import read_hourly
from Data.RolloutWindows import rollout_windows
from Data.TimeSeries import calendar_cutoffs
from Framework.Backend import create_backend
from Temporal.Direct import DIRECT_ARCHITECTURES, fit_direct
from Temporal.DirectCheckpoint import load_direct, save_direct
from Utils.Serialization import atomic_output, write_json


def train(source, checkpoint, settings, horizon, width=32, *, architecture="gated"):
    values, times = read_hourly(source)
    cuts = calendar_cutoffs(times, settings.train_fraction, settings.validation_fraction)
    data = build_dataset(values, times, cutoffs=cuts)
    train_windows = rollout_windows(data, data.train, values, times, horizon)
    validation = rollout_windows(data, data.validation, values, times, horizon)
    model, report = fit_direct(train_windows, validation, data.target_scaler,
                               create_backend(settings.device, settings.dtype, settings.seed),
                               settings, width=width, architecture=architecture)
    save_direct(checkpoint, model, width, provenance={
        "source_sha256": sha256(Path(source).read_bytes()).hexdigest(),
        "cutoffs_utc": [str(value) + "Z" for value in cuts],
        "seed": settings.seed, "epochs": settings.epochs,
    })
    return report


def predict(source, checkpoint):
    values, times = read_hourly(source)
    if (len(values) < 3 or invalid_rows(values[-3:]).any()
            or not (np.diff(times[-3:]) == np.timedelta64(1, "h")).all()):
        raise ValueError("Direct forecasts require three valid consecutive observations")
    model = load_direct(checkpoint)
    prediction = model.predict(values[None, -3:])[0]
    start = pd.Timestamp(times[-1] + np.timedelta64(1, "h"), tz="UTC")
    return pd.DataFrame(prediction, columns=FEATURES,
                        index=pd.date_range(start, periods=len(prediction), freq="h", name="timestamp"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "predict"):
        sub = commands.add_parser(name)
        sub.add_argument("source", type=Path)
        sub.add_argument("checkpoint", type=Path)
        if name == "train":
            sub.add_argument("--epochs", type=int, default=30)
            sub.add_argument("--horizon", type=int, default=24)
            sub.add_argument("--width", type=int, default=32)
            sub.add_argument("--architecture", choices=DIRECT_ARCHITECTURES, default="gated")
        else:
            sub.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        settings = replace(TrainingConfig(), epochs=args.epochs, dropout=0)
        report = train(args.source, args.checkpoint, settings, args.horizon, args.width,
                       architecture=args.architecture)
        write_json(args.checkpoint.with_suffix(".training.json"), report)
    else:
        with atomic_output(args.output) as file:
            predict(args.source, args.checkpoint).to_csv(file)


if __name__ == "__main__":
    main()
