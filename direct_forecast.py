"""Train or use an isolated experimental direct multi-horizon checkpoint."""
import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURES, TrainingConfig
from Data.ForecastHistory import require_history
from Data.Dataset import build_dataset
from Data.Preprocessing import read_hourly
from Data.RolloutWindows import rollout_windows
from Data.TimeSeries import calendar_cutoffs
from Framework.Backend import create_backend
from Temporal.Direct import DIRECT_ARCHITECTURES, fit_direct
from Temporal.DirectCheckpoint import load_direct, save_direct
from Temporal.SequenceForecast import SequenceSpec
from Utils.Serialization import atomic_output, read_json, write_json


def train(source, checkpoint, settings, horizon, width=32, *, architecture="gated", **fit_options):
    values, times = read_hourly(source)
    cuts = calendar_cutoffs(times, settings.train_fraction, settings.validation_fraction)
    data = build_dataset(values, times, cutoffs=cuts)
    context = architecture.context if isinstance(architecture, SequenceSpec) else 3
    if isinstance(architecture, SequenceSpec):
        width = architecture.width
    train_windows = rollout_windows(data, data.train, values, times, horizon, context=context)
    validation = rollout_windows(data, data.validation, values, times, horizon, context=context)
    model, report = fit_direct(train_windows, validation, data.target_scaler,
                               create_backend(settings.device, settings.dtype, settings.seed),
                               settings, width=width, architecture=architecture, **fit_options)
    save_direct(checkpoint, model, width, provenance={
        "source_sha256": sha256(Path(source).read_bytes()).hexdigest(),
        "cutoffs_utc": [str(value) + "Z" for value in cuts],
        "seed": settings.seed, "epochs": settings.epochs,
    })
    return report


def predict(source, checkpoint):
    values, times = read_hourly(source)
    model = load_direct(checkpoint)
    history = require_history(values, times, hours=model.context_hours)
    prediction = model.predict(history[None])[0]
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
            sub.add_argument("--memory-bytes", type=int)
            sub.add_argument("--optimizer", choices=("adam", "adamw", "sgd", "rmsprop", "adagrad"), default="adam")
            sub.add_argument("--weight-decay", type=float, default=0.)
            sub.add_argument("--scheduler", choices=("cosine", "plateau"), default="cosine")
            sub.add_argument("--warmup-epochs", type=int, default=0)
            sub.add_argument("--patience", type=int)
            sub.add_argument("--sequence-config", type=Path, help="Sequence architecture JSON, such as an exported search recipe")
        else:
            sub.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        settings = replace(TrainingConfig(), epochs=args.epochs, dropout=0)
        architecture = args.architecture
        if args.sequence_config is not None:
            record = read_json(args.sequence_config)
            architecture = SequenceSpec(**record.get("architecture", record))
        report = train(args.source, args.checkpoint, settings, args.horizon, args.width,
                       architecture=architecture, memory_budget_bytes=args.memory_bytes,
                       optimizer=args.optimizer, weight_decay=args.weight_decay,
                       scheduler=args.scheduler, warmup_epochs=args.warmup_epochs, patience=args.patience)
        write_json(args.checkpoint.with_suffix(".training.json"), report)
    else:
        with atomic_output(args.output) as file:
            predict(args.source, args.checkpoint).to_csv(file)


if __name__ == "__main__":
    main()
