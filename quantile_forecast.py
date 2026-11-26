"""Train probabilistic forecasts on prepared chronological hourly data."""
import argparse
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from config import FEATURES
from Data.DataQuality import invalid_rows
from Data.ForecastHistory import require_history
from Data.Preprocessing import read_hourly
from Data.RolloutWindows import evaluation_windows
from Framework.Backend import Backend, BackendConfig
from Temporal.Quantile import QuantileForecaster, quantile_columns, quantile_scores
from Utils.ModelLoader import load_forecaster
from Utils.Serialization import atomic_output, write_json


def train_quantiles(source, checkpoint, *, horizon=6, epochs=20, batch_size=64, seed=42,
                    levels=(.05, .5, .95)):
    if min(epochs, batch_size) < 1 or not 1 <= horizon <= 72:
        raise ValueError("Invalid epoch, batch or horizon budget")
    values, timestamps = read_hourly(source)
    train_end, validation_end = int(.7 * len(values)), int(.85 * len(values))
    partitions = [(3, train_end), (train_end, validation_end), (validation_end, len(values))]
    batches = [evaluation_windows(values, timestamps, horizon, np.arange(lo, hi))
               for lo, hi in partitions]
    if min(len(batch.histories) for batch in batches) == 0:
        raise ValueError("Every split needs a complete genuine hourly trajectory")
    training, validation, test = batches
    fit = values[:train_end]
    fit = fit[~invalid_rows(fit)]
    center, scale = fit.mean(axis=0), fit.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.)
    model = QuantileForecaster(Backend(BackendConfig(seed=seed)), center, scale, horizon, levels)
    rng = np.random.default_rng(seed)
    best, history = float("inf"), []
    for epoch in range(epochs):
        order = rng.permutation(len(training.histories))
        for start in range(0, len(order), batch_size):
            batch = order[start:start+batch_size]
            model.step(training.histories[batch], training.targets[batch])
        loss = model.objective(validation.histories, validation.targets)
        history.append(loss)
        if loss < best:
            best, best_epoch = loss, epoch + 1
            model.save(checkpoint)
    model = QuantileForecaster.load(checkpoint)
    return {"source_sha256": sha256(Path(source).read_bytes()).hexdigest(), "seed": seed,
            "training_rows": train_end, "test_start": str(timestamps[validation_end]),
            "usable_origins": [len(batch.histories) for batch in batches],
            "potential_origins": [batch.potential for batch in batches],
            "levels": model.levels, "selection_epoch": best_epoch, "validation_pinball": history,
            "test": quantile_scores(model.predict(test.histories), test.targets, model.levels)}


def predict_quantiles(source, checkpoint, *, steps=None, device="cpu"):
    """Return a CSV-ready record with central estimates and all trained levels."""
    values, timestamps = read_hourly(source)
    model = load_forecaster(checkpoint, device)
    if model.quantile is None:
        raise ValueError("Expected a Jupiter quantile checkpoint")
    history = require_history(values, timestamps, hours=model.history_hours)
    central, trajectories = model.forecast_bundle(history, steps)
    index = pd.date_range(pd.Timestamp(timestamps[-1] + np.timedelta64(1, "h"), tz="UTC"),
                          periods=len(central), freq="h", name="timestamp")
    return pd.DataFrame(
        np.column_stack((central, trajectories.reshape(len(central), -1))),
        index=index, columns=[*FEATURES, *quantile_columns(model.quantile.levels)],
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["predict"]:
        parser = argparse.ArgumentParser(description="Predict from a trained quantile checkpoint")
        parser.add_argument("hourly_csv")
        parser.add_argument("checkpoint")
        parser.add_argument("output")
        parser.add_argument("--steps", type=int)
        parser.add_argument("--device", default="cpu")
        options = parser.parse_args(args[1:])
        result = predict_quantiles(options.hourly_csv, options.checkpoint,
                                   steps=options.steps, device=options.device)
        with atomic_output(options.output) as handle:
            result.to_csv(handle)
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hourly_csv")
    parser.add_argument("checkpoint")
    parser.add_argument("report")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level", type=float, action="append")
    options = parser.parse_args(args)
    write_json(options.report, train_quantiles(options.hourly_csv, options.checkpoint, horizon=options.horizon,
                                               epochs=options.epochs, seed=options.seed,
                                               levels=options.level or (.05, .5, .95)))


if __name__ == "__main__":
    main()
