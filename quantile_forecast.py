"""Train probabilistic forecasts on prepared chronological hourly data."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from Data.DataQuality import invalid_rows
from Data.Preprocessing import read_hourly
from Data.RolloutWindows import evaluation_windows
from Framework.Backend import Backend, BackendConfig
from Temporal.Quantile import QuantileForecaster, quantile_scores
from Utils.Serialization import write_json


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hourly_csv")
    parser.add_argument("checkpoint")
    parser.add_argument("report")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--level", type=float, action="append")
    args = parser.parse_args()
    write_json(args.report, train_quantiles(args.hourly_csv, args.checkpoint, horizon=args.horizon,
                                           epochs=args.epochs, seed=args.seed,
                                           levels=args.level or (.05, .5, .95)))
