"""Train from a prepared hourly CSV and evaluate the best validation model."""
import argparse
import json
from dataclasses import replace

import numpy as np

from config import FEATURES, TrainingConfig
from Data.Dataset import build_dataset
from Data.Preprocessing import read_hourly
from Framework.Backend import create_backend
from Temporal.Model import TemporalModel
from Temporal.Trainer import fit
from Temporal.Validator import evaluate_rollouts, evaluate_rows
from Utils.Checkpoints import save


def train(source, checkpoint, settings, optimizer="adam", loss="huber"):
    values, timestamps = read_hourly(source)
    dataset = build_dataset(values, timestamps, settings.train_fraction,
                            settings.validation_fraction)
    backend = create_backend(settings.device, settings.dtype, settings.seed)
    model = TemporalModel(backend, settings.hidden, settings.dropout)
    result = fit(model, dataset, backend, settings, optimizer, loss)
    metrics = evaluate_rows(model, dataset, dataset.test, backend)
    rollout = evaluate_rollouts(model, dataset, dataset.test, values, timestamps,
                                backend, settings.rollout_steps)
    save(checkpoint, model, dataset, backend, result, settings)
    return {
        "examples": {split: len(getattr(dataset, split))
                     for split in ("train", "validation", "test")},
        "best_epoch": result.best_epoch,
        "validation_loss": result.validation_loss,
        "parameters": model.parameter_count,
        "test": _physical_metrics(metrics),
        "test_rollout": [_physical_metrics(horizon) for horizon in rollout],
    }


def _physical_metrics(metrics):
    return {
            feature: {metric: float(values[i]) if np.isfinite(values[i]) else None
                      for metric, values in metrics.items()}
        for i, feature in enumerate(FEATURES)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("checkpoint")
    parser.add_argument("--epochs", type=int, default=TrainingConfig().epochs)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    args = parser.parse_args()
    settings = replace(TrainingConfig(), epochs=args.epochs, device=args.device)
    print(json.dumps(train(args.source, args.checkpoint, settings, args.optimizer, args.loss),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
