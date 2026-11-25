"""Train from a prepared hourly CSV and evaluate the best validation model."""
import argparse
import json
from dataclasses import replace

from config import TrainingConfig
from Data.Dataset import build_dataset
from Data.Preprocessing import read_hourly
from Data.RolloutWindows import rollout_windows
from Framework.Backend import create_backend
from Temporal.Model import TemporalModel
from Temporal.Baselines import baseline_forecasts
from Temporal.Evaluator import evaluate
from Temporal.Rollout import Rollout, RolloutOptions
from Temporal.RolloutTrainer import fit_rollout
from Temporal.Trainer import fit
from Temporal.Validator import evaluate_rollouts, evaluate_rows
from Utils.Checkpoints import save
from Utils.Metrics import horizon_metrics, physical_metrics as _physical_metrics


def train(source, checkpoint, settings, optimizer="adam", loss="huber", stable_rollout=False):
    values, timestamps = read_hourly(source)
    return train_values(values, timestamps, checkpoint, settings, optimizer, loss,
                        stable_rollout=stable_rollout)


def train_values(values, timestamps, checkpoint, settings, optimizer="adam", loss="huber",
                 *, stable_rollout=True, provenance=None):
    dataset = build_dataset(values, timestamps, settings.train_fraction,
                            settings.validation_fraction)
    backend = create_backend(settings.device, settings.dtype, settings.seed)
    model = TemporalModel(backend, settings.hidden, settings.dropout)
    if stable_rollout:
        return _train_stable(checkpoint, settings, optimizer, loss, values, timestamps,
                             dataset, model, backend, provenance)
    result = fit(model, dataset, backend, settings, optimizer, loss)
    metrics = evaluate_rows(model, dataset, dataset.test, backend)
    rollout = evaluate_rollouts(model, dataset, dataset.test, values, timestamps,
                                backend, settings.rollout_steps)
    save(checkpoint, model, dataset, backend, result, settings, provenance=provenance)
    return {
        "examples": {split: len(getattr(dataset, split))
                     for split in ("train", "validation", "test")},
        "best_epoch": result.best_epoch,
        "validation_loss": result.validation_loss,
        "parameters": model.parameter_count,
        "test": _physical_metrics(metrics),
        "test_rollout": [_physical_metrics(horizon) for horizon in rollout],
    }


def _train_stable(checkpoint, settings, optimizer, loss, values, timestamps,
                  dataset, model, backend, provenance):
    steps = settings.rollout_steps
    training, validation, test = [
        rollout_windows(dataset, rows, values, timestamps, steps)
        for rows in (dataset.train, dataset.validation, dataset.test)
    ]
    options = RolloutOptions()
    result, inputs, fit_diagnostics = fit_rollout(
        model, dataset, training, validation, backend, settings, options,
        optimizer=optimizer, loss=loss, warmup_epochs=settings.epochs // 3,
    )
    engine = Rollout(model, inputs, dataset.target_scaler, backend, options)
    one_step = rollout_windows(dataset, dataset.test, values, timestamps, 1)
    if not len(one_step.starts):
        raise ValueError("No complete test windows")
    prediction = engine.predict(one_step.histories, 1)[:, 0]
    metrics = evaluate(prediction, one_step.targets[:, 0], one_step.histories[:, -1])
    report = {
        "examples": {"train": len(dataset.train), "validation": len(dataset.validation),
                     "test": len(dataset.test)},
        "rollout_windows": {"train": len(training.starts), "validation": len(validation.starts),
                            "test": len(test.starts)},
        "best_epoch": result.best_epoch,
        "validation_loss": result.validation_loss,
        "parameters": model.parameter_count,
        "test": _physical_metrics(metrics),
        "test_rollout": [],
        "fit_diagnostics": fit_diagnostics,
    }
    if len(test.starts):
        persistence, seasonal, fallback = baseline_forecasts(values, timestamps, test)
        projected, raw, diagnostics = engine.predict(test.histories, steps, True)
        for name, predictions in (("test_rollout", projected), ("raw_rollout", raw),
                                  ("persistence", persistence), ("seasonal_persistence", seasonal)):
            report[name] = horizon_metrics(predictions, test.targets, persistence)
        report["seasonal_fallbacks"] = fallback.tolist()
        report["rollout_diagnostics"] = diagnostics
    save(checkpoint, model, replace(dataset, input_scaler=inputs), backend, result, settings,
         provenance=provenance)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("checkpoint")
    parser.add_argument("--epochs", type=int, default=TrainingConfig().epochs)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--legacy", action="store_true",
                        help="Use the original one-step state model")
    args = parser.parse_args()
    settings = replace(TrainingConfig(), epochs=args.epochs, device=args.device)
    print(json.dumps(train(args.source, args.checkpoint, settings, args.optimizer, args.loss,
                           stable_rollout=not args.legacy), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
