"""Evaluate recursive forecasts with matched windows, model sizes, epochs and seeds."""
import argparse
from dataclasses import replace
import json

import numpy as np

from config import TrainingConfig
from Data.Dataset import build_dataset
from Data.Preprocessing import read_hourly
from Data.RolloutWindows import rollout_windows
from Framework.Backend import create_backend
from Temporal.Baselines import baseline_forecasts
from Temporal.Evaluator import evaluate
from Temporal.Model import TemporalModel
from Temporal.Rollout import Rollout, RolloutOptions
from Temporal.RolloutTrainer import fit_rollout
from Temporal.Trainer import fit
from train import _physical_metrics


def score(predictions, targets, persistence):
    return [_physical_metrics(evaluate(predictions[:, i], targets[:, i], persistence[:, i]))
            for i in range(targets.shape[1])]


def run(values, timestamps, settings, seeds, steps=6):
    dataset = build_dataset(values, timestamps, settings.train_fraction,
                            settings.validation_fraction)
    training = rollout_windows(dataset, dataset.train, values, timestamps, 6)
    validation = rollout_windows(dataset, dataset.validation, values, timestamps, 6)
    test = rollout_windows(dataset, dataset.test, values, timestamps, steps)
    if not len(test.starts):
        raise ValueError("No complete test windows for the requested horizon")
    persistence, seasonal, fallback = baseline_forecasts(values, timestamps, test)
    report = {
        "input_rows": len(values),
        "split_target_rows": {split: len(getattr(dataset, split))
                              for split in ("train", "validation", "test")},
        "horizons": list(range(1, steps + 1)),
        "windows": {name: {"scored": len(window.starts), "candidates": window.potential,
                           "first_origin": str(timestamps[window.starts[0] - 1]) if len(window.starts) else None,
                           "last_origin": str(timestamps[window.starts[-1] - 1]) if len(window.starts) else None}
                    for name, window in (("train", training), ("validation", validation), ("test", test))},
        "seasonal_fallbacks": fallback.tolist(),
        "persistence": score(persistence, test.targets, persistence),
        "seasonal_persistence": score(seasonal, test.targets, persistence),
        "runs": [],
    }
    for seed in seeds:
        config = replace(settings, seed=seed)
        for name, options, autoregressive, warmup in variants():
            backend = create_backend(config.device, config.dtype, seed)
            model = TemporalModel(backend, config.hidden, config.dropout)
            if name == "legacy":
                fitted = fit(model, dataset, backend, config)
                inputs = dataset.input_scaler
                diagnostics = {"model_training_samples": config.epochs * len(dataset.train)}
            else:
                fitted, inputs, diagnostics = fit_rollout(
                    model, dataset, training, validation, backend, config, options, autoregressive,
                    warmup_epochs=config.epochs // 3 if warmup else 0
                )
                diagnostics["model_training_samples"] = (
                    config.epochs * len(training.starts) * training.targets.shape[1]
                )
            rollout = Rollout(model, inputs, dataset.target_scaler, backend, options)
            projected, raw, rollout_diagnostics = rollout.predict(test.histories, steps, True)
            report["runs"].append({
                "seed": seed, "variant": name, "best_epoch": fitted.best_epoch,
                "parameters": model.parameter_count, "fit_diagnostics": diagnostics,
                "validation_loss": fitted.validation_loss,
                "projected": score(projected, test.targets, persistence),
                "raw": score(raw, test.targets, persistence),
                "rollout_diagnostics": rollout_diagnostics,
            })
    return report


def variants():
    yield "legacy", RolloutOptions("state", 1.0, None), False, False
    for output in ("state", "delta"):
        for robust in (False, True):
            options = RolloutOptions(output) if robust else RolloutOptions(output, 1.0, None)
            for autoregressive in (False, True):
                label = f"{output}_{'robust' if robust else 'raw'}_{'unrolled' if autoregressive else 'teacher'}"
                yield label, options, autoregressive, False
        yield f"{output}_robust_warmup", RolloutOptions(output), True, True
    yield "delta_smoothing_only_warmup", RolloutOptions("delta", 0.65, None), True, True
    yield "delta_acceleration_only_warmup", RolloutOptions("delta", 1.0, 2.0), True, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="Prepared CSV; omit for the synthetic fixture")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    if args.source:
        values, timestamps = read_hourly(args.source)
    else:
        from main import synthetic_history

        frame = synthetic_history()
        values = frame.iloc[:, 1:].to_numpy()
        timestamps = frame.timestamp.to_numpy(dtype="datetime64[ns]")
    config = TrainingConfig(epochs=args.epochs, batch_size=32, dropout=0)
    report = run(values, timestamps, config, args.seeds, args.steps)
    report["source"] = args.source or "synthetic periodic fixture"
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
