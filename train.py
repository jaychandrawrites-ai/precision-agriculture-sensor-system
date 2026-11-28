"""Train from a prepared hourly CSV and evaluate the best validation model."""
import argparse
import json
from pathlib import Path
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


def train(source, checkpoint, settings, optimizer="adam", loss="huber", stable_rollout=False, *, recovery_path=None, diagnostics_dir=None, preparation_report=None, diagnostic_every=0):
    from hashlib import sha256

    values, timestamps = read_hourly(source)
    provenance = None if diagnostics_dir is None else {"prepared_sha256": sha256(Path(source).read_bytes()).hexdigest()}
    if preparation_report is not None:
        from Utils.Serialization import read_json

        upstream = read_json(preparation_report)["provenance"]
        provenance = {**upstream, "prepared_sha256": sha256(Path(source).read_bytes()).hexdigest()}
    return train_values(values, timestamps, checkpoint, settings, optimizer, loss,
                        stable_rollout=stable_rollout, recovery_path=recovery_path, diagnostics_dir=diagnostics_dir, provenance=provenance, diagnostic_every=diagnostic_every)


def train_values(values, timestamps, checkpoint, settings, optimizer="adam", loss="huber",
                 *, stable_rollout=True, provenance=None, cutoffs=None, recovery_path=None, diagnostics_dir=None, diagnostic_every=0):
    if recovery_path is not None and Path(recovery_path).resolve() == Path(checkpoint).resolve():
        raise ValueError("Training recovery and inference checkpoint must use distinct files")
    dataset = build_dataset(values, timestamps, settings.train_fraction,
                            settings.validation_fraction, cutoffs=cutoffs)
    backend = create_backend(settings.device, settings.dtype, settings.seed)
    model = TemporalModel(backend, settings.hidden, settings.dropout)
    if stable_rollout:
        return _train_stable(checkpoint, settings, optimizer, loss, values, timestamps,
                             dataset, model, backend, provenance, recovery_path, diagnostics_dir, diagnostic_every)
    if recovery_path is not None or diagnostics_dir is not None:
        raise ValueError("Use stable rollout training for epoch recovery or diagnostics")
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
                  dataset, model, backend, provenance, recovery_path, diagnostics_dir, diagnostic_every):
    from time import perf_counter

    steps = settings.rollout_steps
    training, validation, test = [
        rollout_windows(dataset, rows, values, timestamps, steps)
        for rows in (dataset.train, dataset.validation, dataset.test)
    ]
    options = RolloutOptions()
    result, inputs, fit_diagnostics = fit_rollout(
        model, dataset, training, validation, backend, settings, options,
        optimizer=optimizer, loss=loss, warmup_epochs=settings.epochs // 3, recovery_path=recovery_path,
        diagnostic_every=diagnostic_every,
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
        started = perf_counter()
        projected, raw, diagnostics = engine.predict(test.histories, steps, True)
        report["test_inference_seconds"] = perf_counter() - started
        for name, predictions in (("test_rollout", projected), ("raw_rollout", raw),
                                  ("persistence", persistence), ("seasonal_persistence", seasonal)):
            report[name] = horizon_metrics(predictions, test.targets, persistence)
        report["seasonal_fallbacks"] = fallback.tolist()
        report["rollout_diagnostics"] = diagnostics
    save(checkpoint, model, replace(dataset, input_scaler=inputs), backend, result, settings,
         provenance=provenance)
    if diagnostics_dir is not None and len(test.starts):
        from AIDiagnostics import publish
        from dataclasses import asdict
        from hashlib import sha256
        import numpy as np

        report["configuration"] = asdict(settings)
        report["checkpoint_bytes"] = Path(checkpoint).stat().st_size
        report["checkpoint_sha256"] = sha256(Path(checkpoint).read_bytes()).hexdigest()
        report["provenance"] = provenance
        report["target_periods"] = {
            name: {"first_utc": str(timestamps[window.starts[0]]),
                   "last_utc": str(timestamps[window.starts[-1]+steps-1])}
            for name, window in zip(("train", "validation", "test"), (training, validation, test))
            if len(window.starts)}
        publish(diagnostics_dir, report, test.targets, projected, persistence, seasonal,
                model.state_dict(backend), timestamps[test.starts[:, None]+np.arange(steps)])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("checkpoint")
    parser.add_argument("--seed", type=int, default=TrainingConfig().seed)
    parser.add_argument("--preparation-report", type=Path)
    parser.add_argument("--diagnostic-every", type=int, default=0)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--epochs", type=int, default=TrainingConfig().epochs)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    parser.add_argument("--resume-state", help="Save or resume an epoch-boundary CPU training run")
    parser.add_argument("--legacy", action="store_true",
                        help="Use the original one-step state model")
    args = parser.parse_args()
    settings = replace(TrainingConfig(), epochs=args.epochs, device=args.device, seed=args.seed)
    print(json.dumps(train(args.source, args.checkpoint, settings, args.optimizer, args.loss,
                           stable_rollout=not args.legacy, recovery_path=args.resume_state, diagnostics_dir=args.diagnostics, preparation_report=args.preparation_report, diagnostic_every=args.diagnostic_every), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
