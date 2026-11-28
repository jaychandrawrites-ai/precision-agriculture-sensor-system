"""Train, evaluate and export fixed and evolved spectral forecasters."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from AIDiagnostics import publish, publish_evolution
from config import EvolutionConfig, FEATURES, TrainingConfig
from Data.Dataset import build_dataset
from Data.Preprocessing import read_hourly
from Data.RolloutWindows import rollout_windows
from Data.TimeSeries import calendar_cutoffs
from Evolution.Fitness import Candidate
from Evolution.Genome import Genome
from Evolution.Resumable import SearchOptions, run_search, validation_fingerprint
from seasonal_study import utc_threshold
from Temporal.Baselines import baseline_forecasts
from Temporal.Spectral import HORIZON, PERIODS, LossWeights, Modes
from Temporal.SpectralTraining import fit
from Utils.Metrics import horizon_metrics
from Utils.ModelLoader import load_forecaster
from Utils.Serialization import read_json, write_json


def decode(bits, damped):
    architecture = Genome(bits[:17]).decode().model_arguments()
    profile = bits[17:]
    periods = tuple(p for p in PERIODS if p == 24 or profile == (0, 0, 0)
                    or (p in (72, 36) and profile[0])
                    or (p in (18, 12) and profile[1])
                    or (p in (8, 6) and profile[2]))
    return architecture, Modes(periods, damped and bool(profile[0]))


def pattern_metrics(predicted, observed, modes):
    def daily(values):
        coefficients = np.linalg.pinv(Modes((24,)).basis(np.arange(1, HORIZON + 1))) @ values
        return coefficients[:, 1] + 1j * coefficients[:, 2]

    a, b = daily(predicted), daily(observed)
    centered = predicted - predicted.mean(axis=1, keepdims=True)
    truth = observed - observed.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(centered, axis=1) * np.linalg.norm(truth, axis=1)
    basis = modes.basis(np.arange(1, HORIZON + 1))
    inverse = np.linalg.pinv(basis)
    spectrum = np.fft.rfft(predicted - predicted.mean(axis=1, keepdims=True), axis=1)
    physical_energy = np.abs(spectrum)**2
    row = {}
    for j, feature in enumerate(FEATURES):
        eligible = (np.abs(a[:, j]) > 1e-6) & (np.abs(b[:, j]) > 1e-6)
        valid = denominator[:, j] > 1e-12
        row[feature] = {
            "trajectory_correlation": float(np.mean((centered * truth).sum(axis=1)[valid, j] / denominator[valid, j])) if valid.any() else None,
            "correlation_origins": int(valid.sum()),
            "daily_amplitude_mae": float(np.abs(np.abs(a[:, j]) - np.abs(b[:, j])).mean()),
            "daily_phase_mae_radians": float(np.abs(np.angle(a[eligible, j] * b[eligible, j].conj())).mean()) if eligible.any() else None,
            "phase_origins": int(eligible.sum()),
            "derivative_rmse": [float(np.sqrt(np.mean(np.diff(predicted[:, :, j] - observed[:, :, j], n=order, axis=1)**2))) for order in (1, 2)],
            "coefficient_rmse": float(np.sqrt(np.mean((inverse @ (predicted[:, :, j] - observed[:, :, j]).T)**2))),
            "high_frequency_energy_fraction": float(physical_energy[:, 10:, j].sum() / physical_energy[:, :, j].sum()) if physical_energy[:, :, j].sum() else None,
        }
    return row


def residual_width(name, bits):
    if "hybrid" in name:
        return (8, 16, 24, 32)[2 * bits[15] + bits[16]]
    return None


def run(args):
    from Physics.Constraints import violations

    args.output.mkdir(parents=True, exist_ok=True)
    values, times = read_hourly(args.source)
    settings = TrainingConfig(epochs=args.epochs, seed=args.seed, hidden=(32, 16), device=args.device)
    cutoffs = ((args.validation_from, args.test_from) if args.validation_from is not None else
               calendar_cutoffs(times, settings.train_fraction, settings.validation_fraction))
    dataset = build_dataset(values, times, cutoffs=cutoffs)
    train, val = [rollout_windows(dataset, rows, values, times, HORIZON)
                  for rows in (dataset.train, dataset.validation)]
    if not len(train.starts) or not len(val.starts):
        raise ValueError("Training and validation each need a complete 72-hour window")
    weights = LossWeights(**read_json(args.loss_weights)) if args.loss_weights else LossWeights()
    source_hash = sha256(args.source.read_bytes()).hexdigest()
    provenance = {"prepared_sha256": source_hash, "validated_horizon_hours": HORIZON}
    if args.preparation_report:
        provenance = {**read_json(args.preparation_report)["provenance"], **provenance}
    reports = {}
    # Genome uses the existing 20-bit carrier, with different decoding for this experiment.
    architecture = {"hidden": settings.hidden}
    plans = {"fixed_fourier": (architecture, Modes(), [], None)}
    for name, damped, augmentation in (
        ("qiga_fourier", False, 0.), ("qiga_laplace", True, 0.),
        ("qiga_sparse", False, args.sparse_scale),
        ("qiga_hybrid", True, 0.),
        ("qiga_hybrid_sparse", True, args.sparse_scale),
    ):
        if ("hybrid" in name and not args.hybrid) or ("sparse" in name and not augmentation):
            continue

        def score(genome):
            arch, modes = decode(genome.bits, damped)
            model, history, epoch = fit(train, val, times, dataset.target_scaler, arch, modes,
                                        replace(settings, epochs=args.candidate_epochs), weights, augmentation, residual_width=residual_width(name, genome.bits),
                                        joint_epochs=args.joint_epochs)
            result = history["validation_loss"][epoch - 1]
            return Candidate(genome, result + .0001 * (model.model.parameter_count + (model.residual.parameter_count if model.residual else 0)) / 1000, result,
                             (model.model.parameter_count + (model.residual.parameter_count if model.residual else 0)), len(train.starts) * len(history["phase"]),
                             len(history["phase"]), training_seconds=history["training_seconds"],
                             details={"periods": list(modes.periods), "damped": modes.damped,
                                      "components": history["validation_components"][epoch - 1],
                                      "events": sum(history["event_counts"])})

        fingerprint = validation_fingerprint(dataset, settings, (train, val))
        recipe = repr((fingerprint, sha256(Path(__file__).read_bytes()).hexdigest(), name, augmentation, weights, args.joint_epochs))
        winner, _, generations = run_search("hybrid_amplitude", score,
            EvolutionConfig(population=args.population, generations=args.generations,
                            candidate_epochs=args.candidate_epochs), args.seed,
            args.output / f"{name}-search.json", sha256(recipe.encode()).hexdigest(),
            SearchOptions(sequence=True, adaptive_amplitude=True))
        plans[name] = (*decode(winner.genome.bits, damped), generations, winner.genome.bits)

    # All selection has completed before constructing or scoring test windows.
    test = rollout_windows(dataset, dataset.test, values, times, HORIZON)
    if not len(test.starts):
        raise ValueError("Test needs a complete 72-hour window")
    origins = times[test.starts - 1]
    target_times = times[test.starts[:, None] + np.arange(HORIZON)]
    persistence, daily, _ = baseline_forecasts(values, times, test)
    for name, (arch, modes, generations, bits) in plans.items():
        model, history, epoch = fit(train, val, times, dataset.target_scaler, arch, modes, settings,
                                    weights, args.sparse_scale if "sparse" in name else 0., diagnostics=True,
                                    residual_width=residual_width(name, bits), joint_epochs=args.joint_epochs)
        raw = model.forecast(test.histories, origins, HORIZON, raw=True)
        tick = perf_counter()
        prediction = model.forecast(test.histories, origins, HORIZON)
        latency = perf_counter() - tick
        basis = modes.anchored_basis(np.arange(1, HORIZON + 1))
        delta = test.targets - test.histories[:, -1:]
        oracle = test.histories[:, -1:] + basis @ (np.linalg.pinv(basis) @ delta)
        folder = args.output / name
        folder.mkdir(exist_ok=True)
        path = folder / "model.json"
        params = model.model.state_dict(model.backend)
        write_json(path, {"format": "jupiter-spectral-v1", "features": FEATURES,
            "interval_hours": 1, "dtype": settings.dtype, "seed": settings.seed,
            "architecture": {**arch, "dropout": settings.dropout}, "modes": asdict(modes),
            "residual": None if model.residual is None else {
                "width": model.residual.hidden[0], "parameters": model.residual.state_dict(model.backend)},
            "parameters": params, "inputs": model.inputs.state_dict(),
            "targets": model.targets.state_dict(), "provenance": provenance})
        report = {"best_epoch": epoch, "fit_diagnostics": history, "configuration": asdict(settings),
                  "modes": asdict(modes), "winning_bits": bits, "generations": generations,
                  "checkpoint_sha256": sha256(path.read_bytes()).hexdigest(),
                  "parameters": (model.model.parameter_count + (model.residual.parameter_count if model.residual else 0)), "test_inference_seconds": latency,
                  "test_rollout": horizon_metrics(prediction, test.targets, persistence),
                  "persistence": horizon_metrics(persistence, test.targets, persistence),
                  "seasonal_persistence": horizon_metrics(daily, test.targets, persistence),
                  "oracle_reconstruction": horizon_metrics(oracle, test.targets),
                  "patterns": pattern_metrics(prediction, test.targets, modes),
                  "trajectory_ranges": {feature: {
                      "constant_targets": int((np.ptp(test.targets[:, :, j], axis=1) <= 1e-8).sum()),
                      "constant_predictions": int((np.ptp(prediction[:, :, j], axis=1) <= 1e-8).sum()),
                      "target_zero_fraction": float((test.targets[:, :, j] == 0).mean()),
                      "raw_prediction_range": [float(raw[:, :, j].min()), float(raw[:, :, j].max())],
                  } for j, feature in enumerate(FEATURES)},
                  "raw_constraint_counts": {key: int(flag.sum()) for key, flag in violations(raw).items()}}
        diagnostics = folder / "diagnostics"
        publish(diagnostics, report, test.targets, prediction, persistence, daily, params, target_times)
        publish_evolution(diagnostics, generations)
        from SpectralDiagnostics import publish_spectral
        publish_spectral(diagnostics, model, test.histories, origins, test.targets, prediction, history)
        reports[name] = report
    references = {"daily_persistence": daily}
    if args.recursive_reference:
        reference = load_forecaster(args.recursive_reference)
        references["recursive_reference"] = np.stack([reference.forecast(x, HORIZON, origin=t)
                                                        for x, t in zip(test.histories, origins)])
    for name, predictions in references.items():
        reports[name] = {"test_rollout": horizon_metrics(predictions, test.targets, persistence),
                         "patterns": pattern_metrics(predictions, test.targets, Modes())}
    write_json(args.output / "report.json", {"cutoffs": [str(x) + "Z" for x in cutoffs],
        "origins": {name: len(window.starts) for name, window in zip(("train", "validation", "test"), (train, val, test))},
        "loss_weights": asdict(weights), "source_sha256": source_hash, "methods": reports})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preparation-report", type=Path)
    parser.add_argument("--recursive-reference", type=Path)
    parser.add_argument("--validation-from", type=utc_threshold)
    parser.add_argument("--test-from", type=utc_threshold)
    parser.add_argument("--loss-weights", type=Path)
    parser.add_argument("--population", type=int, default=2)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--candidate-epochs", type=int, default=1)
    parser.add_argument("--sparse-scale", type=float, default=0.)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--joint-epochs", type=int, default=1)
    args = parser.parse_args()
    if args.joint_epochs < 0:
        parser.error("Joint epochs must be nonnegative")
    if args.hybrid and args.loss_weights is None:
        args.loss_weights = Path(__file__).resolve().parent / "Examples/spectral/curve-loss.json"
    if ((args.validation_from is None) != (args.test_from is None)
            or (args.validation_from is not None and args.validation_from >= args.test_from)):
        parser.error("Supply both chronological boundaries in increasing order")
    if not np.isfinite(args.sparse_scale) or args.sparse_scale < 0:
        parser.error("Sparse scale must be finite and nonnegative")
    run(args)
