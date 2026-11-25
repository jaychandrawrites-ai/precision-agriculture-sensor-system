"""Compare experimental direct, recursive and hybrid forecasts on shared origins."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from config import TrainingConfig
from Data.Dataset import build_dataset
from Data.Preprocessing import read_hourly
from Data.RolloutWindows import rollout_windows
from Data.TimeSeries import calendar_cutoffs
from Framework.Backend import create_backend
from Temporal.Baselines import baseline_forecasts
from Temporal.Direct import fit_direct
from Temporal.ForecastUncertainty import calibrate, ensemble, evaluate_intervals
from Temporal.Model import TemporalModel
from Temporal.Rollout import Rollout, RolloutOptions
from Temporal.RolloutTrainer import fit_rollout
from Utils.Bootstrap import paired_rmse_difference
from Utils.Metrics import horizon_metrics
from Utils.Serialization import write_json


def compare(values, times, settings, horizon, seeds, mass=.9):
    if not seeds or horizon < 1:
        raise ValueError("Supply at least one seed and a positive horizon")
    cutoffs = calendar_cutoffs(times, settings.train_fraction, settings.validation_fraction)
    data = build_dataset(values, times, cutoffs=cutoffs)
    # Disjoint validation target periods for model selection and interval calibration.
    validation = data.validation
    midpoint = cutoffs[0] + (cutoffs[1] - cutoffs[0]) // 2
    selection = times[data.target_indices[validation]] < midpoint
    a, b = validation[selection], validation[~selection]
    groups = (data.train, a, b, data.test)
    direct_train, direct_val, calibration, test = (
        rollout_windows(data, rows, values, times, horizon) for rows in groups
    )
    if min(len(window.starts) for window in (direct_train, direct_val, calibration, test)) < 1:
        raise ValueError("Source is too short or incomplete for disjoint benchmark windows")
    rec_train, rec_val = (rollout_windows(data, rows, values, times, settings.rollout_steps)
                          for rows in (data.train, a))
    persistence, seasonal, _ = baseline_forecasts(values, times, test)
    members = {mode: [[], []] for mode in ("recursive", "direct", "hybrid")}
    diagnostics = []
    for seed in seeds:
        opts = replace(settings, seed=seed)
        backend = create_backend(opts.device, opts.dtype, seed)
        model = TemporalModel(backend, opts.hidden, opts.dropout)
        started = perf_counter()
        _, inputs, rec_diagnostics = fit_rollout(model, data, rec_train, rec_val, backend, opts,
                                                 RolloutOptions(), warmup_epochs=opts.epochs // 3)
        rec_seconds = perf_counter() - started
        engine = Rollout(model, inputs, data.target_scaler, backend, RolloutOptions())
        started = perf_counter()
        direct, direct_diagnostics = fit_direct(direct_train, direct_val, data.target_scaler,
                                                create_backend(opts.device, opts.dtype, seed),
                                                opts)
        diagnostics.append({"seed": seed, "direct": direct_diagnostics,
                            "recursive": {**rec_diagnostics, "parameters": model.parameter_count},
                            "recursive_fit_seconds": rec_seconds,
                            "direct_fit_seconds": perf_counter() - started,
                            "test_ood_scores": direct.ood_score(test.histories)})
        blend = np.minimum(1, np.arange(horizon) / max(1, opts.rollout_steps))[None, :, None]
        for i, window in enumerate((calibration, test)):
            recursive = engine.predict(window.histories, horizon)
            prediction = direct.predict(window.histories)
            for mode, value in (("recursive", recursive), ("direct", prediction),
                                ("hybrid", recursive * (1 - blend) + prediction * blend)):
                members[mode][i].append(value)
    results = {"persistence": horizon_metrics(persistence, test.targets, persistence),
               "seasonal_persistence": horizon_metrics(seasonal, test.targets, persistence)}
    uncertainty, comparisons = {}, {}
    contiguous = (np.diff(times[test.starts]) == np.timedelta64(1, "h")).all()
    block = max(24, horizon)
    for name, (cal_predictions, test_predictions) in members.items():
        cal_ensemble, _ = ensemble(cal_predictions)
        estimate, spread = ensemble(test_predictions)
        intervals = calibrate(cal_ensemble, calibration.targets, mass)
        results[name] = horizon_metrics(estimate, test.targets, persistence)
        if contiguous and len(test.starts) >= 2 * block:
            comparisons[name] = paired_rmse_difference(estimate, persistence, test.targets,
                                                       block=block, draws=200)
        uncertainty[name] = {**evaluate_intervals(intervals, estimate, test.targets),
                             "mean_member_spread": spread.mean(axis=0)}
    return {"windows": {name: len(window.starts) for name, window in zip(
                ("train", "validation", "calibration", "test"),
                (direct_train, direct_val, calibration, test))},
            "recursive_training_horizon": settings.rollout_steps, "test_horizon": horizon,
            "metrics": results, "diagnostics": diagnostics, "intervals": uncertainty,
            "rmse_difference_vs_persistence": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    settings = TrainingConfig(epochs=args.epochs, dropout=0)
    values, times = read_hourly(args.source)
    result = compare(values, times, settings, args.horizon, args.seeds)
    result["manifest"] = {"source_sha256": sha256(args.source.read_bytes()).hexdigest(),
                          "synthetic": args.synthetic, "training": asdict(settings), "seeds": args.seeds}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
