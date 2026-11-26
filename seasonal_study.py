"""Evaluate causal seasonal neural corrections against daily and weekly references."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from config import FEATURES, TrainingConfig
from Data.Dataset import build_dataset
from Data.Preprocessing import read_hourly
from Data.RolloutWindows import rollout_windows
from Framework.Backend import create_backend
from Temporal.BlockIntervals import BlockIntervals, separated_indices
from Temporal.Direct import fit_direct
from Temporal.SeasonalCheckpoint import save_ensemble, save_member
from Temporal.Seasonality import ConvexCorrection, DailyClimatology, HarmonicRegression, SeasonalLookup
from Utils.Bootstrap import seasonal_temperature_comparison
from Utils.Metrics import horizon_metrics
from Utils.Serialization import atomic_output, write_json


def utc_threshold(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None or stamp.floor("h") != stamp:
        raise ValueError("Specify UTC-offset boundaries on the hour")
    return stamp.tz_convert("UTC").to_datetime64().astype("datetime64[ns]")


def assemble(sources):
    frames = [read_hourly(path) for path in sources]
    values = np.concatenate([frame[0] for frame in frames])
    times = np.concatenate([frame[1] for frame in frames])
    permutation = times.argsort()
    if (np.diff(times[permutation]) <= np.timedelta64(0, "ns")).any():
        raise ValueError("Source years must have distinct timestamps from a single compatible station")
    return values[permutation], times[permutation]


def _windows(values, times, cutoffs, horizon):
    fit_end, calibration_start, final_start = np.asarray(cutoffs, dtype="datetime64[ns]")
    if not times[0] < fit_end < calibration_start < final_start < times[-1]:
        raise ValueError("Four nonempty chronological periods are required")
    dataset = build_dataset(values, times, cutoffs=(fit_end, final_start))
    selection_rows = dataset.validation
    choose = times[dataset.target_indices[selection_rows]] < calibration_start
    rows = (dataset.train, selection_rows[choose], selection_rows[~choose], dataset.test)
    windows = [rollout_windows(dataset, index, values, times, horizon) for index in rows]
    if not all(len(window.starts) for window in windows):
        raise ValueError("Every period needs a complete window at the requested horizon")
    return dataset, windows


def _baselines(values, times, windows, fit_end, horizon):
    lookup = SeasonalLookup(values, times)
    train = times < fit_end
    climate = DailyClimatology.fit(values[train], times[train])
    harmonic = HarmonicRegression.fit(values[train], times[train])
    results, fallbacks = [], []
    leads = np.arange(1, horizon + 1) * np.timedelta64(1, "h")
    for frame in windows:
        origins = times[frame.starts - 1]
        valid = origins[:, None] + leads
        daily, daily_fallback = lookup.predict(origins, frame.histories, horizon, 24)
        weekly, weekly_fallback = lookup.predict(origins, frame.histories, horizon, 168)
        results.append({
            "persistence": np.broadcast_to(frame.histories[:, -1:, :], frame.targets.shape),
            "daily": daily, "weekly": weekly, "climatology": climate.predict(valid),
            "harmonic": harmonic.predict(valid),
        })
        fallbacks.append({"daily": daily_fallback.sum(axis=0), "weekly": weekly_fallback.sum(axis=0)})
    return results, fallbacks, harmonic.condition


def compare(values, times, cutoffs, settings, seeds, *, horizon=72, width=32, nominal=.9, model_output=None):
    if not seeds or not all(type(seed) is int and seed >= 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Provide distinct ensemble seeds")
    dataset, windows = _windows(values, times, cutoffs, horizon)
    train, select, calibrate, test = windows
    base, fallbacks, condition = _baselines(values, times, windows, cutoffs[0], horizon)
    origin_cal = times[calibrate.starts - 1]
    origin_test = times[test.starts - 1]
    calibration_count = len(separated_indices(origin_cal, horizon))
    if not 0 < nominal < 1 or np.ceil((calibration_count + 1) * nominal) > calibration_count:
        raise ValueError("Calibration span is too short for non-overlapping windows at this horizon")
    # Weighted sum is linear and avoids keeping every ensemble member's predictions in RAM.
    sums, squares = {}, {}
    runtimes, diagnostics = {}, {}
    if model_output is not None:
        model_output = Path(model_output)
        model_output.mkdir()
    member_files = []
    for method in ("direct", "daily_residual"):
        runtimes[method], diagnostics[method] = [], []
        for seed in seeds:
            begun = perf_counter()
            anchored = method == "daily_residual"
            config = replace(settings, seed=seed)
            fit, detail = fit_direct(
                train, select, dataset.target_scaler,
                create_backend(config.device, config.dtype, seed), config, width=width,
                training_anchor=base[0]["daily"] if anchored else None,
                validation_anchor=base[1]["daily"] if anchored else None,
            )
            if anchored and model_output is not None:
                member_files.append(save_member(model_output, seed, fit, width))
            runtimes[method].append(perf_counter() - begun)
            diagnostics[method].append(detail)
            for i, window in enumerate(windows[1:], start=1):
                result = fit.predict(
                    window.histories, anchor=base[i]["daily"] if anchored else None
                ).astype(float)
                key = (method, i)
                if key not in sums:
                    sums[key] = result
                    squares[key] = result**2
                else:
                    sums[key] += result
                    squares[key] += result**2
    for (method, index), total in sums.items():
        base[index][method] = total / len(seeds)

    selection = base[1]
    blend = ConvexCorrection.fit(selection["daily"], selection["daily_residual"], select.targets)
    for index in (1, 2, 3):
        run = base[index]
        run["validation_blend"] = blend.predict(run["daily"], run["daily_residual"])
    scores, uncertainty = {}, {}
    for method, predictions in base[3].items():
        scores[method] = {
            "against_persistence": horizon_metrics(predictions, test.targets, base[3]["persistence"]),
            "against_daily": horizon_metrics(predictions, test.targets, base[3]["daily"]),
        }
        if method in ("daily", "direct", "daily_residual", "validation_blend"):
            interval = BlockIntervals.fit(base[2][method], calibrate.targets, origin_cal, nominal)
            uncertainty[method] = interval.score(predictions, test.targets, origin_test)
            if method == "validation_blend" and model_output is not None:
                save_ensemble(model_output, member_files, blend, interval, horizon)
        if method in diagnostics:
            variance = np.maximum(squares[method, 3] / len(seeds) - predictions**2, 0)
            uncertainty[method]["mean_member_spread"] = np.sqrt(variance).mean(axis=0)
    split_names = ("training", "selection", "calibration", "final_test")
    summary = {
        name: {"origins": len(window.starts), "first_origin": str(times[window.starts[0] - 1]) + "Z",
               "last_origin": str(times[window.starts[-1] - 1]) + "Z",
               "potential_windows": window.potential, "seasonal_fallbacks_per_lead": miss}
        for name, window, miss in zip(split_names, windows, fallbacks)
    }
    return {"splits": summary, "horizon": horizon, "settings": asdict(settings),
            "seeds": seeds, "harmonic_design_condition": condition, "validation_blend_weights": blend.weight,
            "fit_seconds": runtimes, "fit_diagnostics": diagnostics, "scores": scores,
            "uncertainty": uncertainty,
            "paired_calendar_comparison": seasonal_temperature_comparison(
                base[3]["validation_blend"], base[3]["daily"], test.targets, origin_test
            ),
            "inference_history": {"direct_hours": 3, "daily_lookup_hours": 24, "weekly_lookup_hours": 168,
                                  "missing_seasonal_slot": "fallback_to_current_state"}}


def export_metrics(path, report):
    rows = []
    for method, comparisons in report["scores"].items():
        for lead, metrics in enumerate(comparisons["against_persistence"], start=1):
            for feature in FEATURES:
                rows.append({"method": method, "lead_hours": lead, "feature": feature, **metrics[feature],
                             "seasonal_skill": comparisons["against_daily"][lead - 1][feature]["persistence_skill"]})
    with atomic_output(path) as handle:
        pd.DataFrame(rows).to_csv(handle, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--selection-from", required=True, type=utc_threshold)
    parser.add_argument("--calibration-from", required=True, type=utc_threshold)
    parser.add_argument("--final-from", required=True, type=utc_threshold)
    parser.add_argument("--horizon", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--nominal", type=float, default=.9)
    parser.add_argument("--station", required=True, help="One station identifier or an explicit synthetic label")
    args = parser.parse_args()
    values, times = assemble(args.sources)
    cutoffs = [args.selection_from, args.calibration_from, args.final_from]
    report = compare(values, times, cutoffs, TrainingConfig(epochs=args.epochs, dropout=0),
                     args.seeds, horizon=args.horizon, width=args.width, nominal=args.nominal, model_output=args.model_output)
    report["manifest"] = {"station": args.station, "sources": [
        {"name": source.name, "sha256": sha256(source.read_bytes()).hexdigest()} for source in args.sources
    ], "utc_boundaries": [str(value) + "Z" for value in cutoffs]}
    write_json(args.output, report)
    export_metrics(args.output.with_suffix(".metrics.csv"), report)


if __name__ == "__main__":
    main()
