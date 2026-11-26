"""Matched-budget direct sequence search using common eligible rolling windows."""

from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from Data.Dataset import build_dataset
from Data.RolloutWindows import rollout_windows
from Data.TimeSeries import calendar_cutoffs
from Evolution.Fitness import sequence_fitness
from Evolution.Genome import MAX_SEQUENCE_CONTEXT
from Evolution.Resumable import run_search, validation_fingerprint
from Framework.Backend import create_backend
from Temporal.Baselines import baseline_forecasts
from Temporal.Direct import fit_direct
from Temporal.DirectCheckpoint import save_direct
from Temporal.SequenceForecast import SequenceSpec
from Utils.Metrics import horizon_metrics


METHODS = ("random", "classical_ga", "quantum_inspired", "hybrid_amplitude")


def _front(candidates):
    # Latency is measured and reported, but noisy wall-clock observations do not steer search.
    def objectives(item):
        return (item.validation_loss, item.parameters, item.parameters * item.model_sample_steps,
                item.raw_invalid_fraction)

    vectors = [objectives(item) for item in candidates]
    keep = []
    seen = set()
    for item, vector in zip(candidates, vectors):
        if vector not in seen and not any(all(a <= b for a, b in zip(other, vector)) and other != vector
                                           for other in vectors):
            seen.add(vector)
            keep.append(item)
    return keep


def _classification(candidate):
    return {"genome": candidate.genome.bits, "architecture": candidate.genome.decode().as_dict(),
            "selection_score": candidate.score, "validation_loss": candidate.validation_loss,
            "parameter_count": candidate.parameters, "sample_steps": candidate.model_sample_steps,
            "parameter_sample_steps": candidate.model_sample_steps * candidate.parameters,
            "training_seconds": candidate.training_seconds,
            "prediction_batch_seconds": candidate.prediction_batch_seconds,
            "prediction_batch_size": candidate.prediction_batch_size,
            "raw_invalid_fraction": candidate.raw_invalid_fraction}


def compare_sequences(values, timestamps, settings, budget, options, *, search_directory=None,
                      export_directory=None, provenance=None):
    if settings.device != "cpu" and options.workers > 1:
        raise ValueError("Parallel sequence candidates require CPU workers")
    options = replace(options, sequence=True)
    if search_directory is not None:
        search_directory = Path(search_directory)
        search_directory.mkdir(parents=True, exist_ok=True)
    if export_directory is not None:
        export_directory = Path(export_directory)
        export_directory.mkdir(parents=True, exist_ok=True)

    horizon = settings.rollout_steps
    cuts = calendar_cutoffs(timestamps, settings.train_fraction, settings.validation_fraction)
    dataset = build_dataset(values, timestamps, settings.train_fraction,
                            settings.validation_fraction, cutoffs=cuts)
    windows = tuple(rollout_windows(dataset, rows, values, timestamps, horizon, context=MAX_SEQUENCE_CONTEXT)
                    for rows in (dataset.train, dataset.validation))
    scorer = sequence_fitness(dataset, settings, budget, windows, parameter_count_limit=options.max_parameters)
    fingerprint = validation_fingerprint(dataset, settings, windows)
    searches = {}
    for method in METHODS:
        checkpoint = None if search_directory is None else search_directory / f"{method}.json"
        searches[method] = run_search(method, scorer, budget, settings.seed, checkpoint, fingerprint, options)

    # Values in the test interval are first used here, after selecting all winners.
    test = rollout_windows(dataset, dataset.test, values, timestamps, horizon, context=MAX_SEQUENCE_CONTEXT)
    if not len(test.starts):
        raise ValueError("No complete held-out windows at the requested matched context")
    persistence, seasonal, fallback = baseline_forecasts(values, timestamps, test)
    report = {"protocol": {"seed": settings.seed, "settings": asdict(settings), "budget": asdict(budget),
                           "options": asdict(options), "effective_parameter_cap": scorer.max_parameters,
                           "context_hours_required_for_matched_windows": MAX_SEQUENCE_CONTEXT,
                           "training_windows": len(windows[0].starts), "validation_windows": len(windows[1].starts),
                           "test_windows": len(test.starts), "output_horizon_hours": horizon,
                           "evaluation_batch_capacity": 16, "memory_measurement": None,
                           "work_convention": "training windows times forecast leads times epochs",
                           "cutoffs_utc": [str(cut) for cut in cuts]},
              "baselines": {"persistence": horizon_metrics(persistence, test.targets, persistence),
                            "seasonal": horizon_metrics(seasonal, test.targets, persistence),
                            "seasonal_fallbacks": fallback.tolist()}}
    fixed = SequenceSpec("dense", 8, activation="relu", residual=False, heads=1)
    for method in ("fixed", *METHODS):
        winner, history, generations = searches.get(method, (None, (), ()))
        spec = winner.genome.decode() if winner is not None else fixed
        backend = create_backend(settings.device, settings.dtype, settings.seed)
        started = perf_counter()
        model, diagnostics = fit_direct(*windows, dataset.target_scaler, backend, settings,
                                        width=spec.width, architecture=spec)
        fit_seconds = perf_counter() - started
        predicted = model.predict(test.histories)
        candidates = [item for group in history for item in group]
        report[method] = {
            "architecture": spec.as_dict(), "search_winner": _classification(winner) if winner else None,
            "candidate_count": len(candidates), "candidate_parameter_count": spec.parameter_count(horizon),
            "actual_evaluations": sum(row["actual_evaluations"] for row in generations),
            "actual_sample_steps": sum(row["actual_training_sample_steps"] for row in generations),
            "actual_parameter_sample_steps": sum(row["actual_parameter_sample_steps"] for row in generations),
            "candidate_families": dict(Counter(item.genome.decode().family for item in candidates)),
            "generations": generations, "validation_pareto_front": [_classification(item) for item in _front(candidates)],
            "final_fit_seconds": fit_seconds, "final_fit_sample_steps": settings.epochs * len(windows[0].starts) * horizon,
            "final_fit": diagnostics, "test_rollout": horizon_metrics(predicted, test.targets, persistence),
        }
        if export_directory is not None:
            name = f"{method}.json"
            save_direct(export_directory / name, model, spec.width,
                        provenance={"search_method": method, "architecture": spec.as_dict(),
                                    "seed": settings.seed, "training": asdict(settings), "budget": asdict(budget),
                                    "validation_fingerprint": fingerprint,
                                    "winner": _classification(winner) if winner else None, **(provenance or {})})
            report[method]["saved_model"] = name
    return report
