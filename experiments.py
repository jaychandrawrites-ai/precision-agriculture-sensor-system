"""Compare temporal architecture searches with an untouched chronological test split."""
import argparse
from dataclasses import replace
import json
from time import perf_counter

from config import EvolutionConfig, TrainingConfig
from Data.Dataset import build_dataset
from Data.Preprocessing import read_hourly
from Data.RolloutWindows import rollout_windows
from Data.TimeSeries import calendar_cutoffs
from Evolution.Fitness import validation_fitness
from Evolution.GeneticAlgorithm import search as genetic_search
from Framework.Backend import create_backend
from QuantumInspired.QuantumEvolution import search as amplitude_search
from Temporal.Baselines import baseline_forecasts
from Temporal.Model import TemporalModel
from Temporal.Rollout import Rollout, RolloutOptions
from Temporal.RolloutTrainer import fit_rollout
from Temporal.Trainer import fit
from Temporal.Validator import evaluate_rollouts, evaluate_rows
from Utils.Metrics import horizon_metrics, physical_metrics


def compare(values, timestamps, settings, budget, *, recursive=False):
    cutoffs = (calendar_cutoffs(timestamps, settings.train_fraction, settings.validation_fraction)
               if recursive else None)
    dataset = build_dataset(values, timestamps, settings.train_fraction,
                            settings.validation_fraction, cutoffs=cutoffs)
    windows = None
    if recursive:
        windows = tuple(rollout_windows(dataset, rows, values, timestamps, settings.rollout_steps)
                        for rows in (dataset.train, dataset.validation))
    score = validation_fitness(dataset, settings, budget, rollout_windows_pair=windows)
    selections = {"fixed": ((settings.hidden, ()), [], None, 0.0)}
    for name, search in (("classical_ga", genetic_search), ("quantum_inspired", amplitude_search)):
        started = perf_counter()
        winner, history = search(score, budget, seed=settings.seed)
        candidates = [candidate for generation in history for candidate in generation]
        selections[name] = (winner.genome.architecture(), candidates, winner, perf_counter() - started)

    # Test windows and scores are only constructed once both selections are final.
    if recursive:
        test = rollout_windows(dataset, dataset.test, values, timestamps, settings.rollout_steps)
        one_step = rollout_windows(dataset, dataset.test, values, timestamps, 1)
        if not len(test.starts):
            raise ValueError("No complete held-out windows for the experiment horizon")
        persistence, seasonal, fallback = baseline_forecasts(values, timestamps, test)
    reports = {}
    for name, ((hidden, skips), candidates, winner, search_seconds) in selections.items():
        backend = create_backend(settings.device, settings.dtype, settings.seed)
        model = TemporalModel(backend, hidden, settings.dropout, skips)
        started = perf_counter()
        if recursive:
            result, inputs, diagnostics = fit_rollout(
                model, dataset, *windows, backend, settings, RolloutOptions(),
                warmup_epochs=settings.epochs // 3,
            )
        else:
            result = fit(model, dataset, backend, settings)
        fit_seconds = perf_counter() - started
        report = {
            "hidden": hidden, "skips": skips, "parameters": model.parameter_count,
            "candidate_count": len(candidates),
            "candidate_epoch_budget": len(candidates) * budget.candidate_epochs,
            "candidate_model_sample_steps": sum(item.model_sample_steps for item in candidates),
            "candidate_parameter_sample_steps": sum(item.parameters * item.model_sample_steps
                                                     for item in candidates),
            "selection_score": winner.score if winner is not None else None,
            "winning_bits": winner.genome.bits if winner is not None else None,
            "search_seconds": search_seconds, "final_fit_seconds": fit_seconds,
            "final_fit_epochs": settings.epochs, "validation_loss": result.validation_loss,
        }
        if recursive:
            engine = Rollout(model, inputs, dataset.target_scaler, backend, RolloutOptions())
            predictions = engine.predict(test.histories, settings.rollout_steps)
            single = engine.predict(one_step.histories, 1)
            report.update({
                "test": horizon_metrics(single, one_step.targets, one_step.histories[:, -1:])[0],
                "test_rollout": horizon_metrics(predictions, test.targets, persistence),
                "origin_count": len(test.starts), "fit_diagnostics": diagnostics,
            })
        else:
            report["test"] = physical_metrics(evaluate_rows(model, dataset, dataset.test, backend))
            report["test_rollout"] = [
                physical_metrics(item) for item in evaluate_rollouts(
                    model, dataset, dataset.test, values, timestamps, backend, settings.rollout_steps
                )
            ]
        reports[name] = report
    if recursive:
        reports["protocol"] = {
            "seed": settings.seed, "calendar_cutoffs_utc": [str(cutoff) + "Z" for cutoff in cutoffs],
            "budget": {"population": budget.population, "generations": budget.generations,
                       "candidate_epochs": budget.candidate_epochs, "horizon": settings.rollout_steps},
        }
        reports["baselines"] = {
            "persistence": horizon_metrics(persistence, test.targets, persistence),
            "seasonal_persistence": horizon_metrics(seasonal, test.targets, persistence),
            "seasonal_fallbacks": fallback.tolist(),
        }
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--epochs", type=int, default=TrainingConfig().epochs)
    parser.add_argument("--population", type=int, default=EvolutionConfig().population)
    parser.add_argument("--generations", type=int, default=EvolutionConfig().generations)
    parser.add_argument("--candidate-epochs", type=int, default=EvolutionConfig().candidate_epochs)
    parser.add_argument("--seed", type=int, default=TrainingConfig().seed)
    parser.add_argument("--legacy", action="store_true")
    args = parser.parse_args()
    settings = replace(TrainingConfig(), epochs=args.epochs, seed=args.seed)
    budget = EvolutionConfig(population=args.population, generations=args.generations,
                             candidate_epochs=args.candidate_epochs)
    report = compare(*read_hourly(args.source), settings, budget, recursive=not args.legacy)
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
