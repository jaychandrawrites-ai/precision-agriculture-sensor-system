"""Compare temporal architecture searches with an untouched chronological test split."""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from config import EvolutionConfig, TrainingConfig
from Data.Dataset import build_dataset
from Data.Preprocessing import read_hourly
from Data.RolloutWindows import rollout_windows
from Data.TimeSeries import calendar_cutoffs
from Evolution.Diagnostics import diversity, pareto_front, random_search
from Evolution.Fitness import validation_fitness
from Evolution.GeneticAlgorithm import search as genetic_search
from Evolution.Genome import Architecture
from Evolution.SequenceSearch import compare_sequences
from Evolution.Resumable import SearchOptions, run_search, validation_fingerprint
from Framework.Backend import create_backend
from QuantumInspired.QuantumEvolution import search as amplitude_search
from Temporal.Baselines import baseline_forecasts
from Temporal.Model import TemporalModel
from Temporal.Rollout import Rollout, RolloutOptions
from Temporal.RolloutTrainer import fit_rollout
from Temporal.Trainer import fit
from Temporal.Validator import evaluate_rollouts, evaluate_rows
from Utils.Metrics import horizon_metrics, physical_metrics


def compare(values, timestamps, settings, budget, *, recursive=False, include_random=False,
            include_hybrid=False, search_directory=None, search_options=SearchOptions()):
    cutoffs = (calendar_cutoffs(timestamps, settings.train_fraction, settings.validation_fraction)
               if recursive else None)
    dataset = build_dataset(values, timestamps, settings.train_fraction,
                            settings.validation_fraction, cutoffs=cutoffs)
    windows = None
    if recursive:
        windows = tuple(rollout_windows(dataset, rows, values, timestamps, settings.rollout_steps)
                        for rows in (dataset.train, dataset.validation))
    score = validation_fitness(dataset, settings, budget, rollout_windows_pair=windows,
                               parameter_count_limit=search_options.max_parameters)
    fixed = Architecture(tuple(settings.hidden), (), ("relu",) * len(settings.hidden))
    selections = {"fixed": (fixed, [], None, 0.0, [])}
    use_structured_search = search_directory is not None or search_options != SearchOptions() or include_hybrid
    if search_directory is not None:
        search_directory = Path(search_directory)
        search_directory.mkdir(exist_ok=True)
    if use_structured_search:
        fingerprint = validation_fingerprint(dataset, settings, windows)
        if settings.device != "cpu" and search_options.workers > 1:
            raise ValueError("Parallel candidate training is CPU-only")
    searches = {"classical_ga": genetic_search, "quantum_inspired": amplitude_search}
    if include_random:
        searches["random"] = random_search
    if include_hybrid:
        searches["hybrid_amplitude"] = None
    for name, search in searches.items():
        started = perf_counter()
        generation_report = []
        if not use_structured_search:
            winner, history = search(score, budget, seed=settings.seed)
        else:
            checkpoint = None if search_directory is None else search_directory / f"{name}.json"
            winner, history, generation_report = run_search(
                name, score, budget, settings.seed, checkpoint,
                fingerprint, search_options
            )
        candidates = [candidate for generation in history for candidate in generation]
        selections[name] = (winner.genome.decode(), candidates, winner, perf_counter() - started, generation_report)

    # Test windows and scores are only constructed once both selections are final.
    if recursive:
        test = rollout_windows(dataset, dataset.test, values, timestamps, settings.rollout_steps)
        one_step = rollout_windows(dataset, dataset.test, values, timestamps, 1)
        if not len(test.starts):
            raise ValueError("No complete held-out windows for the experiment horizon")
        persistence, seasonal, fallback = baseline_forecasts(values, timestamps, test)
    reports = {}
    for name, (architecture, candidates, winner, search_seconds, generation_report) in selections.items():
        backend = create_backend(settings.device, settings.dtype, settings.seed)
        model = TemporalModel(backend, dropout=settings.dropout, **architecture.model_arguments())
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
            **architecture.model_arguments(), "parameters": model.parameter_count,
            "candidate_count": len(candidates),
            "candidate_epoch_budget": len(candidates) * budget.candidate_epochs,
            "candidate_epochs_used": sum(item.epochs for item in candidates),
            "candidate_parameter_step_budget": (None if budget.candidate_parameter_step_limit is None
                                                 else len(candidates) * budget.candidate_parameter_step_limit),
            "candidate_model_sample_steps": sum(item.model_sample_steps for item in candidates),
            "candidate_parameter_sample_steps": sum(item.parameters * item.model_sample_steps
                                                     for item in candidates),
            "actual_candidate_evaluations": (sum(item["actual_evaluations"] for item in generation_report)
                                             if generation_report else len(candidates)),
            "actual_candidate_model_sample_steps": (
                sum(item["actual_training_sample_steps"] for item in generation_report)
                if generation_report else sum(item.model_sample_steps for item in candidates)
            ),
            "actual_candidate_parameter_sample_steps": (
                sum(item["actual_parameter_sample_steps"] for item in generation_report)
                if generation_report else sum(item.parameters * item.model_sample_steps for item in candidates)
            ),
            "selection_score": winner.score if winner is not None else None,
            "winning_bits": winner.genome.bits if winner is not None else None,
            "candidate_diversity": diversity(candidates),
            "generations": generation_report,
            "validation_pareto_front": [
                {"bits": item.genome.bits, **item.genome.decode().model_arguments(),
                 "validation_loss": item.validation_loss,
                 "parameters": item.parameters} for item in pareto_front(candidates)
            ],
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
                       "candidate_epochs": budget.candidate_epochs, "horizon": settings.rollout_steps,
                       "candidate_parameter_step_limit": budget.candidate_parameter_step_limit},
            "search_options": asdict(search_options),
            "fitness": "validation loss plus complexity penalty",
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
    parser.add_argument("--candidate-parameter-step-limit", type=int)
    parser.add_argument("--seed", type=int, default=TrainingConfig().seed)
    parser.add_argument("--search-directory", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cache-fitness", action="store_true")
    parser.add_argument("--random-baseline", action="store_true")
    parser.add_argument("--extended-space", action="store_true",
                        help="Search independent residual and activation choices")
    parser.add_argument("--adaptive-amplitude", action="store_true")
    parser.add_argument("--hybrid-baseline", action="store_true",
                        help="Compare classical offspring and adaptive amplitude sampling in one search")
    parser.add_argument("--max-parameters", type=int)
    parser.add_argument("--parent-selection", choices=("tournament", "rank"), default="tournament")
    parser.add_argument("--crossover", choices=("uniform", "single_point"), default="uniform")
    parser.add_argument("--sequence-space", action="store_true",
                        help="Compare all five methods on matched direct sequence windows")
    parser.add_argument("--export-selected", type=Path,
                        help="Save each retrained direct sequence winner as a forecast checkpoint")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Label the input as synthetic in the report")
    args = parser.parse_args()
    settings = replace(TrainingConfig(), epochs=args.epochs, seed=args.seed)
    budget = EvolutionConfig(population=args.population, generations=args.generations,
                             candidate_epochs=args.candidate_epochs,
                             candidate_parameter_step_limit=args.candidate_parameter_step_limit)
    source_hash = sha256(Path(args.source).read_bytes()).hexdigest()
    series = read_hourly(args.source)
    options = SearchOptions(workers=args.workers, cache=args.cache_fitness,
                            extended=args.extended_space, adaptive_amplitude=args.adaptive_amplitude,
                            max_parameters=args.max_parameters, parent_selection=args.parent_selection,
                            crossover=args.crossover)
    if args.sequence_space:
        if args.legacy:
            parser.error("Direct sequence search cannot use the legacy one-step protocol")
        report = compare_sequences(*series, settings, budget, options,
                                   search_directory=args.search_directory, export_directory=args.export_selected,
                                   provenance={"source_sha256": source_hash, "synthetic": args.synthetic})
    else:
        if args.export_selected is not None:
            parser.error("Saved winner export currently requires --sequence-space")
        report = compare(*series, settings, budget, recursive=not args.legacy,
                         include_random=args.random_baseline, include_hybrid=args.hybrid_baseline,
                         search_directory=args.search_directory, search_options=options)
    report["source_sha256"] = source_hash
    report["synthetic"] = args.synthetic
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
