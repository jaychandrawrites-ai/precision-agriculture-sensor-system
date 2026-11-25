"""Compare temporal architecture searches with an untouched chronological test split."""
import argparse
from dataclasses import replace
import json
from time import perf_counter

from config import EvolutionConfig, TrainingConfig
from Data.Dataset import build_dataset
from Data.Preprocessing import read_hourly
from Evolution.Fitness import validation_fitness
from Evolution.GeneticAlgorithm import search as genetic_search
from Framework.Backend import create_backend
from QuantumInspired.QuantumEvolution import search as amplitude_search
from Temporal.Model import TemporalModel
from Temporal.Trainer import fit
from Temporal.Validator import evaluate_rollouts, evaluate_rows
from train import _physical_metrics


def compare(values, timestamps, settings, budget):
    dataset = build_dataset(values, timestamps, settings.train_fraction,
                            settings.validation_fraction)
    score = validation_fitness(dataset, settings, budget)
    reports = {}
    for name, search in (("fixed", None), ("classical_ga", genetic_search),
                         ("quantum_inspired", amplitude_search)):
        started = perf_counter()
        hidden, skips = settings.hidden, ()
        candidate_count = 0
        if search is not None:
            winner, history = search(score, budget, seed=settings.seed)
            hidden, skips = winner.genome.architecture()
            candidate_count = sum(map(len, history))
        search_seconds = perf_counter() - started
        backend = create_backend(settings.device, settings.dtype, settings.seed)
        model = TemporalModel(backend, hidden, settings.dropout, skips)
        started = perf_counter()
        result = fit(model, dataset, backend, settings)
        fit_seconds = perf_counter() - started
        one_step = evaluate_rows(model, dataset, dataset.test, backend)
        rollout = evaluate_rollouts(model, dataset, dataset.test, values,
                                    timestamps, backend, settings.rollout_steps)
        reports[name] = {
            "hidden": hidden, "skips": skips, "parameters": model.parameter_count,
            "candidate_count": candidate_count,
            "candidate_epoch_budget": candidate_count * budget.candidate_epochs,
            "search_seconds": search_seconds, "final_fit_seconds": fit_seconds,
            "final_fit_epochs": settings.epochs, "validation_loss": result.validation_loss,
            "test": _physical_metrics(one_step),
            "test_rollout": [_physical_metrics(horizon) for horizon in rollout],
        }
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--epochs", type=int, default=TrainingConfig().epochs)
    parser.add_argument("--population", type=int, default=EvolutionConfig().population)
    parser.add_argument("--generations", type=int, default=EvolutionConfig().generations)
    parser.add_argument("--candidate-epochs", type=int, default=EvolutionConfig().candidate_epochs)
    args = parser.parse_args()
    settings = replace(TrainingConfig(), epochs=args.epochs)
    budget = EvolutionConfig(population=args.population, generations=args.generations,
                             candidate_epochs=args.candidate_epochs)
    print(json.dumps(compare(*read_hourly(args.source), settings, budget), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
