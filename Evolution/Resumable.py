"""Generation-boundary recovery for reproducible validation-only search."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import sys
from pathlib import Path

import numpy as np

from Evolution.Diagnostics import diversity
from Evolution.Fitness import Candidate
from Evolution.Genome import Architecture, Genome, WIDTHS
from QuantumInspired.Adaptive import adaptive_update, amplitude_diagnostics
from QuantumInspired.QuantumGenome import QuantumGenome
from Utils.Serialization import read_json, write_json


@dataclass(frozen=True)
class SearchOptions:
    workers: int = 1
    cache: bool = False
    patience: int | None = None
    elite: int = 1
    tournament: int = 3
    extended: bool = False
    sequence: bool = False
    adaptive_amplitude: bool = False
    hybrid_fraction: float = 0.25
    entropy_floor: float = 0.35
    max_parameters: int | None = None
    parent_selection: str = "tournament"
    crossover: str = "uniform"

    def __post_init__(self):
        if min(self.workers, self.elite, self.tournament) < 1 or (
            self.patience is not None and self.patience < 1
        ):
            raise ValueError("Invalid search settings")
        minimum = Architecture((min(WIDTHS),), (), ("relu",)).parameter_count
        if self.max_parameters is not None and (
            type(self.max_parameters) is not int or self.max_parameters < minimum
        ):
            raise ValueError(f"Parameter cap must be at least {minimum}")
        if not 0 < self.hybrid_fraction < 1 or not 0 <= self.entropy_floor <= 1:
            raise ValueError("Invalid hybrid fraction or entropy floor")
        if self.parent_selection not in ("tournament", "rank") or self.crossover not in (
            "uniform", "single_point"
        ):
            raise ValueError("Invalid selection or crossover operator")


def validation_fingerprint(dataset, settings, windows):
    """Hash only the data available to architecture selection."""
    digest = sha256(json.dumps(asdict(settings), sort_keys=True).encode())
    root = Path(__file__).resolve().parents[1]
    digest.update(np.__version__.encode())
    digest.update(sys.version.encode())
    for directory in ("Data", "Evolution", "Framework", "Physics", "QuantumInspired", "Temporal"):
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    if windows is None:
        arrays = [dataset.inputs[dataset.train], dataset.targets[dataset.train],
                  dataset.inputs[dataset.validation], dataset.targets[dataset.validation],
                  dataset.weights[dataset.train]]
    else:
        arrays = [array for window in windows for array in (window.histories, window.targets)]
        arrays.extend([dataset.inputs[dataset.train], dataset.targets[dataset.train]])
    for data in arrays:
        array = np.ascontiguousarray(data, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _pack(candidate):
    data = asdict(candidate)
    del data["genome"]
    return {"bits": candidate.genome.bits, **data}


def _unpack(item):
    values = dict(item)
    genome = Genome(tuple(values.pop("bits")))
    return Candidate(genome, **values)


def _key(genome):
    return str(genome.decode())


def _breed(ranked, rng, config, options, *, size, preserve_elite=True):
    elite_count = min(options.elite, size) if preserve_elite else 0
    elite = [item.genome for item in ranked[:elite_count]]
    mutation = min(.5, config.mutation_probability * (1.5 - diversity(ranked)))
    rank_weights = np.arange(len(ranked), 0, -1, dtype=float)
    rank_weights /= rank_weights.sum()

    def select_parent():
        if options.parent_selection == "rank":
            return ranked[rng.choice(len(ranked), p=rank_weights)].genome
        indices = rng.integers(len(ranked), size=options.tournament)
        return ranked[min(indices)].genome

    pool = elite.copy()
    while len(pool) < size:
        left, right = select_parent(), select_parent()
        if options.crossover == "single_point":
            child = left.offspring(right, rng, mutation)
        else:
            mask = rng.random(len(left.bits)) < .5
            bits = np.where(mask, left.bits, right.bits)
            bits ^= rng.random(len(bits)) < mutation
            child = Genome(tuple(map(int, bits)))
        pool.append(child)
    return pool, mutation


def run_search(mode, score, settings, seed, checkpoint, fingerprint, options=SearchOptions(),
               *, stop_after=None):
    """Checkpoint completed generations. A retried partial generation is recomputed.

    Scores are evaluated in fixed order even with parallel workers. Cache hits
    count as logical evaluations, with actual work and cache hits reported
    separately. Disable caching when matching physical evaluation counts.
    """
    if mode not in ("classical_ga", "quantum_inspired", "random", "hybrid_amplitude") or not fingerprint:
        raise ValueError("Unknown mode or missing validation fingerprint")
    amplitude_mode = mode in ("quantum_inspired", "hybrid_amplitude")
    rng = np.random.default_rng(seed)
    pool = []
    amplitudes = []
    history = []
    diagnostics = []
    cache = {}
    best = None
    stale = 0
    checkpoint = Path(checkpoint) if checkpoint is not None else None
    recipe = {"version": 2, "mode": mode, "seed": seed, "data": fingerprint,
              "evolution": asdict(settings), "options": asdict(options)}
    if checkpoint is not None and checkpoint.exists():
        state = read_json(checkpoint)
        if recipe != state["recipe"]:
            raise ValueError("Search checkpoint recipe, data, or settings differ")
        rng.bit_generator.state = state["rng"]
        history = [[_unpack(candidate) for candidate in generation] for generation in state["history"]]
        diagnostics = state["diagnostics"]
        cache = {name: _unpack(candidate) for name, candidate in state["cache"].items()}
        pool = [Genome(tuple(bits)) for bits in state["population"]]
        amplitudes = [QuantumGenome(extended=options.extended, sequence=options.sequence, angles=angles)
                      for angles in state["angles"]]
        best = min((item for generation in history for item in generation), key=lambda item: item.score)
        stale = state["stale"]
    elif amplitude_mode:
        amplitudes = [QuantumGenome(extended=options.extended, sequence=options.sequence) for _ in range(settings.population)]
    elif mode == "classical_ga":
        pool = [Genome.random(rng, extended=options.extended, sequence=options.sequence) for _ in range(settings.population)]

    options_horizon = getattr(score, "horizon", 1)
    options_cap = getattr(score, "max_parameters", options.max_parameters)
    stop = settings.generations if stop_after is None else min(stop_after, settings.generations)
    if stop < 1:
        raise ValueError("Generation limit must permit evaluation")
    with ThreadPoolExecutor(max_workers=options.workers) as workers:
        for generation in range(len(history), stop):
            if options.patience is not None and stale >= options.patience:
                break
            hybrid_mutation = None
            measured = []
            if amplitude_mode:
                injected = 0
                if mode == "hybrid_amplitude" and history:
                    injected = min(settings.population - 1,
                                   max(1, int(settings.population * options.hybrid_fraction)))
                measured = list(range(len(amplitudes)))
                if injected:
                    excluded = set(rng.choice(len(amplitudes), size=injected, replace=False))
                    measured = [slot for slot in measured if slot not in excluded]
                pool = [amplitudes[slot].measure(rng) for slot in measured]
                if injected:
                    offspring, hybrid_mutation = _breed(history[-1], rng, settings, options,
                                                        size=injected, preserve_elite=False)
                    pool.extend(offspring)
            elif mode == "random":
                pool = [Genome.random(rng, extended=options.extended, sequence=options.sequence) for _ in range(settings.population)]
            repaired = [genome.repair(options_cap, horizon=options_horizon) for genome in pool]
            repairs = sum(before != after for before, after in zip(pool, repaired))
            pool = repaired

            if options.cache:
                pending = {}
                for genome in pool:
                    key = _key(genome)
                    if key not in cache and key not in pending:
                        pending[key] = genome
                actual = list(workers.map(score, pending.values()))
                cache.update(zip(pending, actual))
                candidates = [replace(hit, genome=genome)
                              for genome in pool for hit in (cache[_key(genome)],)]
            else:
                actual = candidates = list(workers.map(score, pool))
            ranked = sorted(candidates, key=lambda item: item.score)
            history.append(ranked)
            if best is None or ranked[0].score < best.score:
                best = ranked[0]
                stale = 0
            else:
                stale += 1
            generation_metrics = {
                "generation": generation + 1, "best_selection_score": best.score,
                "diversity": diversity(ranked), "actual_evaluations": len(actual),
                "cache_hits": len(pool) - len(actual),
                "actual_training_sample_steps": sum(item.model_sample_steps for item in actual),
                "actual_parameter_sample_steps": sum(item.model_sample_steps * item.parameters
                                                     for item in actual),
                "repaired_candidates": repairs,
            }
            if mode == "classical_ga":
                pool, mutation = _breed(ranked, rng, settings, options, size=settings.population)
                generation_metrics["breeding_mutation_probability"] = mutation
            elif amplitude_mode:
                sampled = [(slot, candidates[i]) for i, slot in enumerate(measured)]
                if options.adaptive_amplitude or mode == "hybrid_amplitude":
                    generation_metrics.update(adaptive_update(
                        amplitudes, sampled, candidates, best.genome, settings.rotation_radians,
                        generation / max(1, settings.generations - 1), options.entropy_floor
                    ))
                else:
                    for slot, candidate in sampled:
                        amplitudes[slot].rotate(candidate.genome, best.genome, settings.rotation_radians)
                    generation_metrics.update(amplitude_diagnostics(amplitudes))
                generation_metrics["amplitude_samples"] = len(measured)
                generation_metrics["classical_offspring"] = len(pool) - len(measured)
                if hybrid_mutation is not None:
                    generation_metrics["breeding_mutation_probability"] = hybrid_mutation
            diagnostics.append(generation_metrics)
            state = {"recipe": recipe, "rng": rng.bit_generator.state,
                     "history": [[_pack(item) for item in group] for group in history],
                     "diagnostics": diagnostics, "stale": stale,
                     "cache": {name: _pack(item) for name, item in cache.items()},
                     "population": [item.bits for item in pool],
                     "angles": [item.angles for item in amplitudes]}
            if checkpoint is not None:
                write_json(checkpoint, state, indent=None)
    return best, history, diagnostics
