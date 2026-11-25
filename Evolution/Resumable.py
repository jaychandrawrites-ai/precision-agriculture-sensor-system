"""Generation-boundary recovery for reproducible validation-only search."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from Evolution.Diagnostics import diversity
from Evolution.Fitness import Candidate
from Evolution.Genome import Genome
from QuantumInspired.QuantumGenome import QuantumGenome
from Utils.Serialization import read_json, write_json


@dataclass(frozen=True)
class SearchOptions:
    workers: int = 1
    cache: bool = False
    patience: int | None = None
    elite: int = 1
    tournament: int = 3

    def __post_init__(self):
        if min(self.workers, self.elite, self.tournament) < 1 or (
            self.patience is not None and self.patience < 1
        ):
            raise ValueError("Invalid search settings")


def validation_fingerprint(dataset, settings, windows):
    """Hash only the data available to architecture selection."""
    digest = sha256(json.dumps(asdict(settings), sort_keys=True).encode())
    for path in (Path(__file__),):
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
    return {"bits": candidate.genome.bits, "score": candidate.score,
            "validation_loss": candidate.validation_loss, "parameters": candidate.parameters,
            "model_sample_steps": candidate.model_sample_steps, "epochs": candidate.epochs}


def _unpack(item):
    values = dict(item)
    genome = Genome(tuple(values.pop("bits")))
    return Candidate(genome, **values)


def _key(genome):
    return str(genome.architecture())


def _breed(ranked, rng, config, options):
    elite = [item.genome for item in ranked[:min(options.elite, config.population)]]
    mutation = min(.5, config.mutation_probability * (1.5 - diversity(ranked)))

    def tournament():
        indices = rng.integers(len(ranked), size=options.tournament)
        return ranked[min(indices)].genome

    pool = elite.copy()
    while len(pool) < config.population:
        left, right = tournament(), tournament()
        mask = rng.random(len(left.bits)) < .5
        bits = np.where(mask, left.bits, right.bits)
        bits ^= rng.random(len(bits)) < mutation
        pool.append(Genome(tuple(map(int, bits))))
    return pool


def run_search(mode, score, settings, seed, checkpoint, fingerprint, options=SearchOptions(),
               *, stop_after=None):
    """Checkpoint completed generations. A retried partial generation is recomputed.

    Scores are evaluated in fixed order even with parallel workers. Cache hits
    count as logical evaluations, with actual work and cache hits reported
    separately. Disable caching when matching physical evaluation counts.
    """
    if mode not in ("classical_ga", "quantum_inspired", "random") or not fingerprint:
        raise ValueError("Unknown mode or missing validation fingerprint")
    rng = np.random.default_rng(seed)
    pool = []
    amplitudes = []
    history = []
    diagnostics = []
    cache = {}
    best = None
    stale = 0
    checkpoint = Path(checkpoint)
    recipe = {"version": 1, "mode": mode, "seed": seed, "data": fingerprint,
              "evolution": asdict(settings), "options": asdict(options)}
    if checkpoint.exists():
        state = read_json(checkpoint)
        if recipe != state["recipe"]:
            raise ValueError("Search checkpoint recipe, data, or settings differ")
        rng.bit_generator.state = state["rng"]
        history = [[_unpack(candidate) for candidate in generation] for generation in state["history"]]
        diagnostics = state["diagnostics"]
        cache = {name: _unpack(candidate) for name, candidate in state["cache"].items()}
        pool = [Genome(tuple(bits)) for bits in state["population"]]
        for angles in state["angles"]:
            individual = QuantumGenome()
            individual.angles = np.asarray(angles, dtype=float)
            amplitudes.append(individual)
        best = min((item for generation in history for item in generation), key=lambda item: item.score)
        stale = state["stale"]
    elif mode == "quantum_inspired":
        amplitudes = [QuantumGenome() for _ in range(settings.population)]
    elif mode == "classical_ga":
        pool = [Genome.random(rng) for _ in range(settings.population)]

    stop = settings.generations if stop_after is None else min(stop_after, settings.generations)
    if stop < 1:
        raise ValueError("Generation limit must permit evaluation")
    with ThreadPoolExecutor(max_workers=options.workers) as workers:
        for generation in range(len(history), stop):
            if options.patience is not None and stale >= options.patience:
                break
            if mode == "quantum_inspired":
                pool = [item.measure(rng) for item in amplitudes]
            elif mode == "random":
                pool = [Genome.random(rng) for _ in range(settings.population)]

            if options.cache:
                pending = {}
                for genome in pool:
                    key = _key(genome)
                    if key not in cache and key not in pending:
                        pending[key] = genome
                actual = list(workers.map(score, pending.values()))
                cache.update(zip(pending, actual))
                candidates = [Candidate(genome, hit.score, hit.validation_loss, hit.parameters,
                                        hit.model_sample_steps, hit.epochs)
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
            diagnostics.append({
                "generation": generation + 1, "best_selection_score": best.score,
                "diversity": diversity(ranked), "actual_evaluations": len(actual),
                "cache_hits": len(pool) - len(actual),
                "actual_training_sample_steps": sum(item.model_sample_steps for item in actual),
                "actual_parameter_sample_steps": sum(item.model_sample_steps * item.parameters
                                                     for item in actual),
            })
            if mode == "classical_ga":
                pool = _breed(ranked, rng, settings, options)
            elif mode == "quantum_inspired":
                for individual, candidate in zip(amplitudes, candidates):
                    individual.rotate(candidate.genome, best.genome, settings.rotation_radians)
            state = {"recipe": recipe, "rng": rng.bit_generator.state,
                     "history": [[_pack(item) for item in group] for group in history],
                     "diagnostics": diagnostics, "stale": stale,
                     "cache": {name: _pack(item) for name, item in cache.items()},
                     "population": [item.bits for item in pool],
                     "angles": [item.angles for item in amplitudes]}
            write_json(checkpoint, state, indent=None)
    return best, history, diagnostics
