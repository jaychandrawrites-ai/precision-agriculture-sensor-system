"""Classical amplitude-based search with the same candidate budget as the GA."""
import numpy as np

from QuantumInspired.QuantumGenome import QuantumGenome


def search(score, settings, seed=42):
    rng = np.random.default_rng(seed)
    population = [QuantumGenome() for _ in range(settings.population)]
    history = []
    best = None
    for _ in range(settings.generations):
        candidates = [score(individual.measure(rng)) for individual in population]
        history.append(candidates)
        winner = min(candidates, key=lambda candidate: candidate.score)
        if best is None or winner.score < best.score:
            best = winner
        for individual, candidate in zip(population, candidates):
            individual.rotate(candidate.genome, best.genome, settings.rotation_radians)
    return best, history
