"""Architecture diversity and Pareto front diagnostics on validation candidates."""
import itertools

import numpy as np

from Evolution.Genome import Genome


def random_search(score, settings, seed=42):
    rng = np.random.default_rng(seed)
    history = [[score(Genome.random(rng)) for _ in range(settings.population)]
               for _ in range(settings.generations)]
    return min(itertools.chain.from_iterable(history), key=lambda member: member.score), history


def diversity(candidates):
    genotypes = np.asarray([item.genome.bits for item in candidates], dtype=float)
    if len(genotypes) < 2:
        return 0.
    counts = genotypes.sum(axis=0)
    pair_differences = (counts * (len(genotypes) - counts)).sum()
    return float(pair_differences / (len(genotypes[0]) * len(genotypes) * (len(genotypes) - 1) / 2))


def pareto_front(candidates):
    """Minimize validation loss and parameter count, deduplicating equal scores."""
    result = []
    least_loss = np.inf
    for item in sorted(candidates, key=lambda c: (c.parameters, c.validation_loss)):
        if item.validation_loss < least_loss:
            result.append(item)
            least_loss = item.validation_loss
    return result
