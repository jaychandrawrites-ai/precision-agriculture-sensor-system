"""Classical binary crossover and mutation with validation-fitness selection."""
import math

import numpy as np

from Evolution.Genome import Genome


def search(score, settings, seed=42):
    rng = np.random.default_rng(seed)
    population = [Genome.random(rng) for _ in range(settings.population)]
    survivors = max(1, math.ceil(settings.population * settings.survival_fraction))
    history = []
    for generation in range(settings.generations):
        ranked = sorted((score(genome) for genome in population), key=lambda item: item.score)
        history.append(ranked)
        if generation + 1 < settings.generations:
            parents = [item.genome for item in ranked[:survivors]]
            population = parents.copy()
            while len(population) < settings.population:
                first, second = rng.integers(survivors, size=2)
                population.append(parents[first].offspring(
                    parents[second], rng, settings.mutation_probability
                ))
    return min((candidate for group in history for candidate in group),
               key=lambda item: item.score), history
