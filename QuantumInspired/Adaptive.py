"""Fitness-ranked classical amplitude updates and Shannon entropy diagnostics."""

import math

import numpy as np


def binary_entropy(probability):
    """Return Bernoulli entropy in bits, including at exact probabilities zero and one."""
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected probabilities in [0, 1]")
    entropy = np.zeros_like(p)
    interior = (p > 0) & (p < 1)
    q = p[interior]
    entropy[interior] = -q * np.log2(q) - (1 - q) * np.log2(1 - q)
    return entropy


def amplitude_diagnostics(individuals):
    p = np.asarray([individual.amplitudes[:, 1] ** 2 for individual in individuals])
    return {
        "mean_individual_bit_entropy": float(binary_entropy(p).mean()),
        "mean_population_bit_entropy": float(binary_entropy(p.mean(axis=0)).mean()),
        "mean_population_bit_probabilities": p.mean(axis=0).tolist(),
    }


def adaptive_update(individuals, sampled, evaluated, guide, base_angle, progress, entropy_floor, *, ranked=None):
    """Rotate poor samples more strongly and relax concentrated distributions.

    `sampled` maps the measured population slots to their validation records.
    Classical offspring have validation ranks and can guide the update, but
    their genotypes do not represent measurements of an amplitude vector.
    """
    if not 0 <= progress <= 1 or not 0 <= entropy_floor <= 1 or not 0 <= base_angle <= math.pi / 2:
        raise ValueError("Invalid adaptive amplitude settings")
    order = sorted(evaluated, key=lambda c: c.score) if ranked is None else ranked
    rank = {id(record): index for index, record in enumerate(order)}
    annealed = base_angle * (1 - 0.5 * progress)
    rotations = []
    for slot, candidate in sampled:
        individual = individuals[slot]
        severity = rank[id(candidate)] / max(1, len(evaluated) - 1)
        angle = annealed * (0.5 + 0.5 * severity)
        individual.rotate(candidate.genome, guide, angle)
        rotations.append(angle)
    restored = 0
    for individual in individuals:
        entropy = binary_entropy(individual.amplitudes[:, 1] ** 2).mean()
        if entropy < entropy_floor:
            individual.relax(min(0.3, entropy_floor - entropy))
            restored += 1
    return {**amplitude_diagnostics(individuals),
            "mean_rotation_radians": float(np.mean(rotations)) if rotations else 0.0,
            "relaxed_individuals": restored}
