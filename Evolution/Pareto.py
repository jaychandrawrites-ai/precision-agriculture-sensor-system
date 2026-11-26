"""Deterministic nondominated ranking and diversity for architecture selection."""
from collections import Counter
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ObjectivePolicy:
    """Minimize each objective and optionally constrain latency or raw invalid output."""

    include_latency: bool = False
    max_latency_seconds: float | None = None
    max_estimated_bytes: int | None = None
    max_raw_invalid_fraction: float | None = None

    def __post_init__(self):
        if self.max_latency_seconds is not None and (
            not math.isfinite(self.max_latency_seconds) or self.max_latency_seconds <= 0
        ):
            raise ValueError("Latency limit must be finite and positive")
        if self.max_estimated_bytes is not None and (
            type(self.max_estimated_bytes) is not int or self.max_estimated_bytes < 1
        ):
            raise ValueError("Memory limit must be a positive number of estimated bytes")
        if self.max_raw_invalid_fraction is not None and not 0 <= self.max_raw_invalid_fraction <= 1:
            raise ValueError("Raw invalid fraction limit must be in [0, 1]")

    @property
    def names(self):
        names = ("validation_loss", "parameters", "parameter_sample_steps",
                 "estimated_training_bytes", "raw_invalid_fraction")
        return (*names, "prediction_batch_seconds") if self.include_latency else names

    def values(self, candidate):
        values = (candidate.validation_loss, candidate.parameters,
                  candidate.parameters * candidate.model_sample_steps,
                  candidate.estimated_training_bytes, candidate.raw_invalid_fraction)
        if self.include_latency:
            values = (*values, candidate.prediction_batch_seconds)
        if any(value is None or not math.isfinite(value) for value in values):
            raise ValueError("Candidate has missing or nonfinite Pareto objectives")
        return values

    def constraint_violation(self, candidate):
        constraints = (
            (candidate.prediction_batch_seconds, self.max_latency_seconds),
            (candidate.estimated_training_bytes, self.max_estimated_bytes),
            (candidate.raw_invalid_fraction, self.max_raw_invalid_fraction),
        )
        violation = 0.0
        for value, maximum in constraints:
            if maximum is not None:
                if value is None or not math.isfinite(value):
                    raise ValueError("Candidate has a missing or nonfinite constrained objective")
                violation += max(0., value - maximum) / (maximum if maximum > 0 else 1.)
        return violation


def fronts(candidates, policy):
    """Return increasing constrained nondomination ranks as candidate indices."""
    vectors = [policy.values(item) for item in candidates]
    violations = [policy.constraint_violation(item) for item in candidates]
    dominates = [[] for _ in candidates]
    count = [0] * len(candidates)

    def better(a, b):
        if violations[a] != violations[b]:
            return violations[a] < violations[b]
        return (all(x <= y for x, y in zip(vectors[a], vectors[b])) and vectors[a] != vectors[b])

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if better(i, j):
                dominates[i].append(j)
                count[j] += 1
            elif better(j, i):
                dominates[j].append(i)
                count[i] += 1
    current = [i for i, amount in enumerate(count) if amount == 0]
    result = []
    while current:
        result.append(current)
        following = []
        for i in current:
            for j in dominates[i]:
                count[j] -= 1
                if count[j] == 0:
                    following.append(j)
        current = following
    return result


def crowding(candidates, members, policy):
    """Normalized coordinate-wise neighborhood distances for one front."""
    distances = {i: 0. for i in members}
    if not members:
        return distances
    if len(members) <= 2:
        return dict.fromkeys(members, math.inf)
    vectors = {i: policy.values(candidates[i]) for i in members}
    for axis in range(len(policy.names)):
        order = sorted(members, key=lambda i: (vectors[i][axis], candidates[i].genome.bits))
        lo, hi = vectors[order[0]][axis], vectors[order[-1]][axis]
        if hi == lo:
            continue
        distances[order[0]] = distances[order[-1]] = math.inf
        for left, current, right in zip(order, order[1:], order[2:]):
            distances[current] += (vectors[right][axis] - vectors[left][axis]) / (hi - lo)
    return distances


def rank_candidates(candidates, policy):
    """Pareto rank and crowding for survivor and tournament selection."""
    result = []
    for members in fronts(candidates, policy):
        distance = crowding(candidates, members, policy)
        result.extend(candidates[i] for i in sorted(members, key=lambda i: (
            -distance[i], candidates[i].score, candidates[i].genome.bits)))
    return result


def update_archive(archive, new, policy):
    """Retain unique objective vectors in the best known constrained front."""
    unique = {}
    for item in [*archive, *new]:
        key = policy.constraint_violation(item), policy.values(item)
        existing = unique.get(key)
        if existing is None or (item.score, item.genome.bits) < (existing.score, existing.genome.bits):
            unique[key] = item
    members = list(unique.values())
    ranks = fronts(members, policy)
    return sorted((members[i] for i in ranks[0]), key=lambda item: (
        item.score, item.genome.bits)) if ranks else []


def exploration(candidates):
    """Report both learned phenotype variety and observed bit frequencies."""
    unique = Counter(str(item.genome.decode()) for item in candidates)
    probabilities = np.asarray(list(unique.values()), dtype=float) / len(candidates)
    bits = np.mean([item.genome.bits for item in candidates], axis=0)
    inside = bits[(bits > 0) & (bits < 1)]
    entropy = -np.sum(inside * np.log2(inside) + (1 - inside) * np.log2(1 - inside))
    return {"unique_architectures": len(unique), "architecture_entropy_bits":
            float(-(probabilities * np.log2(probabilities)).sum()),
            "mean_observed_bit_entropy": float(entropy / len(bits)), "observed_bit_probabilities": bits.tolist()}
