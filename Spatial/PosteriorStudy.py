"""Separate conditional field uncertainty, observation noise and scenario spread."""
from dataclasses import dataclass

import numpy as np

from Spatial.Validation import sample_grid


@dataclass(frozen=True)
class PredictiveSummary:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    conditional_variance: np.ndarray
    scenario_variance: np.ndarray
    observation_variance: np.ndarray

    @property
    def total_variance(self):
        return self.conditional_variance + self.scenario_variance + self.observation_variance


def predictive_summary(grid, scenarios, queries, *, probability=0.90, observation_variance=None,
                       seed=42):
    """Equal mixture of caller-supplied scenarios with optional observation noise.

    Scenarios are explicitly weighted equally. Their spread describes model
    sensitivity unless scenario weights and priors have independent support.
    """
    scenarios = tuple(scenarios)
    queries = np.asarray(queries, dtype=float)
    if not scenarios or not 0 < probability < 1 or not len(queries):
        raise ValueError("Expected scenarios, nonempty sites and a probability")
    noise = (np.zeros(len(queries)) if observation_variance is None
             else np.asarray(observation_variance, dtype=float))
    if noise.shape != (len(queries),) or not np.isfinite(noise).all() or (noise < 0).any():
        raise ValueError("Observation variances must be finite and nonnegative")
    estimates = []
    for sampler in scenarios:
        fields = np.asarray(sampler.fields, dtype=float)
        if fields.ndim != 4 or fields.shape[1] < 2 or not np.isfinite(fields).all():
            raise ValueError("Expected finite per-chain field draws")
        estimates.append(sample_grid(grid, fields, queries).reshape(-1, len(queries)))
    if len({len(draws) for draws in estimates}) > 1:
        raise ValueError("Equal scenario weighting requires matched chain and draw counts")
    fields = np.stack(estimates)
    means = fields.mean(axis=1)
    marginal = means.mean(axis=0)
    conditional = fields.var(axis=1).mean(axis=0)
    scenario = means.var(axis=0)
    observed = fields.copy()
    if noise.any():
        rng = np.random.default_rng(seed)
        observed += rng.normal(size=observed.shape) * np.sqrt(noise)
    tail = (1 - probability) / 2
    lower, upper = np.quantile(observed.reshape(-1, len(queries)), [tail, 1-tail], axis=0)
    return PredictiveSummary(marginal, lower, upper, conditional, scenario, noise)


def interval_scores(summary, observed, probability=0.90):
    """Empirical coverage and proper interval scores at independent sites."""
    values = np.asarray(observed, dtype=float)
    if (values.shape != summary.lower.shape or not len(values)
            or not np.isfinite(values).all() or not 0 < probability < 1):
        raise ValueError("Expected aligned finite observations and a probability")
    width = summary.upper - summary.lower
    miscoverage = (np.maximum(summary.lower-values, 0.0) +
                   np.maximum(values-summary.upper, 0.0))
    return {"count": len(values),
            "coverage": float(((values >= summary.lower) & (values <= summary.upper)).mean()),
            "mean_width": float(width.mean()),
            "mean_interval_score": float((width + 2 * miscoverage / (1-probability)).mean()),
            "rmse": float(np.sqrt(np.mean((values-summary.mean)**2)))}


def mixing_report(summary, *, maximum_rhat=1.05, minimum_ess=100, active=None):
    """Expose poor or unavailable diagnostics rather than silently masking them."""
    if maximum_rhat <= 1 or minimum_ess < 1:
        raise ValueError("Invalid mixing thresholds")
    rhat, ess = summary.split_rhat, summary.ess
    active = np.ones(rhat.shape, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if active.shape != rhat.shape:
        raise ValueError("Active cell mask must match the grid")
    known = active & np.isfinite(rhat) & np.isfinite(ess)
    bad = known & ((rhat > maximum_rhat) | (ess < minimum_ess))
    return {"diagnosed": int(known.sum()), "poor_mixing": int(bad.sum()),
            "undetermined": int((active & ~known).sum()), "active": int(active.sum()),
            "threshold_rhat": maximum_rhat, "threshold_ess": minimum_ess,
            "all_active_pass": bool(active.any() and np.array_equal(known, active)
                                    and not bad.any())}
