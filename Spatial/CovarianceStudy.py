"""Visibility-restricted predictions and coherent covariance scenario samples."""
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from Spatial.Kriging import CovarianceModel, KrigingModel, coordinates


@dataclass(frozen=True)
class VisibilityPrediction:
    mean: np.ndarray
    latent_variance: np.ndarray
    active: np.ndarray
    support_count: np.ndarray
    supported: np.ndarray


def visible_kriging(points, values, queries, covariance, domain, *, trend="constant",
                    measurement_variance=None):
    """Fit only sites connected to each query by an admissible straight segment."""
    points, queries = coordinates(points), coordinates(queries)
    values = np.asarray(values, dtype=float)
    noise = (np.zeros(len(points)) if measurement_variance is None
             else np.asarray(measurement_variance, dtype=float))
    if (values.shape != (len(points),) or not np.isfinite(values).all()
            or noise.shape != values.shape or not np.isfinite(noise).all() or (noise < 0).any()
            or trend not in ("constant", "affine") or not domain.contains(points).all()):
        raise ValueError("Invalid observation, trend or measurement uncertainty for field domain")
    active = domain.contains(queries)
    groups = defaultdict(list)
    mean, spread = np.full(len(queries), np.nan), np.full(len(queries), np.nan)
    support = np.zeros(len(queries), dtype=int)
    for i in np.flatnonzero(active):
        visible = tuple(j for j, point in enumerate(points) if domain.visible(point, queries[i]))
        support[i] = len(visible)
        if visible:
            groups[visible].append(i)
    for group, indexes in groups.items():
        selection = np.asarray(group)
        geometry = points[selection]
        if trend == "affine" and np.linalg.matrix_rank(
                np.column_stack((np.ones(len(selection)), geometry - geometry.mean(axis=0)))) < 3:
            continue
        model = KrigingModel(geometry, values[selection], covariance, trend=trend,
                             measurement_variance=noise[selection])
        result = model.predict(queries[indexes])
        mean[indexes], spread[indexes] = result.mean, result.latent_variance
    return VisibilityPrediction(mean, spread, active, support, np.isfinite(mean))


@dataclass(frozen=True)
class CovarianceScenario:
    name: str
    covariance: CovarianceModel

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or not isinstance(self.covariance, CovarianceModel):
            raise ValueError("Each covariance scenario needs a unique name and a model")


@dataclass(frozen=True)
class ScenarioField:
    draws: np.ndarray
    scenario_means: np.ndarray
    conditional_variance: np.ndarray
    scenario_variance: np.ndarray
    diagnostics: tuple

    @property
    def variance(self):
        return self.conditional_variance + self.scenario_variance


def _conditional_covariance(model, queries):
    kernel = np.linalg.solve(model.factor, model.model(model.points, queries))
    drift = model._design(queries).T - model.whitened_design.T @ kernel
    uncertainty = np.linalg.solve(model.drift_factor, drift)
    matrix = model.model(queries, queries) - kernel.T @ kernel + uncertainty.T @ uncertainty
    return (matrix + matrix.T) / 2


def covariance_scenario_fields(points, values, queries, scenarios, *, measurement_variance=None,
                               trend="constant", draws=100, seed=42, max_queries=512):
    """Draw whole latent fields under equal-weight, caller-declared covariance scenarios.

    Full covariance rather than pointwise draws preserves spatial coherence.
    Measurement variance and nugget are used for conditioning only. The
    returned mixture is a sensitivity analysis unless scenario weights are
    independently justified.
    """
    queries, scenarios = coordinates(queries), tuple(scenarios)
    if (type(draws) is not int or draws < 2 or type(max_queries) is not int
            or not 1 <= len(queries) <= max_queries or len(queries) > 4096 or not scenarios
            or len({item.name for item in scenarios}) != len(scenarios)):
        raise ValueError("Invalid dense scenario budget, draws or scenario names")
    if len(queries) * draws * len(scenarios) > 10_000_000:
        raise ValueError("Coherent sample budget exceeded")
    sequences = np.random.SeedSequence(seed).spawn(len(scenarios))
    all_draws, means, variances, diagnostics = [], [], [], []
    for scenario, sequence in zip(scenarios, sequences):
        model = KrigingModel(points, values, scenario.covariance, trend=trend,
                             measurement_variance=measurement_variance)
        pred = model.predict(queries)
        covariance = _conditional_covariance(model, queries)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        scale = max(scenario.covariance.sill, np.abs(eigenvalues).max())
        if eigenvalues.min() < -1e-8 * scale:
            raise FloatingPointError("Conditional covariance is not positive semidefinite")
        eigenvalues = np.maximum(eigenvalues, 0)
        root = eigenvectors * np.sqrt(eigenvalues)
        rng = np.random.default_rng(sequence)
        all_draws.append(pred.mean + rng.normal(size=(draws, len(queries))) @ root.T)
        means.append(pred.mean)
        variances.append(pred.latent_variance)
        weak = []
        if np.ptp(model.points[:, 0]) == 0 or np.ptp(model.points[:, 1]) == 0:
            weak.append("one_axis_site_geometry")
        extent = np.linalg.norm(np.ptp(model.points, axis=0))
        if extent < scenario.covariance.range_m:
            weak.append("range_exceeds_site_extent")
        diagnostics.append({
            "name": scenario.name, "parameters": asdict(scenario.covariance),
            "observation_condition_number": float(np.linalg.cond(model.factor)**2),
            "conditioning_jitter": float(model.jitter),
            "conditional_rank": int(np.count_nonzero(eigenvalues > 1e-10 * scale)),
            "identification_flags": weak,
            "assumption": "Declared scenario, not estimated from held-out responses",
        })
    means = np.stack(means)
    return ScenarioField(np.stack(all_draws), means, np.mean(variances, axis=0),
                         means.var(axis=0), tuple(diagnostics))


def posterior_predictive_checks(draws, observations, *, probability=.90, observation_variance=None,
                                seed=42):
    """Evaluate coherent field draws at held-out observed sites.

    Optional independent observation errors are added once, after the caller
    provides latent predictive draws. Existing observation-predictive draws
    should omit this argument.
    """
    observations = np.asarray(observations, dtype=float)
    draws = np.asarray(draws, dtype=float)
    if (observations.ndim != 1 or not len(observations) or draws.ndim < 2
            or draws.shape[-1] != len(observations) or not 0 < probability < 1
            or draws.size < 2 * len(observations) or not np.isfinite(draws).all()
            or not np.isfinite(observations).all()):
        raise ValueError("Expected held-out observations and finite aligned coherent draws")
    samples = draws.reshape(-1, len(observations))
    if observation_variance is not None:
        noise = np.asarray(observation_variance, dtype=float)
        if noise.shape != observations.shape or not np.isfinite(noise).all() or (noise < 0).any():
            raise ValueError("Expected one finite nonnegative observation variance per site")
        samples = samples + np.random.default_rng(seed).normal(size=samples.shape) * np.sqrt(noise)
    alpha = 1 - probability
    lower, upper = np.quantile(samples, [alpha/2, 1-alpha/2], axis=0)
    mean, sd = samples.mean(axis=0), samples.std(axis=0)
    residual = observations - mean
    covered = (observations >= lower) & (observations <= upper)
    width = upper - lower
    penalty = np.maximum(lower - observations, 0) + np.maximum(observations - upper, 0)
    replicated = {}
    for name, stat in (("mean", np.mean), ("spatial_standard_deviation", np.std),
                       ("minimum", np.min), ("maximum", np.max)):
        truth = float(stat(observations))
        comparison = stat(samples, axis=1)
        replicated[name] = {"observed": truth, "replicate_mean": float(comparison.mean()),
                            "replicate_tail_ge": float((comparison >= truth).mean())}
    return {
        "held_out_sites": len(observations), "predictive_draws": len(samples),
        "probability": probability, "coverage": float(covered.mean()),
        "mean_width": float(width.mean()), "mean_interval_score": float((width + 2*penalty/alpha).mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "site_diagnostics": [
            {"mean": float(m), "lower": float(lo), "upper": float(hi),
             "predictive_sd": float(s), "residual": float(r),
             "standardized_residual": float(r/s) if s > 0 else None, "covered": bool(ok)}
            for m, lo, hi, s, r, ok in zip(mean, lower, upper, sd, residual, covered)
        ],
        "replicated_field_checks": replicated,
        "observation_uncertainty": (
            "independent Gaussian errors added once" if observation_variance is not None
            else "no additional observation errors added"
        ),
    }
