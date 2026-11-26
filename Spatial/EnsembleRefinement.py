"""Shared adaptive kriging geometry for coherent forecast trajectories."""
import numpy as np

from Spatial.AdaptiveGrid import Refinement, refine
from Spatial.Kriging import CovarianceModel, KrigingModel
from spatial_forecast import interpolation_weights


def adaptive_ensemble(points, members, bounds, settings, probability):
    """Return reproducible draws and conditional variances on quadtree centers.

    Thresholds for variation and model spread use dimensionless, per-feature
    scales supplied by the caller. Kriging posterior variance is reported
    separately from empirical checkpoint disagreement.
    """
    scales = np.asarray(settings["feature_scales"], dtype=float)
    if scales.shape != (members.shape[-1],) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Supply one positive finite refinement scale per feature")
    covariance = CovarianceModel(**settings["covariance"])
    trend = settings.get("trend", "constant")
    model = KrigingModel(points, np.zeros(len(points)), covariance, trend=trend)

    def operator(query):
        design = model._design(query)
        kernel = np.linalg.solve(model.factor, covariance(points, query))
        drift = design.T - model.whitened_design.T @ kernel
        coefficient = kernel + model.whitened_design @ model._drift_solve(drift)
        return np.linalg.solve(model.factor.T, coefficient).T

    def driver(query):
        weights = operator(query)
        draws = np.einsum("qs,mslf->mqlf", weights, members) / scales
        # Shift-free scalar refinement driver uses the strongest standardized
        # spatial gradient through a per-channel refinement criterion below.
        mean = draws.mean(axis=0).reshape(len(query), -1)
        variance = draws.var(axis=0).reshape(len(query), -1)
        return mean, variance

    geometry = refine(bounds, driver, Refinement(**settings.get("refinement", {})),
                      observed_xy=points, polygon=settings.get("field_polygon_m"))
    cells = np.array([cell["bounds_m"] for cell in geometry["cells"]])
    query = (cells[:, :2] + cells[:, 2:]) / 2
    weights = operator(query)
    draws = np.einsum("qs,mslf->mlqf", weights, members)
    alpha = (1 - probability) / 2
    lower, upper = np.quantile(draws, [alpha, 1 - alpha], axis=0)
    idw = interpolation_weights(points, query, 2.)
    return {
        "adaptive_bounds_m": cells,
        "adaptive_centers_m": query,
        "adaptive_inside": np.array([cell["inside"] for cell in geometry["cells"]]),
        "adaptive_depth": np.array([cell["depth"] for cell in geometry["cells"]]),
        "adaptive_weights": weights,
        "adaptive_draws": draws,
        "adaptive_mean": draws.mean(axis=0),
        "adaptive_spread": draws.std(axis=0),
        "adaptive_lower": lower,
        "adaptive_upper": upper,
        "adaptive_idw_weights": idw,
        "adaptive_conditional_variance": model.predict(query).latent_variance,
    }, {
        "method": "adaptive_kriging",
        "axes": ["lead", "cell", "feature"],
        "draw_axes": ["member", "lead", "cell", "feature"],
        "weight_axes": ["cell", "site"],
        "conditioning_jitter": model.jitter,
        "feature_scales": scales.tolist(),
        "conditional_variance_interpretation": "variance in the declared covariance model units",
        **{key: value for key, value in geometry.items() if key != "cells"},
    }
