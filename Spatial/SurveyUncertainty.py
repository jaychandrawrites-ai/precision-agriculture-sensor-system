"""Compose calibrated trajectory residuals and declared field scenarios."""
from dataclasses import asdict

import numpy as np

from Spatial.CovarianceStudy import _conditional_covariance
from Spatial.FieldDomain import FieldDomain
from Spatial.Kriging import CovarianceModel, KrigingModel


def predictive_fields(points, members, geometry, settings, mass, *, operator=None):
    """Sample a scenario once per trajectory and export its full provenance.

    Member values are latent site forecasts. The nugget conditions the field
    only. Innovations are spatially correlated within a feature and persistent
    across leads. Cross-feature covariance comes from the input trajectories.
    """
    members = np.asarray(members, dtype=float)
    points = np.asarray(points, dtype=float)
    if (members.ndim != 4 or members.shape[1] != len(points)
            or not members.size or not np.isfinite(members).all() or not 0 < mass < 1):
        raise ValueError("Expected finite member × site × lead × feature trajectories")
    queries = np.asarray(geometry, dtype=float)
    boundary = settings.get("field_polygon_m")
    holes = settings.get("field_holes_m", ())
    if boundary is None and holes:
        raise ValueError("Field holes require an exterior polygon")
    inside = (np.ones(len(queries), dtype=bool) if boundary is None
              else FieldDomain(boundary, tuple(holes)).contains(queries))
    sites, leads, features = members.shape[1:]
    method = settings.get("method", "kriging")
    if method not in {"kriging", "idw", "variational", "adaptive_variational"}:
        raise ValueError("Unknown predictive reconstruction method")
    scenarios = settings.get("covariance_scenarios", [])
    if method != "kriging" and scenarios:
        raise ValueError("Covariance scenarios require kriging. Fixed operators propagate site uncertainty only")
    if method == "idw":
        from spatial_forecast import interpolation_weights
        operator = interpolation_weights(points, queries, 2.)
    if method in {"variational", "adaptive_variational"} and operator is None:
        raise ValueError("Variational prediction requires a configured frozen mesh")
    if method != "kriging":
        operator = np.asarray(operator, dtype=float)
        if operator.shape != (len(queries), len(points)):
            raise ValueError("Reconstruction weights must align with cells and sites")
        inside &= np.isfinite(operator).all(axis=1)

    names = [item["id"] for item in scenarios]
    weights = np.array([item["weight"] for item in scenarios], dtype=float)
    draws = settings.get("draws", 100)
    seed = settings.get("seed", 42)
    budget = settings.get("memory_bytes", 128 * 1024**2)
    n = int(inside.sum())
    if ((method == "kriging" and (not len(scenarios) or not np.isclose(weights.sum(), 1.)))
            or len(names) != len(set(names)) or not np.isfinite(weights).all()
            or (weights <= 0).any()
            or type(draws) is not int or draws < 2 or not 1 <= n <= 1024
            or type(budget) is not int or budget <= 0
            or 8 * (4 * draws * len(queries) * leads * features + 12 * n**2 + 4 * sites**2) > budget):
        raise ValueError("Invalid covariance weights, field geometry or memory budget")
    active = queries[inside]
    scales = np.asarray(settings["feature_scales"], dtype=float)
    if scales.shape != (features,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Expected a positive physical scale for each feature")
    residuals = np.asarray(settings.get("residuals", []), dtype=float)
    origins = settings.get("residual_origins", [])
    if residuals.size and (residuals.ndim != 4 or residuals.shape[1:] != members.shape[1:]
                           or len(residuals) != len(origins) or not np.isfinite(residuals).all()):
        raise ValueError("Calibration residuals must align with all sites, leads and features")
    rng = np.random.default_rng(seed)
    chosen = (rng.choice(len(scenarios), size=draws, p=weights / weights.sum())
              if scenarios else np.full(draws, -1))
    member_weights = settings.get("member_probabilities")
    member_ids = rng.choice(len(members), size=draws, p=member_weights)
    residual_ids = rng.integers(len(residuals), size=draws) if residuals.size else np.full(draws, -1)
    site_draws = members[member_ids].copy()
    if residuals.size:
        site_draws += residuals[residual_ids]
    field = np.full((draws, leads, len(queries), features), np.nan)
    report = []
    chunk = settings.get("chunk_cells", 32)
    if type(chunk) is not int or chunk < 1:
        raise ValueError("chunk_cells must be a positive integer")
    indices = np.flatnonzero(inside)
    if method != "kriging":
        for start in range(0, len(indices), chunk):
            cells = indices[start:start + chunk]
            field[:, :, cells, :] = np.einsum("qs,mslf->mlqf", operator[cells], site_draws)

    for i, item in enumerate(scenarios):
        covariance = CovarianceModel(**item["covariance"])
        model = KrigingModel(points, np.zeros(sites), covariance, trend=item.get("trend", "constant"))
        kernel = np.linalg.solve(model.factor, covariance(points, active))
        drift = model._design(active).T - model.whitened_design.T @ kernel
        operator = np.linalg.solve(
            model.factor.T, kernel + model.whitened_design @ model._drift_solve(drift)).T
        spectrum, basis = np.linalg.eigh(_conditional_covariance(model, active))
        if spectrum.min() < -1e-8 * covariance.sill:
            raise FloatingPointError("Invalid conditional field covariance")
        root = basis * np.sqrt(np.maximum(spectrum, 0))
        for index in np.flatnonzero(chosen == i):
            mean = np.einsum("qs,slf->lqf", operator, site_draws[index])
            innovation = root @ rng.normal(size=(n, features))
            field[index][:, inside, :] = mean + innovation * scales
        report.append({"id": item["id"], "weight": float(weights[i]), "covariance": asdict(covariance),
                       "trend": model.trend, "conditioning_jitter": float(model.jitter),
                       "observation_condition_number": float(np.linalg.cond(model.factor)**2),
                       "realized_draws": int((chosen == i).sum())})
    summary = np.full((8, leads, len(queries), features), np.nan)
    groups = [chosen == value for value in np.unique(chosen)]
    for start in range(0, len(indices), chunk):
        cells = indices[start:start + chunk]
        values = field[:, :, cells, :]
        low, median, high = np.quantile(values, [(1 - mass) / 2, .5, (1 + mass) / 2], axis=0)
        mean = values.mean(axis=0)
        within = np.zeros_like(mean)
        between = np.zeros_like(mean)
        for group in groups:
            subset = values[group]
            frequency = group.mean()
            within += frequency * subset.var(axis=0)
            between += frequency * (subset.mean(axis=0) - mean)**2
        summary[:, :, cells, :] = [mean, median, values.std(axis=0), low, high, high - low,
                                   within, between]
    arrays = dict(zip(("predictive_mean", "predictive_median", "predictive_std",
                       "predictive_lower", "predictive_upper", "predictive_width",
                       "predictive_within_covariance_variance", "predictive_between_covariance_variance"), summary))
    arrays.update(predictive_draws=field, predictive_inside=inside, predictive_centers_m=queries,
                  predictive_site_draws=site_draws, predictive_member_index=member_ids,
                  predictive_residual_index=residual_ids, predictive_covariance_index=chosen)
    return arrays, {"method": method, "chunk_cells": chunk, "seed": seed, "draws": draws, "covariance_scenarios": report,
                    "residual_origins": origins, "axes": ["draw", "lead", "cell", "feature"],
                    "draw_weight": 1 / draws, "member_probabilities": member_weights, "site_axis_count": sites,
                    "assumptions": (["independent scenario, member and residual selections",
                                    "one spatial innovation per feature shared across all leads",
                                    "kriging covariance in standardized feature units",
                                    "latent draws; nugget enters conditioning once",
                                    "boundary masks describe cell centers",
                                    "field predictive coverage unverified"] if method == "kriging" else
                                    ["independent member and residual selections",
                                     "fixed spatial operator with no conditional field innovation",
                                     "field predictive coverage unverified"])}


def variational_trajectories(points, members, bounds, settings, mass):
    """Apply the same finite-volume operator to every member, lead and feature."""
    from Spatial.AdaptiveVariational import VariationalRefinement, adaptive_variational

    refinement = VariationalRefinement(**settings.get("refinement", {}))
    if refinement.uncertainty_threshold is not None:
        raise ValueError("Survey variational geometry uses gradient and observation refinement")
    if len(points) > 64 or refinement.max_cells > 2048:
        raise ValueError("Survey variational solve exceeds site or cell limit")
    options = {"hard": settings.get("hard", False), "polygon": settings.get("field_polygon_m"),
               "holes": settings.get("field_holes_m", ()),
               "measurement_variance": settings.get("measurement_variance")}
    scales = np.asarray(settings["feature_scales"], dtype=float)
    if scales.shape != (members.shape[-1],) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Supply one positive refinement scale per feature")
    if (type(settings.get("memory_bytes", 128 * 1024**2)) is not int
            or 32 * (members.shape[0] * members.shape[2] * members.shape[3] + len(points))
            * refinement.max_cells > settings.get("memory_bytes", 128 * 1024**2)):
        raise ValueError("Survey variational ensemble exceeds memory budget")
    driver = members.mean(axis=0).mean(axis=1) / scales
    channel = np.ptp(driver, axis=0).argmax()
    if options["hard"]:
        # Unequal site indices ensure the mesh distinguishes all hard seeds.
        driver_values = np.arange(len(points), dtype=float) + driver[:, channel] * 1e-9
    else:
        driver_values = driver[:, channel]
    fitted = adaptive_variational(bounds, points, driver_values, refinement, **options)
    cells = fitted["cells"]
    inside = np.array([cell["inside"] for cell in cells])
    frozen = [tuple(cell["mesh_index"]) for cell in cells]
    operator = np.full((len(cells), len(points)), np.nan)
    for index, basis in enumerate(np.eye(len(points))):
        fitted_basis = adaptive_variational(bounds, points, basis, refinement,
                                            _frozen_cells=frozen, **options)
        if not fitted_basis["history"][-1]["solver"]["converged"]:
            raise FloatingPointError("Variational interpolation basis failed to converge")
        operator[:, index] = [cell["center_mean"] for cell in fitted_basis["cells"]]
    draws = np.einsum("qs,mslf->mlqf", operator, members)
    tail = (1 - mass) / 2
    low, median, high = np.quantile(draws, [tail, .5, 1 - tail], axis=0)
    boxes = np.array([cell["bounds_m"] for cell in cells])
    return {"variational_bounds_m": boxes, "variational_centers_m": (boxes[:, :2] + boxes[:, 2:]) / 2,
            "variational_inside": inside, "variational_weights": operator, "variational_draws": draws,
            "variational_mean": draws.mean(axis=0), "variational_std": draws.std(axis=0),
            "variational_median": median, "variational_lower": low, "variational_upper": high,
            "variational_width": high - low}, {"method": "adaptive_variational",
                                              "draw_axes": ["member", "lead", "cell", "feature"],
                                              "refinement_channel": int(channel),
                                              "adaptive": refinement.initial_depth < refinement.max_depth,
                                              "boundary_policy": fitted["boundary_policy"],
                                              "hard": options["hard"]}
