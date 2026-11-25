"""Compare spatial predictors on fixed independent locations and explicit field scenarios."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from Framework.Backend import create_backend
from Spatial.BlockSampler import BlockSampling, block_metropolis
from Spatial.EnergyFunctional import EnergyFunctional, EnergyWeights
from Spatial.GaussianField import exact_gaussian_field
from Spatial.Geostatistics import variogram
from Spatial.Interpolation import idw, radial_basis
from Spatial.Kriging import CovarianceModel, KrigingModel, coordinates, nearest_neighbor
from Spatial.PosteriorStudy import interval_scores, mixing_report, predictive_summary
from Spatial.SeedManager import seed_constraints
from Spatial.SpatialGrid import SpatialGrid
from Spatial.SpatialLoss import held_out_metrics
from Spatial.SpatialModel import reconstruct
from Spatial.Uncertainty import summarize
from Spatial.Validation import sample_grid
from Spatial.VariogramFit import fit_variogram
from config import SpatialConfig
from Utils.Serialization import read_json, write_json


def _metrics(prediction, values):
    return {name: float(score[0]) if np.isfinite(score[0]) else None
            for name, score in held_out_metrics(prediction, values).items()}


def compare_field(grid, points, values, queries, truth, model, *,
                  observation_variance=None, test_variance=None,
                  configurations=None, settings=BlockSampling(), label="user supplied"):
    points, queries = coordinates(points), coordinates(queries)
    values, truth = np.asarray(values, dtype=float), np.asarray(truth, dtype=float)
    if (values.shape != (len(points),) or truth.shape != (len(queries),)
            or not np.isfinite(values).all() or not np.isfinite(truth).all()):
        raise ValueError("Provide one finite value for each training and verification site")
    if (points[:, None] == queries[None, :]).all(axis=-1).any():
        raise ValueError("Training and verification sites must be spatially distinct")
    backend = create_backend(dtype="float64")
    covariances = (("ordinary", "constant"), ("universal", "affine"))
    estimates = {"nearest": nearest_neighbor(points, values, queries),
                 "idw": idw(points, values, queries),
                 "rbf": radial_basis(points, values, queries, model.range_m)}
    conditioning = {}
    for name, trend in covariances:
        try:
            fitted = KrigingModel(points, values, model, trend=trend,
                                  measurement_variance=observation_variance)
            prediction = fitted.predict(queries)
        except ValueError as error:
            if trend != "affine":
                raise
            conditioning[name] = {"available": False, "reason": str(error)}
        else:
            estimates[name] = prediction.mean
            conditioning[name] = {"available": True, "jitter": fitted.jitter,
                                  "mean_latent_variance": float(prediction.latent_variance.mean())}
    field = reconstruct(grid, points, values, backend, SpatialConfig()).field
    estimates["variational"] = sample_grid(grid, backend.to_host(field), queries)
    specifications = configurations if configurations is not None else (EnergyWeights(),)
    constraints = seed_constraints(grid, points, values, 1.0 if observation_variance is not None else 20.0,
                                   measurement_variance=observation_variance)
    scenarios, diagnostics = [], []
    for index, weights in enumerate(specifications):
        energy = EnergyFunctional(grid, constraints, backend, weights)
        samples = block_metropolis(energy, field, settings)
        scenarios.append(samples)
        summary = summarize(samples)
        report = {"gradient_weight": weights.gradient, "curvature_weight": weights.curvature,
                  "mixing": mixing_report(summary, active=~constraints.fixed),
                  "acceptance_mean": float(samples.acceptance.mean()),
                  "acceptance_min": float(samples.acceptance.min()),
                  "scale": samples.proposal_scales.tolist()}
        if field.size <= 256:
            exact = exact_gaussian_field(energy, settings.temperature)
            report["max_mean_error_vs_exact"] = float(np.max(np.abs(summary.mean - exact.mean)))
            report["max_variance_error_vs_exact"] = float(np.max(np.abs(summary.std**2 - exact.variance)))
        diagnostics.append(report)
    predictive = predictive_summary(grid, scenarios, queries, observation_variance=test_variance)
    estimates["block_posterior_mean"] = predictive.mean
    report = {"label": label, "training_sites": len(points), "test_sites": len(queries),
              "covariance": asdict(model), "kriging_conditioning": conditioning,
              "point_metrics": {name: _metrics(item, truth) for name, item in estimates.items()},
              "predictive_interval": interval_scores(predictive, truth),
              "mean_variance": {"conditional": float(predictive.conditional_variance.mean()),
                                "scenario": float(predictive.scenario_variance.mean()),
                                "observation": float(predictive.observation_variance.mean())},
              "posterior_diagnostics": diagnostics, "sampling": asdict(settings)}
    return report


def _synthetic():
    rng = np.random.default_rng(429)
    grid = SpatialGrid(np.arange(9.), np.arange(8.))
    locations = grid.points()
    ordered = rng.permutation(len(locations))
    fit, hold = ordered[:24], ordered[24:]
    model = CovarianceModel(3.0, 1.0, shape="matern32", minor_ratio=0.7)
    latent = (2 + 0.30 * locations[:, 0] - 0.15 * locations[:, 1] +
              rng.multivariate_normal(np.zeros(len(locations)), model(locations, locations)))
    variance = np.where(locations[:, 0] < 4, 0.04, 0.09)
    observed = latent + rng.normal(scale=np.sqrt(variance))
    return compare_field(grid, locations[fit], observed[fit], locations[hold], observed[hold],
                         model, observation_variance=variance[fit], test_variance=variance[hold],
                         configurations=(EnergyWeights(0.75, 0.10), EnergyWeights(1.25, 0.25)),
                         settings=BlockSampling(tile=3, draws=500, burn_in=200, seed=85),
                         label="synthetic Gaussian field with known measurement noise")


def _input(source):
    points = source["fit"]["coordinates"]
    values = source["fit"]["values"]
    model = source.get("covariance")
    if model is None:
        fitted = fit_variogram(variogram(points, values, source["variogram_edges"]))
        model = fitted.model
    else:
        model = CovarianceModel(**model)
    grid = SpatialGrid(source["grid"]["east"], source["grid"]["north"])
    phases = source.get("scenarios")
    if phases is not None:
        phases = tuple(EnergyWeights(**phase) for phase in phases)
    return compare_field(grid, points, values, source["test"]["coordinates"],
                         source["test"]["values"], model,
                         observation_variance=source["fit"].get("variance"),
                         test_variance=source["test"].get("variance"),
                         configurations=phases, settings=BlockSampling(**source.get("sampling", {})),
                         label=source.get("label", "user supplied"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = _synthetic() if args.synthetic else _input(read_json(args.input))
    if args.output is not None:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
