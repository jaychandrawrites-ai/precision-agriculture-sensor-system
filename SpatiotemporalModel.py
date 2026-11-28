"""Compose independent temporal forecasting and measured-site spatial reconstruction."""
from dataclasses import dataclass

import numpy as np

from Spatial.Interpolation import idw, ordinary_kriging
from Spatial.SpatialGrid import local_coordinates
from Spatial.SpatialModel import reconstruct_variational
from Spatial.MonteCarlo import metropolis
from Spatial.Uncertainty import summarize
from Spatial.RobustField import RobustField
from Spatial.SeedManager import seed_constraints
from Spatial.VariationalSolver import VariationalResult


@dataclass
class SpatialPrediction:
    field: np.ndarray
    idw: np.ndarray
    kriging: np.ndarray | None
    solver: object
    uncertainty: object | None
    sampling: object | None


@dataclass(frozen=True)
class SpatiotemporalModel:
    forecaster: object | None = None

    def forecast(self, history, steps, *, origin=None):
        if self.forecaster is None:
            raise ValueError("A saved temporal checkpoint is required")
        return self.forecaster.forecast(history, steps, origin=origin)

    def reconstruct(self, grid, coordinates, observations, origin, backend, spatial_settings,
                    *, kriging_range_m=None, sampling=None, hard_seeds=False,
                    boundary_values=None, boundary_weight=None, boundary="neumann",
                    algorithm="descent", robust=None):
        points = local_coordinates(coordinates, origin)
        if len(np.unique(points, axis=0)) < 2:
            raise ValueError("Spatial reconstruction requires at least two distinct measured sites")
        shape = grid.shape
        baseline = idw(points, observations, grid.points()).reshape(shape)
        if algorithm == "robust_lbfgs":
            if sampling is not None or boundary != "neumann":
                raise ValueError("Robust fitting requires nonperiodic boundaries and quadratic sampling disabled")
            constraints = seed_constraints(
                grid, points, observations, spatial_settings.seed_weight, hard_seeds,
                boundary_values, boundary_weight
            )
            config = robust or {}
            field_energy = RobustField(
                grid, config.get("east_conductance", spatial_settings.gradient_weight),
                config.get("north_conductance", spatial_settings.gradient_weight),
                constraints.targets, constraints.weights,
                curvature=spatial_settings.curvature_weight,
                delta=config.get("fidelity_delta", 1.0)
            )
            solution, optimization = field_energy.fit(
                baseline, constraints.fixed, constraints.fixed_values,
                iterations=spatial_settings.iterations
            )
            _, derivative = field_energy.value_gradient(solution)
            free = derivative[~constraints.fixed]
            gradient_rms = float(np.sqrt(np.mean(free**2)))
            result = VariationalResult(backend.array(solution), optimization.loss_trace,
                                       len(optimization.loss_trace) - 1,
                                       optimization.status == "converged", gradient_rms, 0.)
        else:
            if robust is not None:
                raise ValueError("Robust options require the robust_lbfgs solver")
            energy, result = reconstruct_variational(grid, points, observations, backend,
                                                     spatial_settings, hard_seeds=hard_seeds,
                                                     boundary_values=boundary_values,
                                                     boundary_weight=boundary_weight, boundary=boundary,
                                                     algorithm=algorithm)
        kriging = (ordinary_kriging(points, observations, grid.points(), kriging_range_m).reshape(shape)
                   if kriging_range_m is not None else None)
        samples = metropolis(energy, result.field, sampling) if sampling is not None else None
        return SpatialPrediction(backend.to_host(result.field), baseline, kriging, result,
                                 summarize(samples) if samples is not None else None, samples)
