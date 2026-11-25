"""Compose independent temporal forecasting and measured-site spatial reconstruction."""
from dataclasses import dataclass

import numpy as np

from Spatial.Interpolation import idw, ordinary_kriging
from Spatial.SpatialGrid import local_coordinates
from Spatial.SpatialModel import reconstruct_variational
from Spatial.MonteCarlo import metropolis
from Spatial.Uncertainty import summarize


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

    def forecast(self, history, steps):
        if self.forecaster is None:
            raise ValueError("A saved temporal checkpoint is required")
        return self.forecaster.forecast(history, steps)

    def reconstruct(self, grid, coordinates, observations, origin, backend, spatial_settings,
                    *, kriging_range_m=None, sampling=None, hard_seeds=False):
        points = local_coordinates(coordinates, origin)
        if len(np.unique(points, axis=0)) < 2:
            raise ValueError("Spatial reconstruction requires at least two distinct measured sites")
        energy, result = reconstruct_variational(grid, points, observations, backend,
                                                 spatial_settings, hard_seeds=hard_seeds)
        shape = grid.shape
        baseline = idw(points, observations, grid.points()).reshape(shape)
        kriging = (ordinary_kriging(points, observations, grid.points(), kriging_range_m).reshape(shape)
                   if kriging_range_m is not None else None)
        samples = metropolis(energy, result.field, sampling) if sampling is not None else None
        return SpatialPrediction(backend.to_host(result.field), baseline, kriging, result,
                                 summarize(samples) if samples is not None else None, samples)
