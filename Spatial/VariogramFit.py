"""Weighted least-squares variogram fitting over an explicit finite range search."""
from dataclasses import dataclass, replace

import numpy as np

from Spatial.Interpolation import covariance
from Spatial.Kriging import CovarianceModel


@dataclass(frozen=True)
class VariogramFit:
    model: CovarianceModel
    weighted_rmse: float
    bins_used: int
    candidates: int


def fit_variogram(empirical, *, shapes=("exponential", "spherical", "matern32"),
                  ranges=None, minimum_pairs=3):
    distance = np.asarray(empirical["distances"], dtype=float)
    gamma = np.asarray(empirical["semivariance"], dtype=float)
    count = np.asarray(empirical["pairs"])
    if (distance.ndim != 1 or distance.shape != gamma.shape or distance.shape != count.shape
            or minimum_pairs < 1 or not shapes):
        raise ValueError("Invalid empirical variogram bins")
    usable = (count >= minimum_pairs) & np.isfinite(distance) & np.isfinite(gamma) & (distance > 0)
    distance, gamma, count = distance[usable], gamma[usable], count[usable]
    if len(distance) < 3 or (gamma < 0).any() or not np.any(gamma > 0):
        raise ValueError("Need at least three populated positive-lag bins and nonzero variation")
    if ranges is None:
        ranges = np.geomspace(distance.min() / 2, distance.max() * 3, 64)
    ranges = np.asarray(ranges, dtype=float)
    if ranges.ndim != 1 or not len(ranges) or not np.isfinite(ranges).all() or (ranges <= 0).any():
        raise ValueError("Provide positive candidate covariance ranges")

    weights = count / count.sum()
    root_weight = np.sqrt(weights)
    best = None
    total = len(shapes) * len(ranges)
    for shape in shapes:
        for heading in ranges:
            model = CovarianceModel(float(heading), shape=shape)
            structure = 1.0 - covariance(distance / heading, shape)
            design = np.column_stack((np.ones(len(distance)), structure))
            target = root_weight * gamma
            coefficients = np.linalg.lstsq(design * root_weight[:, None], target, rcond=None)[0]
            if (coefficients < 0).any():
                only_structure = max(0.0, (weights * structure) @ gamma / (weights @ structure**2))
                candidates = ((0., only_structure), (weights @ gamma, 0.))
                coefficients = min(candidates, key=lambda item: weights @ (design @ item - gamma)**2)
            nugget, sill = coefficients
            sill = max(float(sill), np.finfo(float).eps * max(1.0, float(gamma.max())))
            score = float(np.sqrt(weights @ (design @ [nugget, sill] - gamma)**2))
            if best is None or score < best.weighted_rmse:
                best = VariogramFit(replace(model, sill=sill, nugget=float(nugget)), score,
                                    len(distance), total)
    return best
