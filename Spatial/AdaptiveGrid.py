"""Budgeted quadtree refinement for local-metric predictions."""
from dataclasses import asdict, dataclass
from collections import deque

import numpy as np


@dataclass(frozen=True)
class Refinement:
    max_depth: int = 5
    min_width_m: float = 1.
    max_leaves: int = 1024
    max_variation: float | None = None
    max_standard_deviation: float | None = None
    observation_radius_m: float | None = None

    def __post_init__(self):
        if (type(self.max_depth) is not int or not 0 <= self.max_depth <= 15
                or type(self.max_leaves) is not int or not 1 <= self.max_leaves <= 20000):
            raise ValueError("Invalid adaptive grid depth or leaf budget")
        for value in (self.min_width_m, self.max_variation, self.max_standard_deviation,
                      self.observation_radius_m):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("Refinement distances and thresholds must be finite and positive")


def _polygon_mask(points, polygon):
    if polygon is None:
        return np.ones(len(points), dtype=bool)
    inside = np.zeros(len(points), dtype=bool)
    boundary = np.zeros(len(points), dtype=bool)
    x, y = points.T
    for (ax, ay), (bx, by) in zip(polygon, np.roll(polygon, -1, axis=0)):
        dx, dy = bx - ax, by - ay
        if dx == dy == 0:
            continue
        cross = (x - ax) * dy - (y - ay) * dx
        boundary |= ((np.abs(cross) <= 1e-10 * np.hypot(dx, dy))
                     & (x >= min(ax, bx)) & (x <= max(ax, bx))
                     & (y >= min(ay, by)) & (y <= max(ay, by)))
        if dy != 0:
            inside ^= ((ay > y) != (by > y)) & (x < ax + dx * (y - ay) / dy)
    return inside | boundary


def refine(bounds, predictor, settings=Refinement(), *, observed_xy=None, polygon=None):
    """Refine where the model varies, spreads, or meets declared support.

    predictor receives a finite array of local metric points and returns
    (mean, variance). Output cells are masked by their centers, not clipped
    exactly to the polygon, and make no claim of observed subcell detail.
    """
    limits = np.asarray(bounds, dtype=float)
    if limits.shape != (4,) or not np.isfinite(limits).all():
        raise ValueError("Bounds are [west, south, east, north] in local metres")
    west, south, east, north = limits
    if not west < east or not south < north:
        raise ValueError("Invalid field bounds")
    sites = np.empty((0, 2)) if observed_xy is None else np.asarray(observed_xy, dtype=float)
    if sites.ndim != 2 or sites.shape[1] != 2 or not np.isfinite(sites).all():
        raise ValueError("Observation coordinates must be finite east/north pairs")
    if polygon is not None:
        polygon = np.asarray(polygon, dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError("Provide a finite field polygon with at least three vertices")
    queue, leaves, cache = deque([(west, south, east, north, 0)]), [], {}
    allocated, budget_limited = 1, 0
    while queue:
        lo_x, lo_y, hi_x, hi_y, depth = queue.popleft()
        dx, dy = hi_x - lo_x, hi_y - lo_y
        x, y = lo_x + dx / 2, lo_y + dy / 2
        keys = [(x, y), (lo_x, lo_y), (lo_x, hi_y), (hi_x, lo_y), (hi_x, hi_y)]
        unseen = [key for key in keys if key not in cache]
        if unseen:
            mean, variance = (np.asarray(item, dtype=float) for item in predictor(np.asarray(unseen)))
            if (mean.shape != (len(unseen),) or variance.shape != mean.shape or (variance < 0).any()
                    or not all(np.isfinite(item).all() for item in (mean, variance))):
                raise ValueError("Predictor must return finite scalar means and nonnegative variances")
            cache.update({key: (float(m), float(v)) for key, m, v in zip(unseen, mean, variance)})
        values, spread = np.array([cache[key] for key in keys]).T
        mask = _polygon_mask(np.asarray(keys), polygon)
        lower = np.maximum(np.array([lo_x, lo_y]) - sites, 0)
        upper = np.maximum(sites - np.array([hi_x, hi_y]), 0)
        distance = np.linalg.norm(lower + upper, axis=1)
        reasons = []
        if settings.max_variation is not None and np.ptp(values) > settings.max_variation:
            reasons.append("variation")
        if settings.max_standard_deviation is not None and np.sqrt(spread).max() > settings.max_standard_deviation:
            reasons.append("uncertainty")
        if settings.observation_radius_m is not None and (distance <= settings.observation_radius_m).any():
            reasons.append("near_measurement")
        if mask.any() and not mask.all():
            reasons.append("boundary")
        eligible = bool(reasons and depth < settings.max_depth and min(dx, dy) >= 2 * settings.min_width_m)
        if eligible and allocated + 3 <= settings.max_leaves:
            allocated += 3
            queue.extend((left, bottom, right, top, depth + 1)
                         for left, right in ((lo_x, x), (x, hi_x)) for bottom, top in ((lo_y, y), (y, hi_y)))
        else:
            budget_limited += int(eligible)
            leaves.append({"bounds_m": [lo_x, lo_y, hi_x, hi_y], "depth": depth, "inside": bool(mask[0]),
                           "center_mean": float(values[0]), "center_variance": float(spread[0]),
                           "remaining_refinement_drivers": reasons})
    return {"cells": leaves, "settings": asdict(settings), "leaf_count": len(leaves),
            "prediction_sites": len(cache), "budget_limited_cells": budget_limited,
            "maximum_depth": max(cell["depth"] for cell in leaves)}
