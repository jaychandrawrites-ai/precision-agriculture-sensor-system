"""Block-spaced split conformal expansion of trained quantile intervals."""
from dataclasses import dataclass

import numpy as np

from Temporal.BlockIntervals import _quantile_rank, separated_indices


def _aligned(lower, upper, observed, scale):
    lo, hi, y, scale = (np.asarray(a, dtype=float) for a in (lower, upper, observed, scale))
    if (y.ndim != 3 or not y.size or lo.shape != y.shape or hi.shape != y.shape
            or scale.shape != y.shape[1:] or (lo > hi).any() or (scale <= 0).any()
            or not all(np.isfinite(a).all() for a in (lo, hi, y, scale))):
        raise ValueError("Expected finite ordered intervals and positive per-lead/feature scales")
    return lo, hi, y, scale


@dataclass(frozen=True)
class QuantileCalibration:
    scale: np.ndarray
    marginal: np.ndarray
    joint: float
    mass: float
    stride: int
    count: int

    @classmethod
    def fit(cls, lower, upper, observed, origins, scale, *, mass=.9, stride=None):
        lo, hi, y, scale = _aligned(lower, upper, observed, scale)
        stride = y.shape[1] if stride is None else stride
        if stride < y.shape[1] or len(origins) != len(y):
            raise ValueError("Calibration forecast windows must use disjoint target periods")
        keep = separated_indices(origins, stride)
        score = np.maximum(0, np.maximum(lo[keep] - y[keep], y[keep] - hi[keep])) / scale
        return cls(scale, _quantile_rank(score, mass), float(_quantile_rank(score.max(axis=(1, 2)), mass)),
                   mass, stride, len(keep))

    def score(self, lower, upper, observed, origins):
        lo, hi, y, _ = _aligned(lower, upper, observed, self.scale)
        if len(origins) != len(y):
            raise ValueError("Evaluation origins must align with targets")
        keep = separated_indices(origins, self.stride)
        lo, hi, y = lo[keep], hi[keep], y[keep]
        results = {}
        for name, radius in (("raw", 0.), ("marginal", self.scale * self.marginal),
                             ("simultaneous", self.scale * self.joint)):
            low, high = lo - radius, hi + radius
            inside = (low <= y) & (y <= high)
            width = high - low
            penalty = np.maximum(low - y, 0) + np.maximum(y - high, 0)
            results[name] = {"coverage": inside.mean(axis=0),
                             "simultaneous_coverage": float(inside.all(axis=(1, 2)).mean()),
                             "mean_width": width.mean(axis=0),
                             "mean_interval_score": (width + 2 * penalty / (1 - self.mass)).mean(axis=0)}
        return {"mass": self.mass, "calibration_origins": self.count, "test_origins": len(keep),
                "spacing_hours": self.stride, "results": results,
                "assumption": "Spacing avoids overlap. Exchangeability of successive weather regimes is unverified."}
