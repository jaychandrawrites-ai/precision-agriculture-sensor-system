"""Combine losses from independent temporal targets and measured spatial targets."""
from dataclasses import dataclass
import math

from Temporal.Losses import regression_loss


@dataclass(frozen=True)
class CombinedResult:
    loss: float
    temporal: float | None
    spatial: float | None
    temporal_gradient: object
    spatial_gradient: object


def combined_loss(backend, *, temporal=None, spatial=None, temporal_weight=1.0,
                  spatial_weight=1.0, kind="huber"):
    """Each enabled term is (prediction, target, positive feature scales)."""
    if not all(math.isfinite(w) and w >= 0 for w in (temporal_weight, spatial_weight)):
        raise ValueError("Invalid objective weights")

    def component(pair, weight):
        if pair is None or weight == 0:
            return None, None
        prediction, target, scales = (backend.array(array) for array in pair)
        if prediction.shape != target.shape or prediction.ndim != 2 or scales.shape != prediction.shape[1:]:
            raise ValueError("Expected aligned 2D targets and one scale per feature")
        backend.require_finite(scales, "target scales")
        if backend.scalar(scales.min()) <= 0:
            raise ValueError("Target scales must be positive")
        residual = (prediction - target) / scales
        value, gradient = regression_loss(residual, backend.xp.zeros_like(residual), backend, kind=kind)
        return value, weight * gradient / scales

    with backend.scope():
        t, dt = component(temporal, temporal_weight)
        s, ds = component(spatial, spatial_weight)
    if t is None and s is None:
        raise ValueError("Supply at least one active source of targets")
    return CombinedResult(temporal_weight * (t or 0) + spatial_weight * (s or 0), t, s, dt, ds)
