"""Paired moving-block bootstrap for equally spaced forecast origins."""
import numpy as np


def paired_rmse_difference(first, second, truth, *, block, draws=1000, seed=42):
    """Bootstrap each forecast lead independently with shared origin indices."""
    a, b, y = (np.asarray(value, dtype=float) for value in (first, second, truth))
    if (a.shape != b.shape or a.shape != y.shape or a.ndim < 1 or not 1 <= block <= len(a)
            or not all(np.isfinite(value).all() for value in (a, b, y)) or draws < 2):
        raise ValueError("Expected aligned forecasts, valid draw budget and block length")
    if len(a) < 2 * block:
        raise ValueError("At least two full blocks are required")
    squared_a, squared_b = (a - y)**2, (b - y)**2
    n = len(a)
    rng = np.random.default_rng(seed)
    offsets = np.arange(block)
    statistics = []
    for _ in range(draws):
        starts = rng.integers(0, n - block + 1, size=(n + block - 1) // block)
        indices = (starts[:, None] + offsets).ravel()[:n]
        statistics.append(np.sqrt(squared_a[indices].mean(axis=0)) -
                          np.sqrt(squared_b[indices].mean(axis=0)))
    lower, upper = np.quantile(statistics, [.025, .975], axis=0)
    return {"difference": np.sqrt(squared_a.mean(axis=0)) - np.sqrt(squared_b.mean(axis=0)),
            "lower": lower, "upper": upper, "blocks_hours": block, "draws": draws}
