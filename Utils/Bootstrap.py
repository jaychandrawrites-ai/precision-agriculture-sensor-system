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


def paired_calendar_blocks(first, second, truth, origins, *, block_hours=168, draws=4000, seed=42):
    """Paired cluster bootstrap retaining dependence inside each UTC time block.

    Whole blocks are resampled and weighted by their eligible origin counts.
    Bonferroni intervals cover the event family across supplied leads and
    features nominally, subject to the independent-block approximation.
    """
    a, b, y = (np.asarray(value, dtype=float) for value in (first, second, truth))
    times = np.asarray(origins, dtype="datetime64[ns]")
    if (a.shape != b.shape or a.shape != y.shape or a.ndim != 3 or not len(a)
            or not a.shape[1] or not a.shape[2] or times.shape != (len(a),) or np.isnat(times).any()
            or (np.diff(times) <= np.timedelta64(0, "ns")).any()
            or not all(np.isfinite(value).all() for value in (a, b, y))
            or type(block_hours) is not int or block_hours < a.shape[1]
            or type(draws) is not int or draws < 100):
        raise ValueError("Expected aligned lead/feature errors and ordered origins")
    elapsed = (times - times[0]) / np.timedelta64(1, "h")
    labels = np.floor(elapsed / block_hours).astype(int)
    blocks, first_row, counts = np.unique(labels, return_index=True, return_counts=True)
    if len(counts) < 8:
        raise ValueError("At least eight nonempty calendar blocks are required")
    shape = a.shape[1:]
    sq_a, sq_b = (np.add.reduceat((estimate - y)**2, first_row, axis=0).reshape(len(counts), -1)
                  for estimate in (a, b))
    rng = np.random.default_rng(seed)
    selections = rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), size=draws)
    sizes = selections @ counts
    differences = np.sqrt((selections @ sq_a) / sizes[:, None])
    differences -= np.sqrt((selections @ sq_b) / sizes[:, None])
    tail = .025 / (shape[0] * shape[1])
    if draws * tail < 2:
        raise ValueError("Increase draws for informative Bonferroni tail estimates")
    intervals = np.quantile(differences, [.025, .975, tail, 1-tail], axis=0).reshape((4, *shape))
    effect = np.sqrt(sq_a.sum(axis=0) / len(a)) - np.sqrt(sq_b.sum(axis=0) / len(a))
    return {"difference": effect.reshape(shape), "pointwise_lower": intervals[0],
            "pointwise_upper": intervals[1], "family_lower": intervals[2], "family_upper": intervals[3],
            "nominal_mass": .95, "family_size": shape[0] * shape[1], "blocks": len(blocks),
            "block_hours": block_hours, "origins_per_block": counts, "draws": draws}


def seasonal_temperature_comparison(forecast, daily, targets, origins):
    """Predeclared temperature leads, with inference omitted when coverage is short."""
    leads = [h for h in (1, 3, 6, 12, 24, 48, 72) if h <= targets.shape[1]]
    indices = np.array(leads) - 1
    try:
        result = paired_calendar_blocks(forecast[:, indices, :1], daily[:, indices, :1],
                                        targets[:, indices, :1], origins)
    except ValueError as cause:
        return {"leads_hours": leads, "unavailable": str(cause)}
    return {"leads_hours": leads, "contrast": "seasonal blend minus daily persistence temperature RMSE",
            **result}
