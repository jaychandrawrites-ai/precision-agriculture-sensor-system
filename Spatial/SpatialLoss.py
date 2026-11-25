"""Spatial target errors at independently measured held-out sites."""
import numpy as np

from Temporal.Evaluator import evaluate


def held_out_metrics(predictions, targets):
    y = np.asarray(targets, dtype=float)
    p = np.asarray(predictions, dtype=float)
    if y.ndim != 1 or p.shape != y.shape:
        raise ValueError("Expected one spatial prediction per measured target")
    return evaluate(p[:, None], y[:, None])
