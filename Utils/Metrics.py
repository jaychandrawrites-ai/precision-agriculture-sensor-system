"""Metrics for six-feature physical forecasts and normalized comparisons."""
import numpy as np

from config import FEATURES
from Temporal.Evaluator import evaluate


def physical_metrics(metrics):
    return {
        feature: {metric: float(values[i]) if np.isfinite(values[i]) else None
                  for metric, values in metrics.items()}
        for i, feature in enumerate(FEATURES)
    }


def horizon_metrics(predictions, targets, baseline=None):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.shape != targets.shape or targets.ndim != 3 or targets.shape[2] != len(FEATURES):
        raise ValueError("Expected matching [origin, horizon, feature] forecasts and targets")
    if baseline is not None and np.shape(baseline) != predictions.shape:
        raise ValueError("Baseline must match forecasts")
    return [physical_metrics(evaluate(predictions[:, i], targets[:, i],
                                      None if baseline is None else baseline[:, i]))
            for i in range(targets.shape[1])]
