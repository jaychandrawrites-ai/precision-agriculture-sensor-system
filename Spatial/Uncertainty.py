"""Conditional uncertainty maps and a simple split-chain mixing diagnostic."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Uncertainty:
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    split_rhat: np.ndarray


def summarize(samples, credible_mass=0.95):
    fields = np.asarray(samples.fields, dtype=float)
    if (fields.ndim != 4 or fields.shape[0] < 1 or fields.shape[1] < 2
            or not np.isfinite(fields).all() or not 0 < credible_mass < 1):
        raise ValueError("Expected finite [chain, draw, north, east] samples and a credible mass")
    tails = (1 - credible_mass) / 2
    lower, upper = np.quantile(fields, [tails, 1 - tails], axis=(0, 1))
    rhat = np.full(fields.shape[2:], np.nan)
    half = fields.shape[1] // 2
    if half >= 2 and fields.shape[0] >= 2:
        split = np.concatenate((fields[:, :half], fields[:, -half:]), axis=0)
        within = split.var(axis=1, ddof=1).mean(axis=0)
        means = split.mean(axis=1)
        between = half * means.var(axis=0, ddof=1)
        valid = within > 0
        ratio = np.divide(between, within, out=np.zeros_like(within), where=valid)
        rhat[valid] = np.sqrt((half - 1 + ratio[valid]) / half)
        rhat[(~valid) & (between > 0)] = np.inf
    spread = fields.std(axis=(0, 1), ddof=1)
    spread[np.ptp(fields, axis=(0, 1)) == 0] = 0
    return Uncertainty(fields.mean(axis=(0, 1)), spread, lower, upper, rhat)
