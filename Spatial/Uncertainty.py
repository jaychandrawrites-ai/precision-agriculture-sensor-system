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
    ess: np.ndarray


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
    return Uncertainty(fields.mean(axis=(0, 1)), spread, lower, upper, rhat,
                       effective_sample_size(samples))


def effective_sample_size(samples):
    """Estimate per-cell ESS using pooled autocorrelations and positive paired lags.

    Constant cells have no mixing estimate. Compute one north row at a time so
    diagnostics need no full-grid complex copy of the sample array.
    """
    fields = np.asarray(samples.fields, dtype=float)
    chains, draws, height, width = fields.shape
    size = np.full((height, width), np.nan)
    padded = 1 << (2 * draws - 1).bit_length()
    pairs = draws // 2
    for row in range(height):
        strip = fields[:, :, row]
        centered = strip - strip.mean(axis=1, keepdims=True)
        transformed = np.fft.rfft(centered, n=padded, axis=1)
        covariance = np.fft.irfft(np.abs(transformed)**2, n=padded, axis=1)[:, :draws]
        covariance = covariance.mean(axis=0) / draws
        variance = covariance[0]
        correlation = np.divide(covariance, variance,
                                out=np.zeros_like(covariance), where=variance > 0)
        paired = correlation[:2 * pairs].reshape(pairs, 2, width).sum(axis=1)
        monotone = np.minimum.accumulate(paired, axis=0)
        positive = np.maximum(monotone, 0)
        time = np.maximum(1.0, -1 + 2 * positive.sum(axis=0))
        valid = (np.ptp(strip, axis=(0, 1)) > 0) & (variance > 0)
        size[row, valid] = (chains * draws / time)[valid]
    return size
