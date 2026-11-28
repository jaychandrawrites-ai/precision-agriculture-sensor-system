"""Causal direct reconstruction from real-paired complex modal coefficients."""
from dataclasses import dataclass
import math

import numpy as np

from Data.Derivatives import lag_coefficients
from Physics.Constraints import LOWER, UPPER, project

PERIODS = (72, 36, 24, 18, 12, 8, 6)
ANNUAL_HOURS = 365.2425 * 24
HORIZON = 72


@dataclass(frozen=True)
class Modes:
    periods: tuple = PERIODS
    damped: bool = False

    def __post_init__(self):
        periods = tuple(self.periods)
        if not periods or len(set(periods)) != len(periods) or any(p not in PERIODS for p in periods):
            raise ValueError("Expected distinct supported spectral periods")
        object.__setattr__(self, "periods", periods)

    @property
    def width(self):
        return 1 + 2 * (len(self.periods) + int(self.damped))

    def basis(self, hours):
        hours = np.asarray(hours, dtype=float).reshape(-1)
        columns = [np.ones_like(hours)]
        for period in (*self.periods, *((18,) if self.damped else ())):
            phase = 2 * np.pi * hours / period
            columns.extend((np.cos(phase), -np.sin(phase)))
        if self.damped:
            decay = np.exp(-hours / 12)
            columns[-2] *= decay
            columns[-1] *= decay
        return np.column_stack(columns)

    def anchored_basis(self, hours):
        hours = np.asarray(hours)
        return self.basis(hours) - np.exp(-hours[:, None] / 3) * self.basis([0])


def features(histories, origins):
    history = np.asarray(histories, dtype=float)
    times = np.asarray(origins, dtype="datetime64[ns]")
    if (history.ndim != 3 or history.shape[1:] != (3, 6) or times.shape != (len(history),)
            or np.isnat(times).any() or not np.isfinite(history).all()):
        raise ValueError("Spectral inference needs three consecutive hourly states and their UTC origin")
    hours = (times - np.datetime64("1970-01-01")) / np.timedelta64(1, "h")
    day = 2 * np.pi * (hours % 24) / 24
    year = 2 * np.pi * (hours % ANNUAL_HOURS) / ANNUAL_HOURS
    return np.column_stack(((lag_coefficients(.65) @ history).reshape(-1, 18),
                            np.sin(day), np.cos(day), np.sin(year), np.cos(year)))


@dataclass(frozen=True)
class LossWeights:
    value: float = 1.
    spectral: float = .03
    derivative: float = .25
    physical: float = .1
    regularization: float = .002
    level: float = 0.
    shape: float = 0.
    curvature: float = 0.

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in vars(self).values()) or not self.value:
            raise ValueError("Loss weights must be finite and nonnegative, with positive value weight")


def loss(coefficients, target, anchor, basis, reference, lower, upper, weights, xp=np):
    """Objective and coefficient gradient in normalized state units."""
    predicted = anchor + basis @ coefficients
    error = predicted - target
    residual = predicted - xp.clip(predicted, lower, upper)
    velocity = xp.diff(error, axis=1)
    curvature = xp.diff(velocity, axis=1)
    level = xp.mean(error, axis=1, keepdims=True)
    shape = error - level
    coefficient_error = coefficients - reference
    # Downweight weakly identified coefficients and penalize higher frequencies gradually.
    identified = xp.mean(basis**2, axis=0)[:, None]
    penalty = xp.mean(xp.diff(basis, axis=0)**2, axis=0)[:, None]
    terms = {
        "value": xp.mean(error**2),
        "level": xp.mean(level**2),
        "shape": xp.mean(shape**2),
        "curvature": xp.mean(curvature**2),
        "spectral": xp.mean(identified * coefficient_error**2),
        "derivative": xp.mean(velocity**2),
        "physical": xp.mean(residual**2),
        "regularization": xp.mean(penalty * coefficients**2),
    }
    state_gradient = 2 * (weights.value * error + weights.physical * residual
                          + weights.level * level + weights.shape * shape) / error.size
    bending = 2 * weights.curvature * curvature / curvature.size
    state_gradient[:, 2:] += bending
    state_gradient[:, 1:-1] -= 2 * bending
    state_gradient[:, :-2] += bending
    derivative = 2 * weights.derivative * velocity / velocity.size
    state_gradient[:, 1:] += derivative
    state_gradient[:, :-1] -= derivative
    gradient = basis.T @ state_gradient + 2 * (
        weights.spectral * identified * coefficient_error
        + weights.regularization * penalty * coefficients) / coefficients.size
    return sum(getattr(weights, key) * value for key, value in terms.items()), gradient, terms


def sparse_events(rng, start, shape, scale, bound):
    """Each flattened normalized training input consumes one index, starting at one."""
    if type(start) is not int or start < 1 or not math.isfinite(scale) or scale < 0 or not math.isfinite(bound) or bound <= 0:
        raise ValueError("Invalid sparse event settings")
    x = np.arange(start, start + math.prod(shape), dtype=float)
    hits = rng.random(len(x)) < 1 / x
    signs = rng.choice((-1, 1), len(x))
    return (hits * signs * np.minimum(scale * x, bound)).reshape(shape), int(hits.sum())


class SpectralForecaster:
    def __init__(self, model, backend, modes, inputs, targets):
        self.model = model
        self.backend = backend
        self.modes = modes
        self.inputs = inputs
        self.targets = targets

    def coefficients(self, histories, origins):
        inputs = self.inputs.transform(features(histories, origins))
        output, _ = self.model.forward(self.backend.array(inputs), training=False)
        result = self.backend.to_host(output).reshape(-1, self.modes.width, 6)
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite spectral coefficients")
        return result

    def forecast(self, histories, origins, steps, *, raw=False):
        if type(steps) is not int or not 1 <= steps <= HORIZON:
            raise ValueError("Spectral forecast requires an integer horizon from 1 through 72 hours")
        coefficients = self.coefficients(histories, origins)
        leads = np.arange(1, steps + 1)
        change = self.modes.anchored_basis(leads) @ coefficients * self.targets.scale
        prediction = histories[:, -1:, :] + change
        return prediction if raw else project(prediction)
