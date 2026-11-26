"""Portable ordered direct quantile forecasts with manual pinball gradients."""
import numpy as np

from Framework.Backend import Backend, BackendConfig
from Framework.Layers import ActivationLayer, Dense, Sequential
from Framework.Optimizer import Adam
from Utils.Serialization import read_json, write_json


class QuantileForecaster:
    def __init__(self, backend, center, scale, horizon, levels=(.05, .5, .95), width=24):
        self.backend = backend
        self.center, self.scale = np.asarray(center, dtype=float), np.asarray(scale, dtype=float)
        self.levels = np.asarray(levels, dtype=float)
        if (self.center.shape != (6,) or self.scale.shape != (6,) or (self.scale <= 0).any()
                or not all(np.isfinite(x).all() for x in (self.center, self.scale, self.levels))
                or self.levels.ndim != 1 or not len(self.levels)
                or not 0 < self.levels[0] <= self.levels[-1] < 1 or (np.diff(self.levels) <= 0).any()
                or type(horizon) is not int or not 1 <= horizon <= 72 or type(width) is not int or width < 1):
            raise ValueError("Invalid quantile architecture or scaling")
        self.horizon, self.width = horizon, width
        self.network = Sequential([Dense(backend, 18, width), ActivationLayer(backend, "tanh"),
                                   Dense(backend, width, horizon * 6 * len(self.levels))])
        self.optimizer = Adam(self.network.parameters(), backend, learning_rate=.005, max_norm=5.)

    def forward(self, histories):
        backend = self.backend
        h = backend.array(histories)
        if h.ndim != 3 or h.shape[1:] != (3, 6) or not len(h):
            raise ValueError("Expected batch × three chronological observations × six features")
        backend.require_finite(h)
        scaled = (h - backend.array(self.center)) / backend.array(self.scale)
        raw, cache = self.network.forward(scaled.reshape(-1, 18))
        raw = raw.reshape(-1, self.horizon, 6, len(self.levels))
        xp = backend.xp
        increments = xp.logaddexp(0, raw[..., 1:])
        ordered = xp.concatenate((raw[..., :1], raw[..., :1] + xp.cumsum(increments, axis=-1)), axis=-1)
        return ordered, raw, cache

    def objective(self, histories, targets, *, gradients=False):
        backend = self.backend
        with backend.scope():
            ordered, raw, cache = self.forward(histories)
            target = backend.array(targets)
            if target.shape != ordered.shape[:-1]:
                raise ValueError("Expected batch × lead × six verification targets")
            backend.require_finite(target)
            scaled = (target - backend.array(self.center)) / backend.array(self.scale)
            error = scaled[..., None] - ordered
            xp = backend.xp
            levels = backend.array(self.levels)
            loss = xp.maximum(levels * error, (levels - 1) * error).mean()
            if gradients:
                derivative = xp.where(error > 0, -levels, xp.where(error < 0, 1 - levels, 0.))
                suffix = xp.flip(xp.cumsum(xp.flip(derivative, axis=-1), axis=-1), axis=-1)
                suffix[..., 1:] *= xp.exp(-xp.logaddexp(0, -raw[..., 1:]))
                self.optimizer.zero_grad()
                self.network.backward(suffix.reshape(len(target), -1) / ordered.size, cache)
            return float(backend.scalar(loss))

    def step(self, histories, targets):
        loss = self.objective(histories, targets, gradients=True)
        self.optimizer.step()
        return loss

    def predict(self, histories, steps=None):
        steps = self.horizon if steps is None else steps
        if type(steps) is not int or not 1 <= steps <= self.horizon:
            raise ValueError("Requested lead is outside the checkpoint horizon")
        with self.backend.scope():
            values, _, _ = self.forward(histories)
            return self.backend.to_host(values[:, :steps]) * self.scale[:, None] + self.center[:, None]

    def save(self, path):
        write_json(path, dict(format="jupiter-quantile-v1", center=self.center, scale=self.scale,
                              horizon=self.horizon, levels=self.levels, width=self.width,
                              dtype=str(self.backend.dtype), network=self.network.state_dict(self.backend)))

    @classmethod
    def load(cls, path, device="cpu"):
        state = read_json(path)
        if state.pop("format") != "jupiter-quantile-v1":
            raise ValueError("Expected a Jupiter quantile checkpoint")
        network = state.pop("network")
        backend = Backend(BackendConfig(device=device, dtype=state.pop("dtype")))
        forecaster = cls(backend, **state)
        forecaster.network.load_state_dict({key: backend.array(value) for key, value in network.items()}, backend)
        return forecaster


def quantile_scores(predictions, observations, levels):
    q, y, levels = (np.asarray(x, dtype=float) for x in (predictions, observations, levels))
    if (y.ndim != 3 or not len(y) or levels.ndim != 1 or not len(levels)
            or q.shape != (*y.shape, len(levels)) or not 0 < levels[0] <= levels[-1] < 1
            or (np.diff(levels) <= 0).any() or (np.diff(q, axis=-1) < 0).any()
            or not all(np.isfinite(x).all() for x in (q, y, levels))):
        raise ValueError("Expected finite ordered quantile trajectories and aligned observations")
    error = y[..., None] - q
    low, high = q[..., 0], q[..., -1]
    inside = (low <= y) & (y <= high)
    return {"pinball": np.maximum(levels * error, (levels - 1) * error).mean(axis=0),
            "fraction_below": (y[..., None] <= q).mean(axis=0),
            "coverage": inside.mean(axis=0), "mean_width": (high - low).mean(axis=0),
            "simultaneous_coverage": inside.all(axis=(1, 2)).mean(),
            "nominal_interval_mass": levels[-1] - levels[0], "origins": len(y)}
