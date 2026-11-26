"""Optional training on paired field observations with manual spatial gradients."""
from dataclasses import dataclass
import math

from CombinedLoss import combined_loss
from Framework.Optimizer import Adam
from Temporal.Model import TemporalModel
from Utils.Serialization import read_json, write_json


@dataclass(frozen=True)
class JointWeights:
    temporal: float = 1.
    spatial: float = 1.
    physics: float = .1
    regularization: float = .01

    def __post_init__(self):
        if not all(math.isfinite(w) and w >= 0 for w in vars(self).values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        if self.temporal + self.spatial == 0:
            raise ValueError("At least one observation term must be enabled")


class JointTrainer:
    """Train once from aligned issue times with independent survey targets.

    histories is issue × source site × 18, temporal targets is issue ×
    source site × six and spatial targets is issue × survey site × six.
    Survey queries use one externally supplied geometric operator. Incidence
    rows are weighted differences between neighboring survey sites.
    """

    def __init__(self, model, feature_scales, *, weights=JointWeights(), learning_rate=.003):
        self.model = model
        self.backend = model.backend
        if self.backend.is_gpu and model.dropout:
            raise ValueError("Exact stochastic joint training recovery requires CPU")
        self.scales = self.backend.array(feature_scales)
        self.backend.require_finite(self.scales, "feature scales")
        if self.scales.shape != (6,) or self.backend.scalar(self.scales.min()) <= 0:
            raise ValueError("Provide six positive feature scales")
        self.weights = weights
        self.optimizer = Adam(model.parameters(), self.backend, learning_rate=learning_rate, max_norm=5.)

    def objective(self, histories, temporal_targets, spatial_targets, operator, incidence,
                  *, gradients=False):
        backend, weights = self.backend, self.weights
        with backend.scope():
            x, y, z, mapping, edges = map(
                backend.array, (histories, temporal_targets, spatial_targets, operator, incidence))
            if x.ndim != 3 or x.shape[-1] != 18 or not all(x.shape[:2]):
                raise ValueError("Histories must be issue × source site × 18")
            times, sources, _ = x.shape
            if (y.shape != (times, sources, 6) or mapping.ndim != 2
                    or mapping.shape[1] != sources or not mapping.shape[0]
                    or z.shape != (times, mapping.shape[0], 6)
                    or edges.ndim != 2 or edges.shape[1] != mapping.shape[0]):
                raise ValueError("Unaligned temporal, spatial or geometry arrays")
            for array in (x, y, z, mapping, edges):
                backend.require_finite(array)
            xp = backend.xp
            if not bool(backend.scalar(xp.allclose(mapping.sum(axis=1), 1.))):
                raise ValueError("Interpolation operator must reproduce constants")
            if not bool(backend.scalar(xp.allclose(edges.sum(axis=1), 0.))):
                raise ValueError("Incidence rows must annihilate spatial constants")
            raw, cache = self.model.forward(x.reshape(-1, 18), training=gradients)
            source = raw.reshape(times, sources, 6)
            field = mapping @ source
            pair_loss = combined_loss(
                backend, temporal=(raw, y.reshape(-1, 6), self.scales),
                spatial=(field.reshape(-1, 6), z.reshape(-1, 6), self.scales),
                temporal_weight=weights.temporal, spatial_weight=weights.spatial, kind="mse")
            # Nonnegative ET and stress, RH in [0,100], LWI in [0,1].
            lower = backend.array([-xp.inf, 0., 0., 0., 0., 0.])
            upper = backend.array([xp.inf, 100., 1., xp.inf, xp.inf, xp.inf])
            violations = field - xp.clip(field, lower, upper)
            physics = xp.mean((violations / self.scales)**2)
            field_gradient = 2 * weights.physics * violations / (field.size * self.scales**2)
            differences = edges @ (field / self.scales)
            regularization = xp.mean(differences**2) if differences.size else xp.asarray(0.)
            if differences.size:
                field_gradient += (2 * weights.regularization / differences.size
                                   * (edges.T @ differences) / self.scales)
            if gradients:
                self.optimizer.zero_grad()
                if pair_loss.spatial_gradient is not None:
                    field_gradient += pair_loss.spatial_gradient.reshape(field.shape)
                derivative = (mapping.T @ field_gradient).reshape(raw.shape)
                if pair_loss.temporal_gradient is not None:
                    derivative += pair_loss.temporal_gradient
                self.model.backward(derivative, cache)
            total = pair_loss.loss + weights.physics * physics + weights.regularization * regularization
            report = dict(loss=float(backend.scalar(total)), temporal=pair_loss.temporal,
                          spatial=pair_loss.spatial, physics=float(backend.scalar(physics)),
                          regularization=float(backend.scalar(regularization)))
            return backend.to_host(source), backend.to_host(field), report

    def step(self, *batch):
        _, _, report = self.objective(*batch, gradients=True)
        return {**report, **self.optimizer.step()}

    def save(self, path):
        write_json(path, {
            "format": "jupiter-joint-v1",
            "architecture": dict(hidden=self.model.hidden, dropout=self.model.dropout,
                                 skips=self.model.skips, activations=self.model.activations),
            "feature_scales": self.backend.to_host(self.scales),
            "weights": vars(self.weights),
            "model": self.model.state_dict(self.backend),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.backend.rng.get_state() if not self.backend.is_gpu else None,
        })

    @classmethod
    def load(cls, path, backend):
        record = read_json(path)
        if record["format"] != "jupiter-joint-v1":
            raise ValueError("Expected a Jupiter joint training checkpoint")
        model = TemporalModel(backend, **record["architecture"])
        model.load_state_dict({key: backend.array(value) for key, value in record["model"].items()}, backend)
        trainer = cls(model, record["feature_scales"], weights=JointWeights(**record["weights"]))
        optimizer = record["optimizer"]
        optimizer["state"] = [{key: backend.array(value) for key, value in state.items()}
                              for state in optimizer["state"]]
        trainer.optimizer.load_state_dict(optimizer)
        if record["rng"] is not None and not backend.is_gpu:
            name, keys, position, cached, gaussian = record["rng"]
            backend.rng.set_state((name, keys, position, cached, gaussian))
        return trainer
