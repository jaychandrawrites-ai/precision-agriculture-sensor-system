"""Load an immutable checkpoint snapshot for repeat inference."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from Data.Normalization import Standardizer
from Temporal.Predictor import forecast
from Utils.Checkpoints import load_state
from Utils.Serialization import decode_json


@dataclass(frozen=True)
class Forecaster:
    model: object
    backend: object
    inputs: object
    targets: object
    checkpoint_sha256: str
    provenance: dict
    direct: object | None = None
    quantile: object | None = None
    blend: object | None = None
    members: tuple = ()

    @property
    def history_hours(self):
        if self.members:
            return max(model.history_hours for model in self.members)
        return self.direct.context_hours if self.direct is not None else 3

    @property
    def maximum_forecast_hours(self):
        if self.blend is not None:
            return min(len(self.blend.weights), *(member.maximum_forecast_hours for member in self.members))
        if self.quantile is not None:
            return min(72, self.quantile.horizon)
        if self.direct is not None:
            return min(72, self.direct.model.horizon)
        return 72

    def time_requirements(self):
        return {"input_interval_minutes": 60, "consecutive_feature_hours": self.history_hours,
                "checkpoint_maximum_forecast_hours": self.maximum_forecast_hours,
                "validated_horizon_hours": self.provenance.get("validated_horizon_hours"),
                "short_visit_requirement": "At least one eligible hourly field aggregate and compatible adjacent context"}
    def forecast(self, history, steps):
        point, _ = self.forecast_bundle(history, steps)
        return point

    def forecast_bundle(self, history, steps):
        """Return point predictions and optional [lead, feature, probability] quantiles."""
        cap = min(72, self.maximum_forecast_hours)
        if type(steps) is not int or not 1 <= steps <= cap:
            label = "Blend" if self.blend is not None else "Selected checkpoint"
            raise ValueError(f"{label} supports integer horizons from 1 through {cap} hours")
        if self.blend is not None:
            import numpy as np

            predictions = np.asarray([model.forecast(history[-model.history_hours:], steps)
                                      for model in self.members])
            return np.einsum("mlf,lfm->lf", predictions, self.blend.weights[:steps]), None
        if self.quantile is not None:
            from Temporal.Quantile import quantile_center

            trajectories = self.quantile.predict(history[None], steps)[0]
            return quantile_center(trajectories, self.quantile.levels), trajectories
        if self.direct is not None:
            return self.direct.predict(history[None])[0, :steps], None
        return forecast(self.model, history, self.inputs, self.targets, self.backend, steps), None

    def require_compatible_field(self, field_provenance):
        """Compare the method, history and configured physics shared by both domains."""
        if self.members:
            return all([model.require_compatible_field(field_provenance) for model in self.members])
        if "reference_et" not in self.provenance:
            return False
        if self.provenance["reference_et"]["history_hours"] != field_provenance["reference_et"]["history_hours"]:
            raise ValueError("Field ET history length differs from the training configuration")
        for component in ("soil", "crop"):
            if self.provenance["assumptions"][component] != field_provenance["assumptions"][component]:
                raise ValueError(f"Field {component} model differs from the training configuration")
        return True


def load_forecaster(path, device="cpu"):
    payload = Path(path).read_bytes()
    return _load_record(decode_json(payload), sha256(payload).hexdigest(), device)


def _load_record(record, digest, device, *, allow_blend=True):
    provenance = record.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Invalid training provenance")
    if record.get("format") == "jupiter-validation-blend-v1":
        import numpy as np
        from Temporal.ForecastUncertainty import ValidationBlend

        members = record["members"]
        names = [item["name"] for item in members]
        weights = np.asarray(record["weights"], dtype=float)
        if (not allow_blend or len(names) < 2 or len(names) != len(set(names))
                or weights.ndim != 3 or weights.shape[1:] != (6, len(names))
                or not 1 <= len(weights) <= 72 or not np.isfinite(weights).all()
                or (weights < 0).any() or not np.allclose(weights.sum(axis=-1), 1)):
            raise ValueError("Invalid validation blend")
        models = tuple(_load_record(item["state"], item["checkpoint_sha256"], device, allow_blend=False)
                       for item in members)
        first = models[0]
        return Forecaster(first.model, first.backend, first.inputs, first.targets, digest, provenance,
                          blend=ValidationBlend(weights, provenance["validation_origins"], 0), members=models)

    if record.get("format") in ("jupiter-direct-v1", "jupiter-direct-v2", "jupiter-direct-v3"):
        from Temporal.DirectCheckpoint import load_direct_state

        direct = load_direct_state(record, device)
        return Forecaster(direct.model, direct.backend, direct.inputs, direct.targets, digest, provenance, direct)
    if record.get("format") == "jupiter-quantile-v1":
        from Temporal.Quantile import QuantileForecaster

        quantile = QuantileForecaster.from_state(record, device)
        normalization = Standardizer.from_state({"mean": quantile.center, "scale": quantile.scale,
                                                  "epsilon": 1e-8})
        return Forecaster(quantile.network, quantile.backend, normalization, normalization,
                          digest, provenance, quantile=quantile)
    model, backend, inputs, targets = load_state(record, device)
    return Forecaster(model, backend, inputs, targets, digest, provenance)
