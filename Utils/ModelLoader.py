"""Load an immutable checkpoint snapshot for repeat inference."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

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

    @property
    def history_hours(self):
        return self.direct.context_hours if self.direct is not None else 3

    def forecast(self, history, steps):
        if self.direct is not None:
            if type(steps) is not int or not 1 <= steps <= self.direct.model.horizon:
                raise ValueError(f"Direct checkpoint supports 1–{self.direct.model.horizon} forecast hours")
            return self.direct.predict(history[None])[0, :steps]
        return forecast(self.model, history, self.inputs, self.targets, self.backend, steps)

    def require_compatible_field(self, field_provenance):
        """Compare the method, history and configured physics shared by both domains."""
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
    record = decode_json(payload)
    provenance = record.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Invalid training provenance")
    digest = sha256(payload).hexdigest()
    if record.get("format") in ("jupiter-direct-v1", "jupiter-direct-v2", "jupiter-direct-v3"):
        from Temporal.DirectCheckpoint import load_direct_state

        direct = load_direct_state(record, device)
        return Forecaster(direct.model, direct.backend, direct.inputs, direct.targets, digest, provenance, direct)
    model, backend, inputs, targets = load_state(record, device)
    return Forecaster(model, backend, inputs, targets, digest, provenance)
