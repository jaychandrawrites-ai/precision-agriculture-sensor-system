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

    def forecast(self, history, steps):
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
    model, backend, inputs, targets = load_state(record, device)
    provenance = record.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Invalid training provenance")
    return Forecaster(model, backend, inputs, targets, sha256(payload).hexdigest(), provenance)
