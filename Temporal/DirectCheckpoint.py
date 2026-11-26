"""Explicit portable checkpoint format for the standalone experimental model."""
import numpy as np

from config import FEATURES
from Data.Normalization import Standardizer
from Framework.Backend import create_backend
from Temporal.Direct import DirectForecast, make_direct_model
from Utils.Serialization import read_json, write_json


def save_direct(path, forecast, width, *, provenance=None):
    record = {
        "format": "jupiter-direct-v1", "features": FEATURES,
        "horizon": forecast.model.horizon, "width": width,
        "dtype": str(forecast.backend.dtype),
        "inputs": forecast.inputs.state_dict(), "targets": forecast.targets.state_dict(),
        "parameters": forecast.model.state_dict(forecast.backend),
        "provenance": provenance or {},
    }
    if forecast.model.architecture != "gated":
        record["format"] = "jupiter-direct-v2"
        record["architecture"] = forecast.model.architecture
    write_json(path, record)


def load_direct(path, device="cpu"):
    return load_direct_state(read_json(path), device)


def load_direct_state(record, device="cpu"):
    version = record.get("format")
    if version not in ("jupiter-direct-v1", "jupiter-direct-v2") or tuple(record.get("features", ())) != FEATURES:
        raise ValueError("Unsupported experimental forecast checkpoint")
    backend = create_backend(device=device, dtype=record["dtype"])
    architecture = "gated" if version == "jupiter-direct-v1" else record["architecture"]
    model = make_direct_model(backend, record["horizon"], record["width"], architecture)
    inputs = Standardizer.from_state(record["inputs"])
    targets = Standardizer.from_state(record["targets"])
    if inputs.mean.shape != (18,) or targets.mean.shape != (6,):
        raise ValueError("Invalid experimental checkpoint feature transforms")
    parameters = {key: np.asarray(array, dtype=record["dtype"])
                  for key, array in record["parameters"].items()}
    if any(not np.isfinite(array).all() for array in parameters.values()):
        raise ValueError("Nonfinite experimental checkpoint weights")
    model.load_state_dict(parameters, backend)
    return DirectForecast(model, backend, inputs, targets)
