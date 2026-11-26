"""Versioned JSON temporal checkpoints with atomic replacement."""
from dataclasses import asdict

import numpy as np

from assumptions import Assumptions
from config import FEATURES
from Data.Normalization import Standardizer
from Framework.Backend import create_backend
from Temporal.Model import TemporalModel
from Temporal.Rollout import RolloutOptions
from Utils.Serialization import read_json, write_json


def save(path, model, dataset, backend, result, settings, *, provenance=None):
    state = {
        "version": 1,
        "features": FEATURES,
        "interval_hours": 1,
        "architecture": {"hidden": model.hidden, "dropout": model.dropout,
                         "skips": model.skips},
        "parameters": model.state_dict(backend),
        "input_scaler": dataset.input_scaler.state_dict(),
        "target_scaler": dataset.target_scaler.state_dict(),
        "assumptions": (provenance or {}).get("assumptions", Assumptions().metadata()),
        "training_config": asdict(settings),
        "training_result": asdict(result),
    }
    if any(name != "relu" for name in model.activations):
        state["architecture"]["activations"] = model.activations
    options = getattr(model, "rollout_options", None)
    if options is not None:
        state["version"] = 2
        state["rollout_options"] = asdict(options)
    if provenance is not None:
        state["provenance"] = provenance
    write_json(path, state, indent=None)


def load(path, device="cpu"):
    return load_state(read_json(path), device)


def load_state(state, device="cpu"):
    if (state["version"] not in (1, 2) or tuple(state["features"]) != FEATURES
            or state["interval_hours"] != 1):
        raise ValueError("Unsupported checkpoint format")
    settings = state["training_config"]
    backend = create_backend(device=device, dtype=settings["dtype"], seed=settings["seed"])
    model = TemporalModel(backend, **state["architecture"])
    if state["version"] == 2:
        model.rollout_options = RolloutOptions(**state["rollout_options"])
    parameters = {key: np.asarray(value, dtype=settings["dtype"])
                  for key, value in state["parameters"].items()}
    for value in parameters.values():
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint weights")
    model.load_state_dict(parameters, backend)
    input_scaler = Standardizer.from_state(state["input_scaler"])
    target_scaler = Standardizer.from_state(state["target_scaler"])
    if input_scaler.mean.shape != (18,) or target_scaler.mean.shape != (len(FEATURES),):
        raise ValueError("Checkpoint normalization does not match the feature contract")
    return model, backend, input_scaler, target_scaler
