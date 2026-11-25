"""Versioned JSON temporal checkpoints with atomic replacement."""
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from assumptions import Assumptions
from config import FEATURES
from Data.Normalization import Standardizer
from Framework.Backend import create_backend
from Temporal.Model import TemporalModel
from Temporal.Rollout import RolloutOptions


def save(path, model, dataset, backend, result, settings):
    path = Path(path)
    state = {
        "version": 1,
        "features": FEATURES,
        "interval_hours": 1,
        "architecture": {"hidden": model.hidden, "dropout": model.dropout,
                         "skips": model.skips},
        "parameters": model.state_dict(backend),
        "input_scaler": dataset.input_scaler.state_dict(),
        "target_scaler": dataset.target_scaler.state_dict(),
        "assumptions": Assumptions().metadata(),
        "training_config": asdict(settings),
        "training_result": asdict(result),
    }
    options = getattr(model, "rollout_options", None)
    if options is not None:
        state["version"] = 2
        state["rollout_options"] = asdict(options)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(state, file, default=lambda array: array.tolist(), allow_nan=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path, device="cpu"):
    with open(path, encoding="utf-8") as file:
        state = json.load(file)
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
    return (model, backend, Standardizer.from_state(state["input_scaler"]),
            Standardizer.from_state(state["target_scaler"]))
