"""CPU epoch-boundary recovery records for the stable rollout trainer."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from Utils.Serialization import read_json, write_json


def recipe(model, settings, options, training, validation, inputs, targets, **training_options):
    metadata = {"format": "jupiter-rollout-epoch-v1", "settings": asdict(settings),
                "architecture": {"hidden": list(model.hidden), "skips": list(model.skips),
                                 "dropout": model.dropout},
                "rollout": asdict(options), "training": training_options,
                "numpy": np.__version__}
    metadata = json.loads(json.dumps(metadata))
    digest = sha256()
    for array in (training.histories, training.targets, validation.histories, validation.targets,
                  inputs.mean, inputs.scale, targets.mean, targets.scale):
        array = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return {**metadata, "data_sha256": digest.hexdigest()}


def serialize_runtime(model, backend, optimizer, schedule, loader):
    return {"weights": model.state_dict(backend), "optimizer": optimizer.state_dict(),
            "schedule": schedule.state_dict(), "shuffle_rng": loader.rng.bit_generator.state,
            "backend_rng": backend.rng.get_state()}


def decode_runtime(record, model, backend):
    weights = {key: np.asarray(value, dtype=backend.dtype) for key, value in record["weights"].items()}
    optimizer = dict(record["optimizer"])
    states = []
    for parameter, state in zip(model.parameters(), optimizer["state"], strict=True):
        dtype = "float64" if parameter.value.dtype.itemsize == 8 else "float32"
        states.append({key: np.asarray(array, dtype=dtype) for key, array in state.items()})
    optimizer["state"] = states
    rng = record["backend_rng"]
    rng = (rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:])
    for array in [*weights.values(), *(a for state in states for a in state.values())]:
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite recovery state")
    return weights, optimizer, record["schedule"], record["shuffle_rng"], rng


def read_recovery(path, expected):
    if not Path(path).exists():
        return None
    record = read_json(path)
    if record["recipe"] != expected:
        raise ValueError("Recovery data or settings changed; use the original run or a new state path")
    if (record["epoch"] != len(record["history"]) or not 1 <= record["best_epoch"] <= record["epoch"]
            or record["epoch"] > expected["settings"]["epochs"] or
            not all(np.isfinite(record["history"]))):
        raise ValueError("Recovery epoch metadata is invalid")
    return record


def write_recovery(path, record):
    write_json(path, record, indent=None)
