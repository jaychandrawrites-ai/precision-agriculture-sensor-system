"""Fit a portable saved-checkpoint blend on an explicit validation period."""
import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from Data.Preprocessing import read_hourly
from Data.RolloutWindows import evaluation_windows
from Temporal.ForecastUncertainty import ValidationBlend
from Utils.ModelLoader import load_forecaster
from Utils.Serialization import decode_json, write_json


def fit(source, members, output, *, start, end, steps, names=None):
    if not 1 <= steps <= 72:
        raise ValueError("Forecast horizon must be 1–72 hours")
    values, times = read_hourly(source)
    lo, hi = np.searchsorted(times, np.asarray([start, end], dtype="datetime64[ns]"))
    if not 3 <= lo < hi or len(members) < 2:
        raise ValueError("Provide an ordered validation interval and at least two checkpoints")
    checkpoints = [load_forecaster(path) for path in members]
    hashes = [item.checkpoint_sha256 for item in checkpoints]
    names = [path.stem for path in members] if names is None else names
    if len(set(hashes)) != len(hashes) or len(names) != len(members) or len(set(names)) != len(members):
        raise ValueError("Member names and checkpoint contents must be distinct")
    context = max(item.history_hours for item in checkpoints)
    windows = evaluation_windows(values, times, steps, np.arange(lo, hi), context=context)
    if not len(windows.histories):
        raise ValueError("No complete validation windows")
    preds = np.array([[model.forecast(history[-model.history_hours:], steps)
                       for history in windows.histories] for model in checkpoints])
    fitted = ValidationBlend.fit(preds, windows.targets)
    records = []
    for path, digest, name in zip(members, hashes, names):
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError("Checkpoint changed during validation")
        records.append({"name": name, "checkpoint_sha256": digest, "state": decode_json(raw)})
    write_json(output, {"format": "jupiter-validation-blend-v1", "members": records,
                        "weights": fitted.weights,
                        "provenance": {"validation_start_utc": start, "validation_end_exclusive_utc": end,
                                       "validation_origins": fitted.fit_origins,
                                       "source_sha256": sha256(Path(source).read_bytes()).hexdigest()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--member", type=Path, action="append", required=True)
    parser.add_argument("--name", action="append")
    parser.add_argument("--start", required=True, help="Validation start in UTC")
    parser.add_argument("--end", required=True, help="Exclusive validation end in UTC")
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    fit(args.source, args.member, args.output, start=args.start, end=args.end, steps=args.steps, names=args.name)
