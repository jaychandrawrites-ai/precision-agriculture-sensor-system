"""Run an explicitly synthetic offline temporal smoke demonstration."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from config import FEATURES, TrainingConfig
from forecast import forecast_csv
from train import train


def synthetic_history(hours=240):
    phase = np.arange(hours) * 2 * np.pi / 24
    sun = np.maximum(0, np.sin(phase))
    values = np.column_stack((26 + 8 * np.sin(phase), 55 - 25 * np.sin(phase),
                              0.4 - 0.3 * np.sin(phase), 0.5 * sun,
                              0.3 * sun, np.maximum(0, 8 * np.sin(phase) - 4)))
    frame = pd.DataFrame(values, columns=FEATURES)
    frame.insert(0, "timestamp", pd.date_range("2020-01-01", periods=hours, freq="h", tz="UTC"))
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo"])
    parser.add_argument("--legacy", action="store_true", help="Reproduce the original one-step model")
    args = parser.parse_args()
    print("Synthetic fixture for software verification")
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        source = directory / "synthetic.csv"
        checkpoint = directory / "temporal.json"
        synthetic_history().to_csv(source, index=False)
        settings = TrainingConfig(epochs=30, batch_size=32, dropout=0)
        print(json.dumps(train(source, checkpoint, settings, stable_rollout=not args.legacy), indent=2, allow_nan=False))
        print(forecast_csv(source, checkpoint, settings.rollout_steps).to_csv(index=False), end="")


if __name__ == "__main__":
    main()
