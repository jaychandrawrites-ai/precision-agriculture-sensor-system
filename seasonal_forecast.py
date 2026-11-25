"""Run a saved seasonal ensemble with recorded hourly history only."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURES
from Data.Preprocessing import read_hourly
from Temporal.SeasonalCheckpoint import forecast_saved
from Utils.Serialization import atomic_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    values, times = read_hourly(args.source)
    result = forecast_saved(args.checkpoint, values, times)
    first = pd.Timestamp(times[-1] + np.timedelta64(1, "h"), tz="UTC")
    rows = pd.DataFrame(result["point"], columns=FEATURES,
                        index=pd.date_range(first, periods=len(result["point"]), freq="h", name="timestamp"))
    rows["seasonal_observation_missing"] = result["used_current_state"]
    for label in ("lower", "upper", "trajectory_lower", "trajectory_upper"):
        for index, feature in enumerate(FEATURES):
            rows[f"{feature}_{label}"] = result[label][:, index]
    with atomic_output(args.output) as handle:
        rows.to_csv(handle)


if __name__ == "__main__":
    main()
