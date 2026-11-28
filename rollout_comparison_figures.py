"""Compare two saved rollout evaluations on exactly the same origins."""
import argparse
from pathlib import Path

import numpy as np

from config import FEATURES
from Utils.Serialization import read_json, write_json


def compare(before, after, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [Path(before), Path(after)]
    metrics = [read_json(run / "metrics.json") for run in runs]
    with np.load(runs[0] / "evaluation.npz") as old, np.load(runs[1] / "evaluation.npz") as new:
        for key in ("observed", "target_utc_ns"):
            if not np.array_equal(old[key], new[key]):
                raise ValueError("Comparison requires identical observations and target times")
        target = old["observed"]
        predictions = (old["predicted"], new["predicted"])
    lead = np.arange(1, target.shape[1] + 1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    try:
        axes[0, 0].plot(lead, target[0, :, 0], label="Observed")
        axes[1, 0].plot(lead[1:], np.diff(target[0, :, 0]), label="Observed")
        for label, predicted, report in zip(("Before", "After"), predictions, metrics):
            error = predicted - target
            bias = error.mean(axis=0)
            rmse = np.sqrt((error**2).mean(axis=0))
            projection = np.array([row["projected_by_feature"] for row in report["rollout_diagnostics"]]) / len(target)
            changes = np.diff(predicted[:, :, 0], axis=1)
            summary[label.lower()] = {
                "bias": bias, "rmse": rmse, "projection_fraction": projection,
                "first_origin_temperature": predicted[0, :, 0],
                "temperature_change_quantiles": np.quantile(changes, [.1, .5, .9], axis=0),
            }
            axes[0, 0].plot(lead, predicted[0, :, 0], label=label)
            axes[1, 0].plot(lead[1:], changes[0], label=label)
            axes[0, 1].plot(lead, rmse[:, 0], label=label)
            axes[0, 2].plot(lead, bias[:, 0], label=label)
            axes[1, 1].plot(lead, projection[:, 0], label=label)
            axes[1, 2].plot(lead, projection.mean(axis=1), label=label)
        for axis, title in zip(axes.flat, ("First origin temperature (°C)", "Temperature RMSE (°C)",
                "Temperature bias (°C)", "First origin hourly temperature change (°C)",
                "Temperature projection fraction", "Mean projection fraction across features")):
            axis.set(title=title, xlabel="Forecast lead (hours)")
            axis.legend()
        fig.savefig(output / "stability.png", dpi=150)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    try:
        for index, (axis, feature) in enumerate(zip(axes.flat, FEATURES)):
            for label, record in summary.items():
                axis.plot(lead, record["rmse"][:, index], label=label.title())
            axis.set(title=feature, xlabel="Forecast lead (hours)", ylabel="RMSE")
            axis.legend()
        fig.savefig(output / "feature-rmse.png", dpi=150)
    finally:
        plt.close(fig)
    write_json(output / "stability.json", {"features": FEATURES, "origins": len(target),
               "lead_hours": lead, "first_origin_observed_temperature": target[0, :, 0], **summary})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", help="Original run diagnostics directory")
    parser.add_argument("after", help="Revised run diagnostics directory")
    parser.add_argument("output")
    args = parser.parse_args()
    compare(args.before, args.after, args.output)
