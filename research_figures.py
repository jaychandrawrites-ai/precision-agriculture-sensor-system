"""Render figures from saved direct-training reports and forecast artifacts."""
import argparse
from pathlib import Path

from Dashboard.Reports import forecast_series
from Utils.Serialization import read_json


def render(source, output, kind, feature="air_temperature_c"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    try:
        if kind == "training":
            report = read_json(source)
            fit = report.get("fit_diagnostics", report)
            for key, label in (("training_loss", "Training"), ("validation_loss", "Validation")):
                loss = fit[key]
                axis.plot(range(1, len(loss) + 1), loss, label=label)
            axis.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
        elif kind == "forecast":
            report = forecast_series(source)
            draws = report.get("quantiles", {}).get(feature, [])
            hours = range(1, len(report["timestamps"]) + 1)
            axis.plot(hours, report["series"][feature], label="Point forecast")
            if len(draws) >= 2:
                axis.fill_between(hours, draws[0]["value"], draws[-1]["value"], alpha=.2,
                                  label=f"Model quantiles {draws[0]['level']:g}–{draws[-1]['level']:g}")
            axis.set(xlabel="Lead (hours)", ylabel=feature.replace("_", " "), title="Saved forecast")
        elif kind == "evolution-forecast":
            report = read_json(source)
            curves = {name: result["test_rollout"] for name, result in report.items()
                      if isinstance(result, dict) and result.get("test_rollout")}
            curves.update({name: rows for name, rows in report.get("baselines", {}).items()
                           if name in ("persistence", "seasonal_persistence")})
            for name, rows in curves.items():
                axis.plot(range(1, len(rows) + 1), [row[feature]["rmse"] for row in rows],
                          label=name.replace("_", " "))
            if not curves:
                raise ValueError("No matched rollout metrics in the experiment report")
            axis.set(xlabel="Lead (hours)", ylabel="RMSE in physical feature units",
                     title=feature.replace("_", " "))
        else:
            report = read_json(source)
            for method, result in report.items():
                if not isinstance(result, dict):
                    continue
                front = result.get("validation_pareto_front", [])
                points = [(row["parameter_count"], row["validation_loss"]) for row in front
                          if "parameter_count" in row and "validation_loss" in row]
                if points:
                    axis.scatter(*zip(*points), label=method.replace("_", " "))
            if not axis.collections:
                raise ValueError("No sequence Pareto points in the report")
            axis.set(xlabel="Model parameters", ylabel="Validation loss", title="Validation Pareto fronts")
        axis.legend()
        for suffix in (".svg", ".png"):
            figure.savefig(Path(output).with_suffix(suffix), dpi=200)
    finally:
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("training", "forecast", "pareto", "evolution-forecast"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--feature", default="air_temperature_c")
    args = parser.parse_args()
    render(args.source, args.output, args.kind, args.feature)
