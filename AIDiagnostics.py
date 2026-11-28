"""Save held-out rollout predictions and render diagnostics from recorded metrics."""
from pathlib import Path

import numpy as np

from config import FEATURES
from Utils.Serialization import write_arrays, write_json


def publish(directory, report, observed, predicted, persistence, seasonal, parameters, timestamps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_arrays(directory / "evaluation.npz", observed=observed, predicted=predicted,
                 persistence=persistence, seasonal=seasonal, target_utc_ns=timestamps.astype("datetime64[ns]").astype("int64"))
    write_json(directory / "metrics.json", report)
    fit = report["fit_diagnostics"]

    def export(name, draw, rows=2, columns=3):
        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows), layout="constrained")
        try:
            draw(np.asarray(axes).reshape(-1))
            fig.savefig(directory / f"{name}.png", dpi=150)
        finally:
            plt.close(fig)

    def training(axes):
        for label in ("training", "validation"):
            axes[0].plot(np.arange(1, len(fit[label + "_loss"])+1), fit[label + "_loss"], label=label)
        axes[0].axvline(report["best_epoch"], color="black", linestyle=":", label="Selected epoch")
        axes[0].set(xlabel="Epoch", ylabel="Normalized loss")
        axes[0].legend()
        for axis, key, unit in zip(axes[1:], ("learning_rate", "epoch_seconds"), ("Learning rate", "Seconds")):
            if fit.get(key):
                axis.plot(np.arange(1, len(fit[key])+1), fit[key])
                axis.set(xlabel="Epoch", ylabel=unit)
            else:
                axis.set_axis_off()

    def scatter(axes):
        for i, (axis, feature) in enumerate(zip(axes, FEATURES)):
            actual, forecast = observed[:, 0, i], predicted[:, 0, i]
            axis.scatter(actual, forecast, s=3, alpha=.35)
            endpoints = [min(actual.min(), forecast.min()), max(actual.max(), forecast.max())]
            axis.plot(endpoints, endpoints, color="black", linewidth=.8)
            axis.set(title=feature, xlabel="Observed/derived target", ylabel="Prediction")

    def horizon(axes):
        for axis, feature in zip(axes, FEATURES):
            for key in ("test_rollout", "persistence", "seasonal_persistence"):
                axis.plot(np.arange(1, len(report[key])+1), [row[feature]["rmse"] for row in report[key]], label=key)
            axis.set(title=feature, xlabel="Forecast lead (hours)", ylabel="RMSE")
        axes[0].legend(fontsize=7)

    def residuals(axes):
        for i, (axis, feature) in enumerate(zip(axes, FEATURES)):
            axis.hist(observed[:, 0, i] - predicted[:, 0, i], bins=25)
            axis.set(title=feature, xlabel="Target minus prediction", ylabel="Count")

    def weights(axes):
        axis = axes[0]
        axis.boxplot([array.ravel() for array in parameters.values()])
        axis.set_xticks(range(1, len(parameters)+1), list(parameters), rotation=45, ha="right")
        axis.set(ylabel="Parameter value", title="Saved model parameters")

    export("training", training, 1, 3)
    export("actual-predicted", scatter)
    export("horizon", horizon)
    export("residuals", residuals)
    export("parameters", weights, 1, 1)
