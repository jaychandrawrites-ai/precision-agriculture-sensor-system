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
    if fit.get("layer_diagnostics"):
        def internal(axes):
            samples = fit["layer_diagnostics"]
            for name in samples[0]["layers"]:
                epochs = [item["epoch"] for item in samples]
                axes[0].plot(epochs, [item["layers"][name]["gradient"]["l2"] for item in samples], label=name)
                axes[1].plot(epochs, [item["layers"][name]["update_to_weight_norm"] for item in samples])
            axes[0].set(xlabel="Epoch", ylabel="Gradient L2 (before clipping)")
            axes[0].legend(fontsize=7)
            axes[1].set(xlabel="Epoch", ylabel="Update / parameter L2")
            for name, summary in samples[-1]["inference_activations"].items():
                axes[2].hist(summary["sample"], bins=20, histtype="step", label=name)
            axes[2].set(xlabel="Activation", ylabel="Sample count")
        export("internal", internal, 1, 3)


def publish_evolution(directory, generations):
    """Render per-generation measurements from a recorded validation search."""
    if not generations:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    write_json(directory / "evolution.json", generations)
    steps = [item["generation"] for item in generations]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    fitness = [item.get("fitness") for item in generations]
    try:
        if all(fitness):
            axes[0, 0].boxplot(fitness, positions=steps)
            for summarize, label in ((np.mean, "Mean"), (np.median, "Median")):
                axes[0, 0].plot(steps, [summarize(row) for row in fitness], label=label)
        axes[0, 0].plot(steps, [item["best_selection_score"] for item in generations], label="Best so far")
        axes[0, 0].legend()
        axes[0, 0].set_ylabel("Validation fitness (lower is better)")
        for axis, key, label in (
            (axes[0, 1], "parameter_counts", "Parameters"),
            (axes[0, 2], "candidate_training_seconds", "Candidate training seconds"),
            (axes[1, 2], "mean_population_bit_probabilities", "Probability of bit 1"),
        ):
            samples = [(step, [value for value in item.get(key, ()) if value is not None])
                       for step, item in zip(steps, generations)]
            measured = [(step, values) for step, values in samples if values]
            if measured:
                positions, rows = zip(*measured)
                axis.boxplot(rows, positions=positions)
            axis.set_ylabel(label)
        axes[1, 0].plot(steps, [item["diversity"] for item in generations], label="Hamming diversity")
        for key, label in (("breeding_mutation_probability", "Mutation probability"),
                           ("mean_individual_bit_entropy", "Amplitude entropy")):
            points = [(item["generation"], item[key]) for item in generations if key in item]
            if points:
                axes[1, 0].plot(*zip(*points), marker=".", label=label)
        axes[1, 0].legend()
        axes[1, 1].plot(steps, [item.get("retained_genotypes", np.nan) for item in generations])
        axes[1, 1].set_ylabel("Distinct genotypes retained from prior generation")
        for axis in axes.flat:
            axis.set_xlabel("Generation")
        fig.savefig(directory / "evolution.png", dpi=150)
    finally:
        plt.close(fig)
