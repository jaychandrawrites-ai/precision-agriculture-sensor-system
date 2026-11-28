"""Render and export measured spectral trajectories and modal contributions."""
from pathlib import Path

import numpy as np

from Utils.Serialization import write_arrays


def publish_spectral(directory, model, histories, origins, targets, predictions, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    coefficients = model.coefficients(histories, origins)
    lead = np.arange(1, 73)
    basis = model.modes.anchored_basis(lead)
    contributions = basis * coefficients[0, :, 0] * model.targets.scale[0]
    write_arrays(directory / "spectral.npz", coefficients=coefficients, lead=lead,
                 basis=basis, first_temperature_components=contributions,
                 event_counts=np.asarray(history["event_counts"]))
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    try:
        axes[0, 0].plot(lead, targets[0, :, 0], label="Observed")
        axes[0, 0].plot(lead, predictions[0, :, 0], label="Forecast")
        axes[0, 0].legend()
        axes[0, 0].set(title="First-origin temperature", xlabel="Lead (hours)", ylabel="°C")
        axes[0, 1].plot(lead, contributions[:, 0], label="Constant and anchor")
        labels = [str(p) + " h" for p in model.modes.periods]
        if model.modes.damped:
            labels.append("18 h transient")
        for index, label in enumerate(labels):
            axes[0, 1].plot(lead, contributions[:, 1 + 2*index:3 + 2*index].sum(axis=1), label=label)
        axes[0, 1].set(title="Modal contributions", xlabel="Lead (hours)", ylabel="°C")
        axes[0, 1].legend(fontsize=7)
        axes[0, 2].hist(coefficients[:, 1::2, 0].ravel(), bins=48, alpha=.65, label="Real")
        axes[0, 2].hist(coefficients[:, 2::2, 0].ravel(), bins=48, alpha=.65, label="Imaginary")
        axes[0, 2].legend()
        axes[0, 2].set(title="Temperature coefficient distribution")
        rms = np.sqrt(np.mean(coefficients[:, 1::2, 0]**2 + coefficients[:, 2::2, 0]**2, axis=0))
        axes[1, 0].bar(labels, rms)
        axes[1, 0].set(title="Temperature mode amplitudes", ylabel="Normalized RMS")
        axes[1, 0].tick_params(axis="x", rotation=30)
        epochs = np.arange(1, len(history["event_counts"]) + 1)
        axes[1, 1].bar(epochs, history["event_counts"])
        axes[1, 1].set(title="Recorded sparse training events", xlabel="Epoch")
        axes[1, 2].plot(epochs, history["gradient_norm"])
        axes[1, 2].set(title="Peak pre-clipping gradient norm", xlabel="Epoch")
        figure.savefig(directory / "spectral.png", dpi=150)
    finally:
        plt.close(figure)
