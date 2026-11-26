"""Render report-derived physical scenario envelopes and sensitivity figures."""
from pathlib import Path
import re

import numpy as np

from config import FEATURES


LABELS = ("Air temperature (°C)", "Relative humidity (%)", "Leaf wetness index (0–1)",
          "Reference ET (mm/h)", "Actual ET (mm/h)", "Cotton heat stress (index)")


def render_scenario_figures(report, destination):
    """Write matching SVG/PNG figures without modifying the experiment report."""
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError("Figure export requires matplotlib") from exc
    if tuple(report["features"]) != FEATURES:
        raise ValueError("Scenario report feature order differs from the forecast contract")
    output = Path(destination)
    output.mkdir()
    leads = np.asarray(report["forecast_leads_hours"])
    names = []
    for name, entry in report["sites"].items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("Invalid site identifier for figure export")
        quantiles = np.asarray(entry["forecast"]["quantiles_by_lead_feature"], dtype=float)
        point = np.asarray(entry["nominal_forecast"], dtype=float)
        if quantiles.shape != (3, len(leads), len(FEATURES)) or point.shape != quantiles.shape[1:]:
            raise ValueError("Scenario figure data does not match lead and feature axes")

        figure = Figure(layout="constrained", figsize=(10, 8))
        FigureCanvasAgg(figure)
        figure.suptitle(f"Physical scenario envelopes — {name}\n"
                         f"{report['central_mass']:.0%} central scenario range, conditional on checkpoint",
                         fontsize=13)
        for i, axis in enumerate(figure.subplots(3, 2).flat):
            axis.fill_between(leads, quantiles[0, :, i], quantiles[2, :, i],
                              color="#007d7c", alpha=.18, label="Scenario range")
            axis.plot(leads, quantiles[1, :, i], color="#007d7c", label="Scenario median")
            axis.plot(leads, point[:, i], color="#34465c", linestyle="--", label="Nominal forecast")
            axis.set(xlabel="Hours from issue", ylabel=LABELS[i])
            axis.grid(color="#c0cad5", linewidth=.6, alpha=.45)
            if i == 0:
                axis.legend(frameon=False, fontsize=8)

        rankings = entry["forecast"]["rank_correlation_by_lead_feature"]
        ranks = np.asarray([rankings[key] for key in rankings], dtype=float)
        rank_figure = Figure(layout="constrained", figsize=(10, 7))
        FigureCanvasAgg(rank_figure)
        rank_figure.suptitle(f"Scenario sensitivity at +{leads[-1]} h — {name}\n"
                             "Spearman rank correlation, undefined entries are blank", fontsize=13)
        axis = rank_figure.subplots()
        colors = axis.imshow(ranks[:, -1, :], aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_xticks(np.arange(len(LABELS)), LABELS, rotation=28, ha="right")
        axis.set_yticks(np.arange(len(rankings)), list(rankings))
        for row, values in enumerate(ranks[:, -1, :]):
            for column, value in enumerate(values):
                if np.isfinite(value):
                    axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                              color="white" if abs(value) > .65 else "#263444", fontsize=8)
        rank_figure.colorbar(colors, ax=axis, label="Rank correlation", shrink=.85)

        for label, plot in (("forecast-envelope", figure), ("rank-sensitivity", rank_figure)):
            for extension in ("png", "svg"):
                path = output / f"{name}-{label}.{extension}"
                plot.savefig(path, dpi=160)
                names.append(path.name)
    return names
