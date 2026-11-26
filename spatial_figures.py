"""Render a survey predictive mean and interval width in local metric coordinates."""
import argparse
from pathlib import Path

import numpy as np

from Utils.Serialization import read_json


def render(manifest_path, output, *, lead, feature):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    index = manifest["features"].index(feature)
    if not 1 <= lead <= len(manifest["valid_at"]):
        raise ValueError("Requested lead is outside the export")
    with np.load(manifest_path.with_suffix(".npz"), allow_pickle=False) as arrays:
        mean = arrays["predictive_mean"][lead - 1, :, index]
        width = arrays["predictive_width"][lead - 1, :, index]
        centers = arrays["predictive_centers_m"]
        active = arrays["predictive_inside"]
        sites = arrays["predictive_site_xy_m"]
        method = manifest["predictive"]["method"]
        key = "variational_bounds_m" if method in {"variational", "adaptive_variational"} else "adaptive_bounds_m"
        if key in arrays and len(arrays[key]) == len(centers):
            boxes = arrays[key]
        else:
            dx, dy = (np.diff(arrays[name])[0] for name in ("east_m", "north_m"))
            boxes = np.column_stack((centers - [dx / 2, dy / 2], centers + [dx / 2, dy / 2]))
    fig, panels = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
    units = ("°C", "%", "index", "mm/interval", "mm/interval", "index")[index]
    try:
        for ax, layer, title, palette in zip(panels, (mean, width), ("Predictive mean", "Interval width"),
                                              ("viridis", "magma")):
            patches = [Rectangle((x0, y0), x1-x0, y1-y0) for x0, y0, x1, y1 in boxes[active]]
            display = PatchCollection(patches, cmap=palette, edgecolor="none")
            display.set_array(layer[active])
            ax.add_collection(display)
            ax.scatter(*sites.T, s=14, facecolor="white", edgecolor="black", label="Forecast site")
            ax.autoscale_view()
            ax.set(aspect="equal", xlabel="East (m)", ylabel="North (m)", title=title)
            fig.colorbar(display, ax=ax, label=units, shrink=.8)
        panels[0].legend(loc="best")
        fig.suptitle(f"{feature.replace('_', ' ')} · {lead} h\nValid {manifest['valid_at'][lead - 1]}")
        for suffix in (".svg", ".png"):
            fig.savefig(Path(output).with_suffix(suffix), dpi=220)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--feature", default="air_temperature_c")
    args = parser.parse_args()
    render(args.manifest, args.output, lead=args.lead, feature=args.feature)
