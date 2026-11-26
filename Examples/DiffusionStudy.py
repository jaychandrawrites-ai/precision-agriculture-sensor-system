"""Small held-out comparison on an explicitly synthetic local-metric field."""
import argparse

import numpy as np

from Spatial.DiffusionMap import diffusion_map
from Spatial.Interpolation import idw
from Spatial.Kriging import CovarianceModel, KrigingModel
from Spatial.SpatialGrid import SpatialGrid
from Utils.Serialization import write_json


def compare():
    axis = np.linspace(-60, 60, 25)
    grid = SpatialGrid(axis, axis)
    queries = grid.points()
    inside = np.linalg.norm(queries, axis=1) <= 60
    fit = np.array([[-35, 0], [35, 0], [0, 35], [0, -35], [0, 0]])
    held = np.array([[-25, 25], [25, 25], [-25, -25], [25, -25], [45, 15]])
    truth = lambda points: 24 + .06 * points[:, 0] + np.sin(points[:, 1] / 40)
    values = truth(fit)
    field, diagnostic = diffusion_map(grid, fit, values, inside.reshape(grid.shape), iterations=1000)
    indices = np.argmin(np.linalg.norm(queries[:, None] - held[None, :], axis=2), axis=0)
    predictions = {
        "diffusion": field.ravel()[indices],
        "idw": idw(fit, values, held),
        "ordinary_kriging": KrigingModel(fit, values, CovarianceModel(40.)).predict(held).mean,
    }
    actual = truth(held)
    return {"synthetic": True, "feature": "air_temperature_c", "fit_sites": len(fit),
            "held_out_sites": len(held), "diffusion_converged": diagnostic["converged"],
            "metrics": {name: {"rmse": np.sqrt(np.mean((pred - actual)**2)),
                               "mae": np.mean(abs(pred - actual))}
                        for name, pred in predictions.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    args = parser.parse_args()
    write_json(args.report, compare())
