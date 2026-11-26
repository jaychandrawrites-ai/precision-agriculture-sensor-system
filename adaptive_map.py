"""Fit an experimental adaptive kriging grid from explicit local-metric observations."""
import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd

from Spatial.AdaptiveGrid import Refinement, refine
from Spatial.Kriging import CovarianceModel, KrigingModel
from Utils.Serialization import read_json, write_json


def adaptive_map(source, configuration):
    payload = Path(source).read_bytes()
    frame = pd.read_csv(BytesIO(payload))
    points = frame[["east_m", "north_m"]].to_numpy(dtype=float)
    feature = configuration["feature"]
    measurement_variance = configuration.get("measurement_variance_column")
    result = adaptive_kriging(points, frame[feature], configuration,
                              variance=frame[measurement_variance] if measurement_variance else None)
    return {"source_sha256": sha256(payload).hexdigest(), "feature": feature, **result}


def adaptive_kriging(points, values, configuration, *, variance=None):
    model = KrigingModel(points, values, CovarianceModel(**configuration["covariance"]),
                         trend=configuration.get("trend", "constant"),
                         measurement_variance=variance)

    def predictor(query):
        output = model.predict(query)
        return output.mean, output.latent_variance

    result = refine(configuration["bounds_m"], predictor,
                    Refinement(**configuration.get("refinement", {})),
                    observed_xy=points, polygon=configuration.get("field_polygon_m"))
    return {"observations": len(points), "conditioning_jitter": model.jitter,
            "uncertainty": "conditional kriging variance", **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, adaptive_map(args.observations, read_json(args.configuration)))


if __name__ == "__main__":
    main()
