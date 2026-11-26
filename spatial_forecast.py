"""Map aligned multi-site forecast draws and retain their joint member structure."""
import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURES
from Spatial.SpatialGrid import SpatialGrid, local_coordinates
from Utils.Serialization import read_json, write_arrays, write_json


def _weights(points, queries, power):
    distance = np.linalg.norm(queries[:, None] - points[None], axis=2)
    exact = distance == 0
    nearest = distance.min(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(nearest == 0, exact, (nearest / distance)**power)
    return weights / weights.sum(axis=1, keepdims=True)


def ensemble_maps(grid, points, members, probability=.9, power=2., batch_size=512):
    """Apply a fixed linear IDW map to each coherent [draw, site, feature] array."""
    points, members = np.asarray(points, dtype=float), np.asarray(members, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or np.unique(points, axis=0).shape != points.shape
            or members.ndim != 3 or members.shape[0] < 2 or members.shape[1] != len(points)
            or not members.shape[-1] or not np.isfinite(points).all() or not np.isfinite(members).all()
            or not 0 < probability < 1 or not np.isfinite(power) or power <= 0
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError("Need distinct sites, at least two coherent draws, and finite features")
    queries = grid.points()
    arrays = {name: [] for name in ("mean", "spread", "lower", "upper")}
    tail = (1 - probability) / 2
    for start in range(0, len(queries), batch_size):
        weights = _weights(points, queries[start:start + batch_size], power)
        draws = weights @ members
        lower, upper = np.quantile(draws, [tail, 1-tail], axis=0)
        for label, value in (("mean", draws.mean(axis=0)), ("spread", draws.std(axis=0)),
                             ("lower", lower), ("upper", upper)):
            arrays[label].append(value)
    return {name: np.concatenate(values).reshape((*grid.shape, members.shape[-1]))
            for name, values in arrays.items()}


def _utc(column):
    if not column.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", na=False).all():
        raise ValueError("Forecast origin and validity timestamps require explicit offsets")
    return pd.to_datetime(column, format="ISO8601", utc=True)


def map_draw_file(source, config, output):
    payload = Path(source).read_bytes()
    frame = pd.read_csv(BytesIO(payload), dtype={"site_id": str, "draw_id": str})
    common = {"site_id", "draw_id", "latitude_deg", "longitude_deg",
              "forecast_origin", "valid_at", "value_origin"}
    features = config["features"]
    if (not isinstance(features, list) or not features or not all(x in FEATURES for x in features)
            or len(set(features)) != len(features) or not common | set(features) <= set(frame.columns)
            or frame.empty or not frame["value_origin"].eq("model_forecast").all()
            or frame[["site_id", "draw_id"]].isna().any().any()):
        raise ValueError("Draw input requires model forecast rows, matched IDs, and six-contract features")
    origins, times = _utc(frame["forecast_origin"]), _utc(frame["valid_at"])
    if origins.nunique() != 1 or times.nunique() != 1 or times.iloc[0] <= origins.iloc[0]:
        raise ValueError("Use one common forecast origin and one later valid time")
    chart = frame.set_index(["draw_id", "site_id"]).sort_index()
    if not chart.index.is_unique:
        raise ValueError("Each draw and site can appear only once")
    draw_ids, site_ids = (sorted(chart.index.get_level_values(level).unique()) for level in (0, 1))
    expected = pd.MultiIndex.from_product((draw_ids, site_ids), names=chart.index.names)
    if not chart.index.equals(expected):
        raise ValueError("All draws must contain the same sites; missing sites cannot be imputed")
    coordinate_cols = ["latitude_deg", "longitude_deg"]
    shape = (len(draw_ids), len(site_ids))
    coords = chart[coordinate_cols].to_numpy(dtype=float).reshape((*shape, 2))
    if not np.equal(coords, coords[:1]).all():
        raise ValueError("Site coordinates must be fixed across draws")
    points = local_coordinates(coords[0], config["origin_lat_lon"])
    members = chart[features].to_numpy(dtype=float).reshape((*shape, len(features)))
    grid = SpatialGrid(config["east_m"], config["north_m"])
    probability = config.get("central_mass", .9)
    arrays = ensemble_maps(grid, points, members, probability=probability,
                           power=config.get("idw_power", 2.))
    output = Path(output)
    output.mkdir()
    write_arrays(output / "forecast-field.npz", east_m=grid.east, north_m=grid.north, **arrays)
    report = {"source_sha256": sha256(payload).hexdigest(), "features": features,
              "origin": origins.iloc[0].isoformat(), "valid_at": times.iloc[0].isoformat(),
              "site_count": len(site_ids), "draw_count": len(draw_ids), "central_mass": probability,
              "source_kind": "model_forecasts", "map_operator": "fixed_idw",
              "uncertainty": "forecast ensemble spread propagated through a fixed interpolation operator",
              "validation": "no independent field labels provided"}
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draws", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    map_draw_file(args.draws, read_json(args.config), args.output)


if __name__ == "__main__":
    main()
