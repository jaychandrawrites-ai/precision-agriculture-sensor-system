from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from Spatial.SpatialGrid import SpatialGrid
from spatial_forecast import ensemble_maps, map_draw_file


class ForecastMaps(unittest.TestCase):
    def test_idw_preserves_joint_between_site_draw_dependence(self):
        grid = SpatialGrid([0, 1, 2], [0, 1])
        points = [[0, 0], [2, 0]]
        members = np.array([[[-1., 0.], [1., 2.]], [[1., 0.], [-1., 2.]]])
        results = ensemble_maps(grid, points, members, batch_size=2)
        np.testing.assert_allclose(results["spread"][0, :, 0], [1, 0, 1])
        np.testing.assert_allclose(results["mean"][0, :, 1], [0, 1, 2])

    def test_map_file_rejects_unmatched_sites_and_retains_forecast_provenance(self):
        base = {"forecast_origin": "2019-01-01T00:00Z", "valid_at": "2019-01-01T03:00Z",
                "value_origin": "model_forecast", "latitude_deg": 0.}
        rows = [{**base, "site_id": site, "draw_id": draw, "longitude_deg": lon,
                 "air_temperature_c": 20 + draw} for site, lon in (("a", 0.), ("b", .00001))
                for draw in (1, 2)]
        config = {"origin_lat_lon": [0., 0.], "features": ["air_temperature_c"],
                  "east_m": [0., 1.], "north_m": [0., 1.]}
        with TemporaryDirectory() as path:
            source = Path(path) / "draws.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            report = map_draw_file(source, config, Path(path) / "map")
            self.assertEqual(report["site_count"], 2)
            self.assertEqual(report["source_kind"], "model_forecasts")
            pd.DataFrame(rows[:-1]).to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "same sites"):
                map_draw_file(source, config, Path(path) / "map2")
