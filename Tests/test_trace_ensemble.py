"""End-to-end modeled spread without reporting synthetic coordinates."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from Acquisition.AITrace import TraceStore
from Acquisition.TraceAnalysis import ensemble_forecast_maps
from Examples.SyntheticTrace import run
from Utils.Serialization import read_json


class TraceEnsembleTests(unittest.TestCase):
    def test_shared_draw_axis_and_time_index_survive_survey_mapping(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic"
            report = run(output, ensemble=True)
            self.assertEqual(report["ready_locations"], 5)
            manifest = read_json(output / "ensemble-maps.json")
            self.assertEqual(len(manifest["valid_at"]), 6)
            self.assertEqual(len(manifest["features"]), 6)
            self.assertEqual(len(set(manifest["checkpoint_sha256"])), 2)
            self.assertNotIn("latitude", str(manifest))
            with np.load(output / "ensemble-maps.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["site_member_forecasts"].shape, (2, 5, 6, 6))
                self.assertEqual(arrays["mean"].shape, (6, 25, 25, 6))
                draws = np.einsum("yxs,mslf->mlyxf", arrays["interpolation_weights"],
                                  arrays["site_member_forecasts"])
                np.testing.assert_allclose(draws.mean(axis=0), arrays["mean"], atol=1e-10)
                np.testing.assert_allclose(draws.std(axis=0), arrays["spread"], atol=1e-10)
                self.assertTrue(np.any(arrays["spread"] > 0))
                self.assertTrue(np.all(arrays["lower"] <= arrays["upper"]))

            adaptive = {
                "feature_scales": [5., 10., .1, .1, .1, .2],
                "covariance": {"range_m": 150.},
                "refinement": {"max_depth": 2, "observation_radius_m": 5., "max_leaves": 16},
                "field_polygon_m": [[0., 0.], [30., 0.], [30., 30.], [0., 30.]],
            }
            arrays, metadata = ensemble_forecast_maps(
                TraceStore(output / "trace", recover=False), manifest["survey_id"],
                read_json(output / "forecasts.json"), manifest["issued_at"], steps=6, adaptive=adaptive)
            self.assertGreater(metadata["adaptive"]["leaf_count"], 1)
            self.assertEqual(arrays["adaptive_draws"].shape, (2, 6, metadata["adaptive"]["leaf_count"], 6))
            draws = np.einsum("qs,mslf->mlqf", arrays["adaptive_weights"], arrays["site_member_forecasts"])
            np.testing.assert_allclose(draws, arrays["adaptive_draws"])
            np.testing.assert_allclose(draws.mean(axis=0), arrays["adaptive_mean"])
            np.testing.assert_allclose(np.quantile(draws, .95, axis=0), arrays["adaptive_upper"])
            np.testing.assert_allclose(arrays["adaptive_weights"].sum(axis=1), 1.)
            self.assertLess(arrays["adaptive_inside"].sum(), metadata["adaptive"]["leaf_count"])
