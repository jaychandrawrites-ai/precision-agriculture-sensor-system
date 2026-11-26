import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from Acquisition.AITrace import TraceStore
from Acquisition.TraceAnalysis import ensemble_forecast_maps
from Examples.SyntheticTrace import run
from Utils.Serialization import read_json


class SurveyUncertaintyTests(unittest.TestCase):
    def test_coherent_composition_and_preissue_calibration(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "trace-fixture"
            run(root, ensemble=True)
            manifest = read_json(root / "ensemble-maps.json")
            forecast = read_json(root / "forecasts.json")
            store = TraceStore(root / "trace", recover=False)
            args = store, manifest["survey_id"], forecast, manifest["issued_at"]
            geometry = {"feature_scales": [1.] * 6, "covariance": {"range_m": 50.},
                        "refinement": {"max_depth": 2, "observation_radius_m": 5.}}
            baseline, _ = ensemble_forecast_maps(*args, steps=6, adaptive=geometry)
            source = baseline["site_member_forecasts"]
            settings = {
                "feature_scales": [.2] * 6, "draws": 20, "residual_sites": manifest["sites"],
                "residual_origins": ["2020-06-01T00:00:00Z", "2020-06-01T06:00:00Z"],
                "residuals": np.stack((np.full_like(source[0], .03), np.full_like(source[0], -.02))),
                "physical": {
                    "issue_at": manifest["issued_at"], "sites": manifest["sites"],
                    "features": manifest["features"], "checkpoint_sha256": manifest["checkpoint_sha256"][0],
                    "draw_ids": ["fixture-low", "fixture-high"],
                    "forecasts": np.stack((source[0] - .1, source[0] + .1)),
                },
                "covariance_scenarios": [
                    {"id": "short", "weight": .5, "covariance": {"range_m": 30.}},
                    {"id": "long", "weight": .5, "covariance": {"range_m": 100., "nugget": .05}},
                ],
            }
            arrays, metadata = ensemble_forecast_maps(*args, steps=6, adaptive=geometry, predictive=settings)
            repeated, _ = ensemble_forecast_maps(*args, steps=6, adaptive=geometry, predictive=settings)
            draws = arrays["predictive_draws"]
            np.testing.assert_array_equal(draws, repeated["predictive_draws"])
            np.testing.assert_allclose(draws.mean(axis=0), arrays["predictive_mean"])
            np.testing.assert_allclose(np.quantile(draws, .5, axis=0), arrays["predictive_median"])
            self.assertEqual(draws.shape[:2] + draws.shape[3:], (20, 6, 6))
            self.assertEqual(len(set(arrays["predictive_covariance_index"])), 2)
            self.assertNotIn(0, arrays["predictive_member_index"])
            np.testing.assert_allclose(
                arrays["predictive_site_draws"],
                arrays["predictive_source_trajectories"][arrays["predictive_member_index"]]
                + settings["residuals"][arrays["predictive_residual_index"]])
            self.assertEqual(metadata["predictive"]["members"][-1]["physical_scenario_id"], "fixture-high")
            settings["memory_bytes"] = 64
            with self.assertRaisesRegex(ValueError, "memory"):
                ensemble_forecast_maps(*args, steps=6, adaptive=geometry, predictive=settings)
            settings["residual_origins"][-1] = manifest["issued_at"]
            with self.assertRaisesRegex(ValueError, "pre-issue"):
                ensemble_forecast_maps(*args, steps=6, adaptive=geometry, predictive=settings)

    def test_variational_operator_preserves_shared_geometry_and_constant_values(self):
        from Spatial.SurveyUncertainty import variational_trajectories

        points = np.array([[1., 1.], [7., 1.], [1., 7.]])
        members = np.ones((2, 3, 2, 6)) * np.arange(1, 7)
        arrays, metadata = variational_trajectories(
            points, members, (0, 0, 8, 8),
            {"feature_scales": [1.] * 6, "hard": True,
             "refinement": {"max_depth": 2, "observation_radius_m": .2}}, .9)
        self.assertTrue(metadata["hard"])
        self.assertGreater(len(arrays["variational_bounds_m"]), 4)
        expected = np.broadcast_to(np.arange(1, 7), arrays["variational_draws"].shape)
        np.testing.assert_allclose(arrays["variational_draws"], expected, atol=1e-6)

    def test_scenarios_rebuild_from_closed_survey_segments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "trace"
            run(root, ensemble=True)
            manifest = read_json(root / "ensemble-maps.json")
            store = TraceStore(root / "trace", recover=False)
            options = {"feature_scales": [1.] * 6, "draws": 8,
                       "covariance_scenarios": [{"id": "fixture", "weight": 1.,
                                                 "covariance": {"range_m": 50.}}]}
            physical = {"checkpoint": root / "synthetic-model.json",
                        "field_configuration": read_json(root / "field-settings.json"),
                        "scenarios": {"draws": 4, "ranges": [
                            {"path": "crop.critical_temperature_c", "lower": 25., "upper": 42.}
                        ]}}
            arrays, report = ensemble_forecast_maps(
                store, manifest["survey_id"], read_json(root / "forecasts.json"),
                manifest["issued_at"], steps=6, grid_size=5, predictive=options,
                physical_sources=physical)
            self.assertEqual(arrays["predictive_source_trajectories"].shape, (6, 5, 6, 6))
            self.assertEqual(len(report["predictive"]["physical_provenance"]["parameter_draws"]), 4)

    def test_irregular_mask_excludes_hole_and_preserves_statistics(self):
        from Spatial.SurveyUncertainty import predictive_fields

        settings = {
            "feature_scales": [1.], "draws": 8,
            "field_polygon_m": [[0, 0], [8, 0], [8, 3], [5, 3], [5, 8], [0, 8]],
            "field_holes_m": [[[1, 1], [3, 1], [3, 3], [1, 3]]],
            "covariance_scenarios": [{"id": "shape", "weight": 1., "covariance": {"range_m": 10.}}],
        }
        arrays, _ = predictive_fields(
            [[.5, .5], [4., 4.]], np.ones((2, 2, 3, 1)),
            [[.5, .5], [2., 2.], [7., 6.], [4., 6.]], settings, .9)
        np.testing.assert_array_equal(arrays["predictive_inside"], [True, False, False, True])
        self.assertTrue(np.isnan(arrays["predictive_mean"][:, 1:3]).all())
        np.testing.assert_allclose(arrays["predictive_mean"], arrays["predictive_draws"].mean(axis=0))

    def test_combined_variational_draws_are_independent_of_cell_chunk_size(self):
        from Spatial.SurveyUncertainty import predictive_fields, variational_trajectories

        points = np.array([[1., 1.], [7., 1.], [1., 7.]])
        members = np.arange(72, dtype=float).reshape(2, 3, 2, 6)
        mesh, _ = variational_trajectories(
            points, members, (0, 0, 8, 8),
            {"feature_scales": [1.] * 6, "hard": True,
             "refinement": {"max_depth": 2, "observation_radius_m": .2}}, .9)
        settings = {"method": "adaptive_variational", "feature_scales": [1.] * 6,
                    "draws": 12, "residuals": np.full((2, 3, 2, 6), .25),
                    "residual_origins": ["a", "b"]}
        args = points, members, mesh["variational_centers_m"]
        outputs = [predictive_fields(*args, {**settings, "chunk_cells": chunk}, .9,
                                     operator=mesh["variational_weights"])[0]
                   for chunk in (1, 5, 32)]
        for other in outputs[1:]:
            np.testing.assert_allclose(outputs[0]["predictive_draws"], other["predictive_draws"])
            np.testing.assert_allclose(outputs[0]["predictive_upper"], other["predictive_upper"])
        np.testing.assert_allclose(outputs[0]["predictive_draws"], np.einsum(
            "qs,mslf->mlqf", mesh["variational_weights"], outputs[0]["predictive_site_draws"]))
        with self.assertRaisesRegex(ValueError, "require kriging"):
            predictive_fields(*args, {**settings, "covariance_scenarios": [{}]}, .9,
                              operator=mesh["variational_weights"])
