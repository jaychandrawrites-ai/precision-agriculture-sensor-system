"""Covariance conditioning regressions across supported geostatistical kernels."""
import unittest

import numpy as np

from Spatial.Kriging import CovarianceModel, KrigingModel


class CovarianceFamilyTests(unittest.TestCase):
    def test_rotated_kernels_are_positive_definite_and_preserve_variance_contract(self):
        points = np.array([[0., 0.], [2., 0.], [0., 2.], [2., 2.], [1., 3.]])
        values = np.sin(points[:, 0])
        for family in ("spherical", "exponential", "gaussian", "matern32", "matern52"):
            with self.subTest(family=family):
                config = dict(range_m=4., sill=2., shape=family, angle_deg=31., minor_ratio=.4)
                cov = CovarianceModel(**config)
                kernel = cov(points, points)
                np.testing.assert_allclose(kernel, kernel.T)
                np.testing.assert_allclose(kernel.diagonal(), 2.)
                self.assertGreater(np.linalg.eigvalsh(kernel).min(), 0.)
                exact = KrigingModel(points, values, cov).predict(points)
                np.testing.assert_allclose(exact.mean, values, atol=1e-10)
                np.testing.assert_allclose(exact.latent_variance, 0., atol=1e-10)
                noisy = KrigingModel(points, values, CovarianceModel(**config, nugget=.3),
                                     measurement_variance=np.linspace(0., .5, len(points))).predict(points)
                self.assertTrue((noisy.latent_variance > 0).all())

    def test_duplicate_sites_are_conditioned_without_losing_constant_drift(self):
        points = np.array([[0., 0.], [0., 0.], [1e-12, 0.], [2., 0.], [0., 2.]])
        for family in ("spherical", "exponential", "gaussian", "matern32", "matern52"):
            with self.subTest(family=family):
                model = KrigingModel(points, np.full(len(points), 3.),
                                     CovarianceModel(range_m=50., shape=family))
                self.assertGreater(model.jitter, 0.)
                result = model.predict([[1., 1.], [4., 5.]])
                np.testing.assert_allclose(result.mean, 3.)
                self.assertTrue(np.isfinite(result.latent_variance).all())

    def test_invalid_parameters_are_refused(self):
        for values in ({"range_m": 0}, {"sill": -1}, {"minor_ratio": 2},
                       {"nugget": -.1}, {"angle_deg": np.nan}, {"shape": "unsupported"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CovarianceModel(**({"range_m": 1.} | values))

    def test_posterior_report_keeps_holdout_values_out_of_conditioning(self):
        from validate_spatial import evaluate

        spec = {
            "conditioning": {"ids": ["a", "b", "c"], "points": [[0, 0], [0, 1], [1, 0]], "values": [3, 3, 3]},
            "held_out": {"ids": ["x", "y"], "points": [[1, 1], [2, 2]], "values": [3, 3]},
            "scenarios": [{"id": "first", "covariance": {"range_m": 1}},
                          {"id": "second", "covariance": {"range_m": 2, "shape": "matern52"}}],
            "draws": 1000,
        }
        first = evaluate(spec)
        spec["held_out"]["values"] = [30, 30]
        second = evaluate(spec)
        a, b = (report["verification"]["site_diagnostics"] for report in (first, second))
        np.testing.assert_allclose([item["mean"] for item in a], 3., atol=.08)
        self.assertEqual([item["mean"] for item in a], [item["mean"] for item in b])
        self.assertGreater(second["verification"]["rmse"], first["verification"]["rmse"])
        spec["held_out"]["points"][0] = [0, 0]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluate(spec)
