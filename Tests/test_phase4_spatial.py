import unittest

import numpy as np

from Numerical.Differentiation import jacobian
from Spatial.Geostatistics import compare_interpolators, spatial_folds, variogram
from Spatial.RobustField import RobustField
from Spatial.SpatialGrid import SpatialGrid


class AdvancedSpatialTests(unittest.TestCase):
    def test_spatial_groups_and_constant_interpolation(self):
        points = np.array([[0, 0], [0, 0], [0, 1], [3, 2], [4, 0]])
        folds = list(spatial_folds(points))
        self.assertTrue(any(np.array_equal(test, [0, 1]) for _, test in folds))
        result = compare_interpolators(points, np.full(len(points), 2.), 3, block_size=1.5)
        self.assertEqual(sum(result["fold_sizes"]), len(points))
        for metrics in result["metrics"].values():
            self.assertLess(metrics["rmse"][0], 1e-8)
        result = variogram(points, points[:, 0], [0, 2, 5, 9])
        self.assertEqual(result["pairs"].sum(), 10)
        self.assertTrue(np.isnan(result["semivariance"][-1]))

    def test_robust_anisotropic_gradient_and_fixed_cell(self):
        grid = SpatialGrid([0, 1, 2], [0, 2, 4])
        rng = np.random.default_rng(3)
        y = rng.normal(size=grid.shape)
        field = RobustField(grid, [[1, 2], [3, 4], [5, 6]], .2, y,
                            np.arange(9).reshape(3, 3), curvature=.4)
        x = rng.normal(size=grid.shape)
        numeric = jacobian(lambda p: field.value_gradient(p.reshape(grid.shape))[0], x.ravel())
        np.testing.assert_allclose(numeric, field.value_gradient(x)[1].ravel(), atol=1e-7)
        fixed = np.eye(3, dtype=bool)
        estimate, result = field.fit(x, fixed, y, iterations=500)
        np.testing.assert_array_equal(estimate[fixed], y[fixed])
        self.assertEqual(result.status, "converged")
        self.assertLess(result.loss_trace[-1], result.loss_trace[0])

    def test_burn_in_adaptation_is_reproducible(self):
        from Framework.Backend import create_backend
        from Spatial.EnergyFunctional import EnergyFunctional
        from Spatial.MonteCarlo import SamplingConfig, metropolis
        from Spatial.SeedManager import seed_constraints

        grid = SpatialGrid([0, 1], [0, 1])
        energy = EnergyFunctional(grid, seed_constraints(grid, [[0, 0]], [1], 10),
                                  create_backend(dtype="float64"))
        config = SamplingConfig(draws=20, burn_in=80, adapt=True, proposal="uniform")
        first = metropolis(energy, np.ones(grid.shape), config)
        second = metropolis(energy, np.ones(grid.shape), config)
        self.assertFalse(np.all(first.proposal_scales == 1))
        np.testing.assert_array_equal(first.fields, second.fields)
