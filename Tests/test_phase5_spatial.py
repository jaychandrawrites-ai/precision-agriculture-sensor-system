import unittest

import numpy as np

from Framework.Backend import create_backend
from Spatial.BlockSampler import BlockSampling, block_metropolis
from Spatial.EnergyFunctional import EnergyFunctional, EnergyWeights
from Spatial.GaussianField import exact_gaussian_field
from Spatial.Kriging import CovarianceModel, KrigingModel, nearest_neighbor
from Spatial.PosteriorStudy import interval_scores, mixing_report, predictive_summary
from Spatial.SeedManager import seed_constraints
from Spatial.SpatialGrid import SpatialGrid
from Spatial.Uncertainty import summarize
from Spatial.VariogramFit import fit_variogram


class KrigingTests(unittest.TestCase):
    def test_affine_trend_extrapolates_exact_line_and_batch_shapes(self):
        sites = np.array([[0, 0], [1, 2], [0, 3], [2, 0], [2, 3]], dtype=float)
        response = 20 + 3 * sites[:, 0] - sites[:, 1]
        query = np.array([[3., 3.], [1.5, 2.0], [0., 0.]])
        model = KrigingModel(sites, response, CovarianceModel(3, shape="spherical"), trend="affine")
        prediction = model.predict(query, batch_size=1)
        np.testing.assert_allclose(prediction.mean, 20 + 3 * query[:, 0] - query[:, 1], atol=1e-10)
        self.assertAlmostEqual(prediction.latent_variance[-1], 0.0, places=12)
        self.assertGreater(prediction.latent_variance[0], prediction.latent_variance[1])

    def test_heteroscedastic_replicates_and_anisotropy(self):
        sites = np.array([[0, 0], [0, 0], [1, 0]], dtype=float)
        values = [1, 100, 1]
        shape = CovarianceModel(2, shape="matern32", minor_ratio=0.25)
        model = KrigingModel(sites, values, shape, measurement_variance=[0.001, 1000, 0.001])
        self.assertLess(model.predict([[0, 0]]).mean[0], 1.1)
        cov = shape(np.zeros((1, 2)), np.array([[1, 0], [0, 1]]))[0]
        self.assertGreater(cov[0], cov[1])

    def test_rank_and_nearest_ties(self):
        sites = [[0, 0], [0, 1], [0, 2]]
        with self.assertRaisesRegex(ValueError, "geometry"):
            KrigingModel(sites, [3, 2, 5], CovarianceModel(1), trend="affine")
        np.testing.assert_allclose(nearest_neighbor(sites, [3, 2, 5], [[0, 0.5], [0, 2]]), [2.5, 5])

    def test_variogram_fits_exact_exponential(self):
        distance = np.linspace(0.3, 9, 18)
        gamma = 0.15 + 2.5 * -np.expm1(-distance / 3.2)
        result = fit_variogram({"distances": distance, "semivariance": gamma,
                                "pairs": np.full(len(distance), 30)},
                               ranges=[2, 3.2, 6], shapes=("spherical", "exponential"))
        self.assertEqual(result.model.shape, "exponential")
        self.assertAlmostEqual(result.model.range_m, 3.2)
        self.assertAlmostEqual(result.model.sill, 2.5)
        self.assertAlmostEqual(result.model.nugget, 0.15)
        self.assertLess(result.weighted_rmse, 1e-10)


class BlockFieldTests(unittest.TestCase):
    def test_exact_and_metropolis_conditional_moments(self):
        backend = create_backend(dtype="float64")
        grid = SpatialGrid(np.arange(2.), np.arange(2.))
        constraints = seed_constraints(grid, [[0., 0.]], [3.], 1, hard=True)
        energy = EnergyFunctional(grid, constraints, backend, EnergyWeights(2.0, 0.0))
        exact = exact_gaussian_field(energy)
        np.testing.assert_allclose(exact.mean, 3)
        matrix = np.array([[4., 0, -2], [0, 4, -2], [-2, -2, 4]])
        np.testing.assert_allclose(exact.covariance, np.linalg.inv(matrix))
        settings = BlockSampling(draws=2800, thin=1, burn_in=500, seed=112)
        first = block_metropolis(energy, np.full(grid.shape, 3.0), settings)
        second = block_metropolis(energy, np.full(grid.shape, 3.0), settings)
        np.testing.assert_array_equal(first.fields, second.fields)
        np.testing.assert_array_equal(first.fields[..., 0, 0], 3.0)
        np.testing.assert_allclose(first.fields.mean(axis=(0, 1)), exact.mean, atol=0.095)
        np.testing.assert_allclose(first.fields.var(axis=(0, 1)), exact.variance, atol=0.095)
        components, _ = energy.value_gradient(backend.array(first.fields[0, 0]))
        self.assertAlmostEqual(first.energy_trace[0, 0], sum(components), places=10)
        self.assertTrue(mixing_report(summarize(first), active=~constraints.fixed)["all_active_pass"])

    def test_density_and_variance_propagation(self):
        backend = create_backend(dtype="float64")
        grid = SpatialGrid(np.arange(2.), np.arange(2.))
        with self.assertRaisesRegex(ValueError, "improper"):
            energy = EnergyFunctional(grid, seed_constraints(grid, [[0, 0]], [1], 2),
                                      backend, EnergyWeights(0, 0))
            block_metropolis(energy, np.ones(grid.shape), BlockSampling(draws=2, burn_in=0))
        first = seed_constraints(grid, [[0, 0], [0, 0]], [1, 100], 1,
                                 measurement_variance=[1, 99])
        self.assertAlmostEqual(first.targets[0, 0], 1.99)
        self.assertAlmostEqual(first.weights[0, 0], 100 / 99)
        settings = BlockSampling(draws=5, burn_in=0)
        scenarios = [block_metropolis(
            EnergyFunctional(grid, seed_constraints(grid, grid.points(), [value] * 4,
                                                     1, hard=True), backend),
            np.zeros(grid.shape), settings) for value in (1., 3.)]
        intervals = predictive_summary(grid, scenarios, [[0, 0]], observation_variance=[4.])
        np.testing.assert_array_equal(intervals.mean, 2)
        np.testing.assert_array_equal(intervals.conditional_variance, 0)
        np.testing.assert_array_equal(intervals.scenario_variance, 1)
        np.testing.assert_array_equal(intervals.total_variance, 5)
        self.assertEqual(interval_scores(intervals, [2.])["coverage"], 1.)

    def test_preconditioned_variational_solution_matches_reference(self):
        from Spatial.SpatialModel import reconstruct_variational
        from config import SpatialConfig

        backend = create_backend(dtype="float64")
        grid = SpatialGrid(np.arange(4.), np.arange(4.))
        energy, result = reconstruct_variational(
            grid, [[0, 0], [2, 3], [2, 2]], [2, 1, -1], backend,
            SpatialConfig(iterations=1000), algorithm="preconditioned_cg",
            measurement_variance=[1e-4, 0.1, 0.1])
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.field, exact_gaussian_field(energy).mean, atol=1e-5)
