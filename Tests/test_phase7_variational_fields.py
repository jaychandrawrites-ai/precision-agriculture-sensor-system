import unittest

import numpy as np

from Framework.Backend import create_backend
from Spatial.AdaptiveVariational import FiniteVolumeEnergy, VariationalRefinement, adaptive_variational
from Spatial.BlockSampler import BlockSampling, block_metropolis
from Spatial.CovarianceStudy import (CovarianceScenario, covariance_scenario_fields,
                                     posterior_predictive_checks, visible_kriging)
from Spatial.EnergyFunctional import EnergyFunctional
from Spatial.FieldDomain import FieldDomain
from Spatial.GaussianField import exact_gaussian_field
from Spatial.Kriging import CovarianceModel, KrigingModel
from Spatial.MonteCarlo import SamplingConfig, metropolis
from Spatial.SeedManager import seed_constraints
from Spatial.SpatialGrid import SpatialGrid
from Spatial.VariationalSolver import minimize


SQUARE = [[0, 0], [8, 0], [8, 8], [0, 8]]
HOLE = [[2, 2], [6, 2], [6, 6], [2, 6]]


class AdaptiveVariationalTests(unittest.TestCase):
    def test_gradient_matches_finite_differences_on_unequal_cells(self):
        energy = FiniteVolumeEnergy([1, 2, 3], [(0, 1, .3, 1), (1, 2, .5, 2)],
                                    1.3, .8, [1, 2, 0], [.2, .5, 0])
        x = np.array([3., 4., 2.])
        components, grad = energy.evaluate(x)
        self.assertEqual(set(components), {"gradient", "curvature", "fidelity"})
        for i in range(3):
            d = np.eye(3)[i]*1e-5
            plus = sum(energy.evaluate(x + d)[0].values())
            minus = sum(energy.evaluate(x - d)[0].values())
            self.assertAlmostEqual(grad[i], (plus - minus)*5e4, places=8)
            self.assertAlmostEqual(energy.diagonal()[i], energy.action(np.eye(3)[i])[i])

    def test_budget_and_hard_constraints_constant_field(self):
        settings = VariationalRefinement(initial_depth=1, max_depth=4, max_cells=40,
                                         observation_radius_m=.5, uncertainty_threshold=.8)
        result = adaptive_variational([0, 0, 8, 8], [[1, 1], [7, 7]], [2, 2], settings, hard=True,
                                      uncertainty=lambda p: (p[:, 0] > 5).astype(float))
        active = [cell for cell in result["cells"] if cell["inside"]]
        self.assertEqual(len(active), result["leaf_count"])
        self.assertLessEqual(len(active), settings.max_cells)
        self.assertGreater(len({cell["depth"] for cell in active}), 1)
        self.assertEqual(sum(cell["hard_constraint"] for cell in active), 2)
        np.testing.assert_allclose([cell["center_mean"] for cell in active], 2, atol=1e-7)
        self.assertTrue(result["history"][-1]["solver"]["converged"])

    def test_polygon_hole_exclusion_and_conflicting_hard_geometry(self):
        settings = VariationalRefinement(initial_depth=1, max_depth=4, max_cells=160,
                                         gradient_threshold=.2)
        result = adaptive_variational([0, 0, 8, 8], [[1, 1], [7, 7]], [1, 3], settings,
                                      hard=True, polygon=SQUARE, holes=[HOLE])
        self.assertGreater(result["maximum_depth"], 1)
        used = np.array([x["bounds_m"] for x in result["cells"] if x["inside"]])
        centers = (used[:, :2] + used[:, 2:]) / 2
        self.assertTrue(FieldDomain(SQUARE, (HOLE,)).contains(centers).all())
        self.assertEqual(sum(x["hard_constraint"] for x in result["cells"]), 2)
        with self.assertRaisesRegex(ValueError, "conflicting hard"):
            adaptive_variational([0, 0, 8, 8], [[1, 1], [1, 1]], [1, 3], settings, hard=True)

    def test_regular_domain_solver_and_samplers_ignore_exclusions(self):
        grid = SpatialGrid(np.arange(9), np.arange(9))
        backend = create_backend(dtype="float64")
        domain = FieldDomain(SQUARE, (HOLE,))
        constraints = seed_constraints(grid, [[1., 1.], [7., 7.]], [3., 3.], 15., hard=True)
        energy = EnergyFunctional(grid, constraints, backend, domain=domain)
        solution = minimize(energy, np.full(grid.shape, 3)).field
        active = backend.to_host(energy.active)
        np.testing.assert_allclose(solution[active], 3., atol=1e-7)
        np.testing.assert_array_equal(solution[~active], 0.)
        exact = exact_gaussian_field(energy)
        np.testing.assert_allclose(exact.mean[active], 3., atol=1e-8)
        for samples in (metropolis(energy, solution, SamplingConfig(draws=3, burn_in=2)),
                        block_metropolis(energy, solution, BlockSampling(draws=3, burn_in=2))):
            np.testing.assert_array_equal(samples.fields[..., ~active], 0.)

    def test_excluded_cells_do_not_anchor_active_components(self):
        grid = SpatialGrid(np.arange(9), np.arange(9))
        backend = create_backend(dtype="float64")
        domain = FieldDomain(SQUARE, (HOLE,))
        constraints = seed_constraints(grid, [[1., 1.]], [3.], 15.)
        no_constraints = type(constraints)(np.zeros(grid.shape), constraints.targets,
                                           constraints.fixed, constraints.fixed_values)
        energy = EnergyFunctional(grid, no_constraints, backend, domain=domain)
        with self.assertRaisesRegex(ValueError, "unanchored"):
            metropolis(energy, np.zeros(grid.shape))


class CovarianceUncertaintyTests(unittest.TestCase):
    def test_visibility_does_not_bridge_excluded_hole(self):
        domain = FieldDomain(SQUARE, (HOLE,))
        prediction = visible_kriging([[1, 4], [7, 4]], [1, 100],
                                      [[1.5, 4], [6.5, 4], [4, 4]],
                                      CovarianceModel(3), domain)
        np.testing.assert_array_equal(prediction.support_count, [1, 1, 0])
        np.testing.assert_array_equal(prediction.supported, [True, True, False])
        np.testing.assert_allclose(prediction.mean[:2], [1, 100])
        self.assertTrue(np.isnan(prediction.mean[2]))

    def test_coherent_covariance_matches_analytic_moments(self):
        points = [[0, 0], [5, 0], [0, 4], [5, 4]]
        values = [1., 3., 2., 5.]
        queries = [[2, 2], [2.01, 2], [4, 2]]
        scenarios = [CovarianceScenario("steady", CovarianceModel(2, sill=2, nugget=.2, shape="gaussian")),
                     CovarianceScenario("oriented", CovarianceModel(4, sill=1, nugget=.1,
                                                                   shape="spherical", angle_deg=28,
                                                                   minor_ratio=.63))]
        result = covariance_scenario_fields(points, values, queries, scenarios, draws=10000,
                                            measurement_variance=[.2, 0, .1, .2])
        self.assertEqual(result.draws.shape, (2, 10000, 3))
        np.testing.assert_allclose(result.draws.mean(axis=1), result.scenario_means, rtol=.03)
        np.testing.assert_allclose(result.draws.reshape(-1, 3).var(axis=0), result.variance, rtol=.05)
        self.assertGreater(np.corrcoef(result.draws[0].T)[0, 1], .98)
        self.assertTrue(all(item["observation_condition_number"] >= 1 for item in result.diagnostics))

    def test_predictive_checks_use_observation_noise_once(self):
        fields = np.repeat([[1., 2.]], 10000, axis=0)
        no_noise = posterior_predictive_checks(fields, [1, 2])
        self.assertIsNone(no_noise["site_diagnostics"][0]["standardized_residual"])
        with_noise = posterior_predictive_checks(fields, [1, 2], observation_variance=[1, 0])
        self.assertAlmostEqual(with_noise["site_diagnostics"][0]["predictive_sd"], 1, places=1)
        self.assertEqual(with_noise["site_diagnostics"][1]["predictive_sd"], 0)
        self.assertEqual(with_noise["coverage"], 1.)
        self.assertIn("spatial_standard_deviation", with_noise["replicated_field_checks"])
