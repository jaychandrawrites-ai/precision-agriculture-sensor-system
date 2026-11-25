import unittest

import numpy as np

from config import SpatialConfig
from Framework.Backend import create_backend
from Spatial.EnergyFunctional import EnergyFunctional, EnergyWeights
from Spatial.Hamiltonian import hessian_columns
from Spatial.MonteCarlo import SamplingConfig, metropolis
from Spatial.SeedManager import Constraints, seed_constraints
from Spatial.SpatialGrid import SpatialGrid, local_coordinates
from Spatial.SpatialModel import reconstruct_variational
from Spatial.Uncertainty import summarize
from Spatial.VariationalSolver import minimize


class SpatialTests(unittest.TestCase):
    def setUp(self):
        self.grid = SpatialGrid([0, 2, 4], [0, 1])
        self.backend = create_backend(dtype="float64")

    def test_energy_gradients_and_sparse_hessian_match_finite_difference(self):
        constraints = seed_constraints(self.grid, [[0, 0], [4, 1]], [5, 8], 3)
        field = np.array([[1, 2, 4], [3, 5, 6]], dtype=float)
        for boundary in ("neumann", "periodic"):
            with self.subTest(boundary=boundary):
                energy = EnergyFunctional(self.grid, constraints, self.backend, boundary=boundary)
                _, gradient = energy.value_gradient(field)
                columns = hessian_columns(energy)
                for i, (nodes, values, diagonal) in enumerate(columns):
                    direction = np.zeros_like(field)
                    direction.flat[i] = 1e-5
                    plus, plus_grad = energy.value_gradient(field + direction)
                    minus, minus_grad = energy.value_gradient(field - direction)
                    self.assertAlmostEqual(gradient.flat[i], (sum(plus) - sum(minus)) / 2e-5,
                                           places=7)
                    expected = np.zeros(field.size)
                    expected[nodes] = values
                    self.assertAlmostEqual(diagonal, expected[i])
                    np.testing.assert_allclose((plus_grad - minus_grad).ravel() / 2e-5,
                                               expected, atol=1e-9)

    def test_hard_and_soft_boundaries_and_seed_fidelity(self):
        hard_boundary = np.array([[np.nan, 12, np.nan], [np.nan] * 3])
        settings = SpatialConfig(iterations=500)
        energy, hard = reconstruct_variational(self.grid, [[0, 0], [4, 1]], [5, 8],
                                               self.backend, settings, hard_seeds=True,
                                               boundary_values=hard_boundary)
        self.assertEqual(hard.field[0, 1], 12)
        self.assertEqual(hard.field[0, 0], 5)
        self.assertEqual(hard.field[1, 2], 8)
        self.assertTrue((np.diff(hard.energies) <= 1e-10).all())
        with self.assertRaisesRegex(ValueError, "stability"):
            minimize(energy, hard.field, step=2 / energy.lipschitz)
        _, soft = reconstruct_variational(self.grid, [[0, 0], [4, 1]], [5, 8],
                                          self.backend, settings, hard_seeds=True,
                                          boundary_values=hard_boundary, boundary_weight=0.1)
        self.assertLess(soft.field[0, 1], 12)
        self.assertEqual(soft.field[0, 0], 5)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            seed_constraints(self.grid, [[2, 0]], [4], 3, hard=True, boundary=hard_boundary)

    def test_metropolis_matches_independent_gaussian_and_is_reproducible(self):
        shape = self.grid.shape
        frozen = np.zeros(shape, dtype=bool)
        frozen[0, 0] = True
        targets = np.full(shape, 3.0)
        constraints = Constraints(np.full(shape, 4.0), targets, frozen, targets)
        energy = EnergyFunctional(self.grid, constraints, self.backend, EnergyWeights(0, 0))
        config = SamplingConfig(draws=450, burn_in=150, thin=3, chain_count=4, seed=99,
                                temperature=1.0, proposal_scale=1.4)
        result = metropolis(energy, targets, config)
        twin = metropolis(energy, targets, config)
        np.testing.assert_array_equal(result.fields, twin.fields)
        self.assertTrue(((result.acceptance > 0.3) & (result.acceptance < 0.8)).all())
        stats = summarize(result)
        np.testing.assert_allclose(stats.mean[~frozen], 3, atol=0.10)
        np.testing.assert_allclose(stats.std[~frozen], 0.5, atol=0.08)
        self.assertEqual(stats.std[0, 0], 0)
        self.assertTrue(np.isnan(stats.split_rhat[0, 0]))
        self.assertTrue((stats.split_rhat[~frozen] < 1.1).all())
        components, _ = energy.value_gradient(result.fields[0, -1])
        self.assertAlmostEqual(sum(components), result.energy_trace[0, -1], places=9)

    def test_sampler_rejects_unanchored_field_and_handles_dateline(self):
        shape = self.grid.shape
        empty = Constraints(np.zeros(shape), np.zeros(shape), np.zeros(shape, bool), np.zeros(shape))
        energy = EnergyFunctional(self.grid, empty, self.backend)
        with self.assertRaisesRegex(ValueError, "unanchored"):
            metropolis(energy, np.zeros(shape))
        east, north = local_coordinates([[0, -179.999]], [0, 179.999])[0]
        self.assertAlmostEqual(east, 222.39, places=2)
        self.assertEqual(north, 0)
