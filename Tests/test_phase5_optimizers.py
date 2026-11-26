import unittest

import numpy as np

from Framework.Backend import create_backend
from Framework.Layers import Parameter
from Framework.Optimizer import Adagrad, GradientAccumulator, SGD


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.backend = create_backend(dtype="float64")

    def parameter(self, value):
        weights = self.backend.array(value, copy=True)
        return Parameter(weights, self.backend.zeros(weights.shape), "test")

    def test_adagrad_matches_reference_and_portable_state(self):
        parameter = self.parameter([1.5, -0.6])
        optimizer = Adagrad([parameter], self.backend, learning_rate=.15, initial_accumulator=.1,
                            epsilon=1e-5)
        expected = parameter.value.copy()
        accumulated = np.full(2, .1)
        for gradient in (np.array([1., -2.]), np.array([-.3, 1.])):
            parameter.grad[...] = gradient
            accumulated += gradient * gradient
            expected -= .15 * gradient / (np.sqrt(accumulated) + 1e-5)
            optimizer.step()
        np.testing.assert_allclose(parameter.value, expected, atol=1e-12)
        loaded = self.parameter(parameter.value)
        replica = Adagrad([loaded], self.backend, initial_accumulator=.1, epsilon=1e-5)
        replica.load_state_dict(optimizer.state_dict())
        for opt, p in ((optimizer, parameter), (replica, loaded)):
            p.grad[...] = [2., .4]
            opt.step()
        np.testing.assert_array_equal(loaded.value, parameter.value)

    def test_weighted_microbatch_gradient_flushes_before_clipping(self):
        parameter = self.parameter([1., 1.])
        optimizer = SGD([parameter], self.backend, learning_rate=.1, max_norm=1.)
        accumulator = GradientAccumulator(optimizer)
        parameter.grad[...] = [4., 1.]
        accumulator.add(1)
        optimizer.zero_grad()
        parameter.grad[...] = [2., 3.]
        accumulator.add(3)
        accumulator.flush()
        expected_gradient = np.array([2.5, 2.5])
        np.testing.assert_allclose(parameter.grad, expected_gradient)
        result = optimizer.step()
        self.assertAlmostEqual(result["gradient_norm"], np.linalg.norm(expected_gradient))
        np.testing.assert_allclose(parameter.value, 1 - .1 * expected_gradient / np.linalg.norm(expected_gradient))
        with self.assertRaisesRegex(ValueError, "empty"):
            accumulator.flush()

    def test_nonfinite_microbatch_does_not_change_accumulated_values(self):
        parameter = self.parameter([0., 0.])
        optimizer = SGD([parameter], self.backend)
        accumulator = GradientAccumulator(optimizer)
        parameter.grad[...] = [1., 2.]
        accumulator.add(2)
        parameter.grad[...] = [9., np.nan]
        with self.assertRaises(FloatingPointError):
            accumulator.add(1)
        self.assertEqual(accumulator.total_weight, 2)
        accumulator.flush()
        np.testing.assert_array_equal(parameter.grad, [1., 2.])
