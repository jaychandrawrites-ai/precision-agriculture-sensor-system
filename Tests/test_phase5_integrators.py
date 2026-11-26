import unittest

import numpy as np

from Numerical.Differentiation import hessian_vector
from Numerical.ODESolver import integrate_rk45


class IntegratorTests(unittest.TestCase):
    def test_rk45_accuracy_with_rejection_and_exact_endpoint(self):
        def derivative(time, state):
            return np.array([state[1], -state[0]])

        trajectory = integrate_rk45(derivative, [1, 0], 0, 7, step=7, rtol=1e-9, atol=1e-11)
        self.assertGreater(trajectory.rejected, 0)
        self.assertEqual(trajectory.times[-1], 7)
        np.testing.assert_allclose(trajectory.states[:, 0], np.cos(trajectory.times), atol=4e-9)
        np.testing.assert_allclose(trajectory.states[:, 1], -np.sin(trajectory.times), atol=4e-9)

    def test_projection_recomputes_endpoint_derivative(self):
        last = []
        def derivative(time, state):
            last.append((time, state.item()))
            return state * 0 + 1

        trajectory = integrate_rk45(derivative, [0], 0, 1, step=.25, max_step=.25,
                                    project=lambda state: np.minimum(state, .2))
        np.testing.assert_allclose(trajectory.states[1:], .2)
        self.assertIn((.25, .2), last)

    def test_hessian_vector_matches_quadratic(self):
        matrix = np.array([[5., 1.], [1., 3.]])
        point = np.array([-.7, 3.])
        direction = np.array([.01, -8.])
        np.testing.assert_allclose(hessian_vector(matrix.__matmul__, point, direction),
                                   matrix @ direction, atol=1e-8)
        np.testing.assert_array_equal(hessian_vector(matrix.__matmul__, point, np.zeros(2)), 0)


if __name__ == "__main__":
    unittest.main()
