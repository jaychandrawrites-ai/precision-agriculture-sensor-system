import unittest

import numpy as np

from Numerical.Differentiation import hessian, jacobian
from Numerical.ODESolver import integrate_implicit
from Numerical.Optimization import lbfgs


class NumericalTests(unittest.TestCase):
    def test_derivatives(self):
        x = np.array([1.3, -0.5])
        np.testing.assert_allclose(jacobian(lambda p: [p[0] * p[1], np.sin(p[0])], x),
                                   [[x[1], x[0]], [np.cos(x[0]), 0]], atol=1e-9)
        matrix = np.array([[3., 1.], [1., 9.]])
        np.testing.assert_allclose(hessian(lambda p: matrix @ p, x), matrix, atol=1e-8)

    def test_implicit_stiff_decay(self):
        decay = np.array([1., 800.])
        result = integrate_implicit(lambda t, y: -decay * y, [1, 1], 0, 1,
                                    jac=lambda t, y: -np.diag(decay), step=1)
        self.assertGreater(result.rejected, 0)
        self.assertEqual(result.times[-1], 1)
        self.assertAlmostEqual(result.states[-1, 0], np.exp(-1), delta=.005)
        self.assertLess(abs(result.states[-1, 1]), 1e-7)

    def test_lbfgs_minimum(self):
        diagonal = np.geomspace(1, 100, 6)
        result = lbfgs(lambda x: (.5 * x @ (diagonal * x), diagonal * x), np.ones(6))
        self.assertEqual(result.status, "converged")
        self.assertTrue((np.diff(result.loss_trace) <= 0).all())
        np.testing.assert_allclose(result.point, 0, atol=1e-6)
