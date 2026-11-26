import unittest

import numpy as np

from Framework.Backend import create_backend
from Framework.GradientCheck import check_model
from Framework.SequenceLayers import LSTM


class LSTMTests(unittest.TestCase):
    def test_parameter_and_input_adjoints(self):
        backend = create_backend(dtype="float64")
        layer = LSTM(backend, 2, 3)
        x = np.random.default_rng(42).normal(size=(2, 4, 2))
        dy = np.random.default_rng(51).normal(size=(2, 4, 3))
        reports = check_model(layer, x, dy, backend, samples=30)
        self.assertTrue(all(report.passed for report in reports))
        _, cache = layer.forward(x)
        actual = layer.backward(dy, cache)
        dx = np.zeros_like(x)
        for i in np.ndindex(x.shape):
            plus, minus = x.copy(), x.copy()
            plus[i] += 1e-5
            minus[i] -= 1e-5
            dx[i] = np.sum((layer.forward(plus)[0] - layer.forward(minus)[0]) * dy) / 2e-5
        np.testing.assert_allclose(actual, dx, atol=1e-8, rtol=1e-6)

    def test_future_changes_leave_prefix_unchanged(self):
        layer = LSTM(create_backend(dtype="float64"), 1, 2)
        original = np.zeros((1, 5, 1))
        altered = original.copy()
        altered[:, 3:] = 8
        np.testing.assert_array_equal(layer.forward(original)[0][:, :3],
                                      layer.forward(altered)[0][:, :3])
