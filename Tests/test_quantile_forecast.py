import tempfile
import unittest
from pathlib import Path

import numpy as np

from Framework.Backend import Backend, BackendConfig
from Temporal.Quantile import QuantileForecaster, quantile_scores


class QuantileTests(unittest.TestCase):
    def test_pinball_gradient_ordered_forecast_and_checkpoint(self):
        rng = np.random.default_rng(104)
        model = QuantileForecaster(Backend(BackendConfig(dtype="float64")), np.zeros(6),
                                   np.ones(6), 3, width=6)
        x = rng.normal(size=(6, 3, 6))
        y = rng.normal(size=(6, 3, 6))
        initial = model.objective(x, y, gradients=True)
        for parameter in model.network.parameters():
            index = (0,) * parameter.value.ndim
            old = parameter.value[index]
            h = 1e-5
            parameter.value[index] = old + h
            plus = model.objective(x, y)
            parameter.value[index] = old - h
            minus = model.objective(x, y)
            parameter.value[index] = old
            self.assertAlmostEqual(float(parameter.grad[index]), (plus - minus) / (2*h), places=7)
        for _ in range(40):
            model.step(x, y)
        self.assertLess(model.objective(x, y), initial)
        q = model.predict(x)
        self.assertTrue((np.diff(q, axis=-1) >= 0).all())
        self.assertEqual(quantile_scores(q, y, model.levels)["coverage"].shape, (3, 6))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "quantile.json"
            model.save(path)
            loaded = QuantileForecaster.load(path)
            np.testing.assert_array_equal(loaded.predict(x), q)
            with self.assertRaisesRegex(ValueError, "outside"):
                loaded.predict(x, 4)
