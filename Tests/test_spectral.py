import unittest

import numpy as np

from Temporal.Spectral import LossWeights, Modes, features, loss, sparse_events


class SpectralTests(unittest.TestCase):
    def test_modes_and_exact_anchor(self):
        hours = np.arange(73)
        modes = Modes((24, 12), True)
        coeff = np.array([8, 3, 2, -.5, 0, 1, 0])
        expected = (8 + 3*np.cos(2*np.pi*hours/24) - 2*np.sin(2*np.pi*hours/24)
                    - .5*np.cos(2*np.pi*hours/12) + np.cos(2*np.pi*hours/18)*np.exp(-hours/12))
        np.testing.assert_allclose(modes.basis(hours) @ coeff, expected)
        np.testing.assert_array_equal(modes.anchored_basis(hours)[0], np.zeros(modes.width))
        np.testing.assert_allclose(np.linalg.pinv(modes.anchored_basis(hours)) @ (
            modes.anchored_basis(hours) @ coeff), coeff, atol=1e-12)

    def test_gradient(self):
        rng = np.random.default_rng(42)
        basis = Modes((24, 12), True).anchored_basis(np.arange(1, 73))
        coeff = rng.normal(size=(2, 7, 6))
        target = rng.normal(size=(2, 72, 6))
        reference = np.linalg.pinv(basis) @ target

        def evaluate(coef):
            return loss(coef, target, 0, basis, reference, -.9, 1.1, LossWeights(level=.2, shape=.8, curvature=.3))

        _, gradient, _ = evaluate(coeff)
        for index in np.ndindex(coeff.shape):
            change = np.zeros_like(coeff)
            change[index] = 1e-6
            approx = (evaluate(coeff+change)[0] - evaluate(coeff-change)[0])/2e-6
            self.assertAlmostEqual(gradient[index], approx, places=8)

    def test_calendar_and_events(self):
        times = np.array(["2025-01-01", "2025-01-02"], dtype="datetime64[h]")
        values = features(np.zeros((2, 3, 6)), times)
        np.testing.assert_allclose(values[0, 18:20], values[1, 18:20])
        first, _ = sparse_events(np.random.default_rng(19), 1, (1000,), .01, .25)
        second, _ = sparse_events(np.random.default_rng(19), 1, (1000,), .01, .25)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(abs(first[0]), .01)
        self.assertTrue(np.all(abs(first) <= .25))
        self.assertLess(np.count_nonzero(first), 25)
        repeated = [sparse_events(np.random.default_rng(i), 10, (1,), .01, .25)[0][0]
                    for i in range(2000)]
        self.assertAlmostEqual(np.count_nonzero(repeated) / 2000, .1, delta=.025)
        np.testing.assert_allclose(np.abs(np.array(repeated)[np.nonzero(repeated)]), .1)

    def test_checkpoint_all_prefixes_and_required_origin(self):
        from dataclasses import asdict
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from config import FEATURES
        from Data.Normalization import Standardizer
        from Framework.Backend import create_backend
        from Temporal.Model import TemporalModel
        from Utils.ModelLoader import load_forecaster
        from Utils.Serialization import write_json

        backend = create_backend()
        modes = Modes()
        model = TemporalModel(backend, (4,), input_size=22, output_size=90)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            record = {"format": "jupiter-spectral-v1", "features": FEATURES,
                "interval_hours": 1, "modes": asdict(modes), "architecture": {"hidden": [4]},
                "dtype": "float32", "seed": 42, "parameters": model.state_dict(backend),
                "inputs": Standardizer().fit(np.zeros((3, 22))).state_dict(),
                "targets": Standardizer().fit(np.zeros((3, 6))).state_dict()}
            write_json(path, record)
            self.assertIsNone(load_forecaster(path).spectral.residual)
            residual = TemporalModel(backend, (8,), activations=("tanh",), input_size=22, output_size=432)
            record["residual"] = {"width": 8, "parameters": residual.state_dict(backend)}
            write_json(path, record)
            loader = load_forecaster(path)
            history = np.tile([25, 50, .4, .2, .1, .1], (3, 1))
            origin = np.datetime64("2025-01-01T16")
            full = loader.forecast(history, 72, origin=origin)
            self.assertEqual(full.shape, (72, 6))
            for hours in range(1, 73):
                np.testing.assert_allclose(loader.forecast(history, hours, origin=origin), full[:hours])
            with self.assertRaisesRegex(ValueError, "UTC origin"):
                loader.forecast(history, 72)
            with self.assertRaisesRegex(ValueError, "integer horizons"):
                loader.forecast(history, 73, origin=origin)

    def test_level_and_shape_separate_offset_from_curve_error(self):
        basis = Modes((24,)).basis(np.arange(1, 73))
        coefficients = np.zeros((1, 3, 6))
        coefficients[:, 0] = 2
        coefficients[:, 1] = 1
        target = np.zeros((1, 72, 6))
        _, _, terms = loss(coefficients, target, 0, basis, coefficients, -10, 10, LossWeights())
        self.assertAlmostEqual(terms["level"], 4)
        self.assertAlmostEqual(terms["shape"], .5)
        self.assertAlmostEqual(terms["value"], terms["level"] + terms["shape"])
        self.assertGreater(terms["curvature"], 0)

    def test_joint_residual_gradient(self):
        from Temporal.Spectral import residual_transform

        rng = np.random.default_rng(7)
        basis = Modes((24,), True).anchored_basis(np.arange(1, 73))
        coefficients = rng.normal(size=(1, 5, 6))
        output = rng.normal(size=(1, 432))
        target = rng.normal(size=(1, 72, 6))
        weights = LossWeights(level=.5, shape=1, curvature=.15)

        def objective(c, y):
            correction, derivative = residual_transform(y)
            score, spectral, _, residual = loss(c, target, 0, basis, c * 0, -.8, .8,
                                                 weights, correction=correction)
            return score, spectral, (residual * derivative).reshape(y.shape)

        _, spectral, residual = objective(coefficients, output)
        for values, expected, evaluate in (
            (coefficients, spectral, lambda x: objective(x, output)[0]),
            (output, residual, lambda x: objective(coefficients, x)[0]),
        ):
            for index in range(0, values.size, 11):
                offsets = np.zeros_like(values)
                offsets.flat[index] = 1e-6
                actual = (evaluate(values + offsets) - evaluate(values - offsets)) / 2e-6
                self.assertAlmostEqual(actual, expected.flat[index], places=8)
