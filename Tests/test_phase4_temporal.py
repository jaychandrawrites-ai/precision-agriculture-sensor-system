import unittest

import numpy as np

from config import TrainingConfig
from forecast_benchmark import compare
from Framework.Backend import create_backend
from Framework.GradientCheck import check_model
from main import synthetic_history
from Temporal.Direct import DirectModel
from Temporal.ForecastUncertainty import calibrate, evaluate_intervals


class DirectTests(unittest.TestCase):
    def test_gated_and_derivative_manual_gradient(self):
        backend = create_backend(dtype="float64")
        model = DirectModel(backend, 2, width=4)
        x = backend.normal((4, 18))
        upstream = backend.normal((4, 12))
        self.assertTrue(all(report.passed for report in check_model(model, x, upstream, backend)))

    def test_interval_order_statistic(self):
        actual = np.zeros((19, 2, 6))
        errors = np.broadcast_to(np.arange(19)[:, None, None], actual.shape)
        intervals = calibrate(errors, actual, .9)
        np.testing.assert_array_equal(intervals.radius, 17)
        np.testing.assert_array_equal(evaluate_intervals(intervals, actual, actual)["coverage"], 1.)
        with self.assertRaises(ValueError):
            calibrate(errors[:2], actual[:2], .9)

    def test_shared_windows_benchmark(self):
        frame = synthetic_history(480)
        values = frame.iloc[:, 1:].to_numpy()
        times = frame.timestamp.to_numpy(dtype="datetime64[ns]")
        report = compare(values, times, TrainingConfig(epochs=2, hidden=(8,), dropout=0),
                         horizon=12, seeds=[5, 6])
        for mode in ("direct", "recursive", "hybrid"):
            self.assertEqual(len(report["metrics"][mode]), 12)
            self.assertEqual(report["intervals"][mode]["coverage"].shape, (12, 6))
            self.assertEqual(report["intervals"][mode]["calibration_origins"], report["windows"]["calibration"])

    def test_direct_checkpoint_round_trip(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from direct_forecast import predict, train

        with TemporaryDirectory() as temp:
            path = Path(temp)
            synthetic_history().to_csv(path / "input.csv", index=False)
            train(path / "input.csv", path / "model.json", TrainingConfig(epochs=2),
                  horizon=8, width=4)
            first = predict(path / "input.csv", path / "model.json")
            second = predict(path / "input.csv", path / "model.json")
            self.assertEqual(first.shape, (8, 6))
            np.testing.assert_array_equal(first, second)

    def test_reliability_threshold_uses_calibration_only(self):
        from Temporal.ForecastUncertainty import reliability

        truth = np.zeros((3, 2, 6))
        result = reliability([1., 2., 3.], [2., 3., 100.], truth + 1, truth)
        self.assertAlmostEqual(result["threshold"], 2.9)
        self.assertEqual(result["groups"]["flagged"]["origins"], 2)
        np.testing.assert_array_equal(result["groups"]["within_reference"]["rmse"], 1)
