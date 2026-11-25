import unittest

import numpy as np

from Data.Dataset import build_dataset
from Data.Derivatives import lag_coefficients, latest_input
from Data.Normalization import Standardizer
from Data.RolloutWindows import rollout_windows
from Framework.Backend import create_backend
from main import synthetic_history
from Temporal.Baselines import baseline_forecasts
from Temporal.Model import TemporalModel
from Temporal.Rollout import Rollout, RolloutOptions


class RolloutTests(unittest.TestCase):
    def test_backward_slopes_are_causal_and_match_original_at_unit_weight(self):
        history = np.arange(18, dtype=float).reshape(3, 6) ** 2
        np.testing.assert_allclose((lag_coefficients() @ history).ravel(), latest_input(history))
        features = lag_coefficients(0.65) @ history
        expected = 0.65 * (history[2] - history[1]) + 0.35 * (history[1] - history[0])
        np.testing.assert_allclose(features[1], expected)

    def test_rollout_gradients_against_finite_difference(self):
        backend = create_backend(dtype="float64", seed=17)
        rng = np.random.default_rng(33)
        center = np.array([25, 50, 0.5, 0.4, 0.3, 0.5])
        histories = center + 0.03 * rng.normal(size=(3, 3, 6))
        targets = center + 0.08 * rng.normal(size=(3, 4, 6))
        output_scaler = Standardizer.from_state({
            "mean": center, "scale": [5, 10, 0.2, 0.2, 0.2, 0.5], "epsilon": 1e-8,
        })
        input_scaler = Standardizer.from_state({
            "mean": np.r_[center, np.zeros(12)], "scale": [0.6] * 18, "epsilon": 1e-8,
        })
        for output in ("state", "delta"):
            model = TemporalModel(backend, (5,))
            # Force bound and proportional-cap activity without evaluating a kink.
            model.layers[-1].bias.value[2:] = [4, 0, 2, -3]
            engine = Rollout(model, input_scaler, output_scaler, backend,
                             RolloutOptions(output), crop_coefficient=0.8)
            projected, raw, _ = engine.predict(histories, 4, True)
            self.assertTrue((raw[:, :, 2] > 1).any())
            self.assertTrue((raw[:, :, 4] > projected[:, :, 4]).any())
            for autoregressive in (True, False):
                with self.subTest(output=output, autoregressive=autoregressive):
                    model.zero_grad()
                    engine.objective(histories, targets, autoregressive=autoregressive, training=False)
                    for param in model.parameters():
                        for flat in rng.choice(param.value.size, min(12, param.value.size), replace=False):
                            index = np.unravel_index(flat, param.value.shape)
                            original = param.value[index]
                            scores = []
                            for direction in (1, -1):
                                param.value[index] = original + direction * 1e-6
                                scores.append(engine.objective(
                                    histories, targets, autoregressive=autoregressive,
                                    training=False, backward=False,
                                ))
                            param.value[index] = original
                            np.testing.assert_allclose(param.grad[index], (scores[0] - scores[1]) / 2e-6,
                                                       rtol=1e-6, atol=1e-7)

    def test_windows_preserve_targets_and_seasonal_baseline_never_reads_future(self):
        frame = synthetic_history()
        values = frame.iloc[:, 1:].to_numpy()
        timestamps = frame.timestamp.to_numpy(dtype="datetime64[ns]")
        dataset = build_dataset(values, timestamps)
        windows = rollout_windows(dataset, dataset.test, values, timestamps, 24)
        self.assertEqual(len(windows.starts), 13)
        np.testing.assert_array_equal(windows.targets[0], values[windows.starts[0]:windows.starts[0] + 24])
        windows.starts = windows.starts[:1]
        windows.targets = windows.targets[:1]
        windows.histories = windows.histories[:1]
        _, seasonal, fallback = baseline_forecasts(values, timestamps, windows, period_hours=5)
        future = values.copy()
        future[windows.starts[0]:] = np.nan
        _, with_no_future, no_future_fallback = baseline_forecasts(
            future, timestamps, windows, period_hours=5
        )
        np.testing.assert_array_equal(seasonal, with_no_future)
        np.testing.assert_array_equal(fallback, no_future_fallback)
        future[windows.starts[0] - 4] = np.nan
        persistence, seasonal, fallback = baseline_forecasts(future, timestamps, windows, period_hours=5)
        self.assertEqual(fallback[1], 1)
        np.testing.assert_array_equal(seasonal[:, 1], persistence[:, 1])

    def test_stable_checkpoint_preserves_forecast_mode(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from config import TrainingConfig
        from forecast import forecast_csv
        from train import train
        from Utils.Checkpoints import load

        with TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            checkpoint = Path(directory) / "model.json"
            frame = synthetic_history()
            frame.to_csv(source, index=False)
            settings = TrainingConfig(epochs=3, batch_size=32, dropout=0)
            report = train(source, checkpoint, settings, stable_rollout=True)
            self.assertEqual(report["rollout_windows"], {"train": 160, "validation": 31, "test": 31})
            model, backend, inputs, targets = load(checkpoint)
            self.assertEqual(model.rollout_options, RolloutOptions())
            projected, raw, diagnostics = Rollout(
                model, inputs, targets, backend, model.rollout_options
            ).predict(frame.iloc[-3:, 1:].to_numpy()[None], 6, True)
            actual = forecast_csv(source, checkpoint, 6).iloc[:, 1:].to_numpy()
            np.testing.assert_allclose(actual, projected[0], rtol=1e-6, atol=1e-7)
            self.assertEqual(len(diagnostics), 6)
            self.assertEqual(raw.shape, projected.shape)
