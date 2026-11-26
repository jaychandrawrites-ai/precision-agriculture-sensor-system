"""Matched context search and reload tests on a synthetic weather series."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from config import EvolutionConfig, TrainingConfig
from Data.Dataset import build_dataset
from Data.RolloutWindows import evaluation_windows, rollout_windows
from Evolution.Genome import Genome
from Evolution.Resumable import SearchOptions
from Evolution.SequenceSearch import compare_sequences
from Framework.Backend import create_backend
from Temporal.Direct import fit_direct, make_direct_model
from Temporal.DirectCheckpoint import load_direct, save_direct
from Temporal.SequenceForecast import DENSE_FAMILIES, FAMILIES, SequenceSpec
from Utils.ModelLoader import load_forecaster
from forecast import forecast_csv
from main import synthetic_history


class SequenceEvolutionTests(unittest.TestCase):
    def fixture(self):
        frame = synthetic_history(200)
        return frame, frame.iloc[:, 1:].to_numpy(), frame.timestamp.to_numpy(dtype="datetime64[ns]")

    def test_all_families_train_evaluate_and_round_trip(self):
        _, values, times = self.fixture()
        dataset = build_dataset(values, times)
        settings = TrainingConfig(epochs=1, batch_size=64)
        training = rollout_windows(dataset, dataset.train, values, times, 2, context=6)
        validation = rollout_windows(dataset, dataset.validation, values, times, 2, context=6)
        for family in FAMILIES:
            with self.subTest(family=family), TemporaryDirectory() as directory:
                context = 3 if family in DENSE_FAMILIES else 6
                spec = SequenceSpec(family, 8, depth=2, context=context, heads=4)
                backend = create_backend()
                built = make_direct_model(backend, 2, 8, spec)
                self.assertEqual(sum(p.value.size for p in built.parameters()), spec.parameter_count(2))
                model, details = fit_direct(training, validation, dataset.target_scaler, backend,
                                            settings, width=8, architecture=spec)
                self.assertTrue(np.isfinite(details["validation_loss"]).all())
                path = Path(directory) / "forecast.json"
                save_direct(path, model, 8)
                restored = load_direct(path)
                self.assertEqual(restored.model.architecture, spec)
                query = validation.histories[:2]
                np.testing.assert_array_equal(restored.predict(query), model.predict(query))

    def test_context_repair_and_missing_window_eligibility(self):
        _, values, times = self.fixture()
        self.assertEqual(evaluation_windows(values, times, 3, context=24).starts[0], 24)
        values[15, :] = np.nan
        candidates = evaluation_windows(values, times, 3, context=24)
        self.assertEqual(candidates.starts[0], 40)
        generator = np.random.default_rng(19)
        for _ in range(25):
            genom = Genome.random(generator, sequence=True)
            repaired = genom.repair(900, horizon=6)
            self.assertLessEqual(repaired.decode().parameter_count(6), 900)
            self.assertEqual(repaired, genom.repair(900, horizon=6))

    def test_five_methods_share_windows_and_export_a_forecast_checkpoint(self):
        frame, values, times = self.fixture()
        training = TrainingConfig(seed=42, epochs=1, rollout_steps=2, batch_size=64)
        budget = EvolutionConfig(population=2, generations=1, candidate_epochs=1,
                                 candidate_parameter_step_limit=1_000_000)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "prepared.csv"
            frame.to_csv(source, index=False)
            report = compare_sequences(values, times, training, budget, SearchOptions(max_parameters=3000),
                                       export_directory=output)
            for method in ("random", "classical_ga", "quantum_inspired", "hybrid_amplitude"):
                row = report[method]
                self.assertEqual(row["actual_evaluations"], 2)
                self.assertLessEqual(row["actual_parameter_sample_steps"], 2_000_000)
                self.assertEqual(row["test_rollout"], row["test_rollout"][:2])
                checkpoint = output / row["saved_model"]
                inference = load_forecaster(checkpoint)
                self.assertEqual(inference.provenance["search_method"], method)
                csv = forecast_csv(source, checkpoint, 2)
                self.assertEqual(csv.shape, (2, 7))
                self.assertTrue(np.isfinite(csv.iloc[:, 1:].to_numpy()).all())
            self.assertEqual(report["protocol"]["context_hours_required_for_matched_windows"], 24)

    def test_configured_sequence_stacks_have_correct_gradients_and_saved_inference(self):
        from Framework.GradientCheck import check_model

        specs = (SequenceSpec("convolution", 8, depth=2, context=12, kernel=3, dilation=2),
                 SequenceSpec("attention", 8, depth=2, context=12, heads=4))
        frame, values, times = self.fixture()
        self.assertEqual(specs[0].receptive_field, 9)
        with TemporaryDirectory() as directory:
            source = Path(directory) / "prepared.csv"
            frame.to_csv(source, index=False)
            for spec in specs:
                with self.subTest(family=spec.family):
                    backend = create_backend(dtype="float64")
                    model = make_direct_model(backend, 2, spec.width, spec)
                    data = backend.normal((2, spec.context * 6))
                    output, _ = model.forward(data, training=False)
                    reports = check_model(model, data, backend.normal(output.shape), backend,
                                          samples=3, atol=2e-6, rtol=1e-4)
                    self.assertTrue(all(report.passed for report in reports), reports)
                    checkpoint = Path(directory) / f"{spec.family}.json"
                    from direct_forecast import train
                    train(source, checkpoint, TrainingConfig(epochs=1), 2, architecture=spec)
                    loaded = load_forecaster(checkpoint)
                    self.assertEqual(loaded.direct.model.architecture, spec)
                    np.testing.assert_allclose(forecast_csv(source, checkpoint, 2).iloc[:, 1:],
                                               loaded.forecast(values[-spec.context:], 2))
