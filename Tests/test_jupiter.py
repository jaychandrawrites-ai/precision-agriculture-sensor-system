from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from Acquisition.AcquisitionController import capture
from Acquisition.DataLogger import replay
from config import EvolutionConfig, SpatialConfig, TrainingConfig
from Data.Dataset import build_dataset
from Data.Derivatives import latest_input
from Data.Resampling import hourly_mean
from Evolution.Fitness import validation_fitness
from Evolution.GeneticAlgorithm import search as genetic_search
from Evolution.Genome import Genome
from forecast import forecast_csv
from Framework.Backend import create_backend
from Framework.GradientCheck import check_model
from main import synthetic_history
from QuantumInspired.QuantumEvolution import search as amplitude_search
from QuantumInspired.QuantumGenome import QuantumGenome
from Spatial.SpatialGrid import SpatialGrid
from Spatial.SpatialModel import reconstruct
from Temporal.Model import TemporalModel
from train import train


def history():
    frame = synthetic_history()
    return frame.iloc[:, 1:].to_numpy(), frame.timestamp.to_numpy(dtype="datetime64[ns]")


class DataTests(unittest.TestCase):
    def test_split_scaling_and_derivatives(self):
        values, timestamps = history()
        dataset = build_dataset(values, timestamps)
        idx = dataset.target_indices[dataset.train]
        expected = np.stack([latest_input(values[i-3:i]) for i in idx])
        np.testing.assert_allclose(dataset.input_scaler.mean, expected.mean(axis=0))
        np.testing.assert_allclose(dataset.target_scaler.mean, values[idx].mean(axis=0))
        np.testing.assert_allclose(dataset.targets, values[dataset.target_indices])
        self.assertLess(dataset.train[-1], dataset.validation[0])
        self.assertLess(dataset.validation[-1], dataset.test[0])

    def test_gaps_and_imputed_targets(self):
        values, timestamps = history()
        missing_value, missing_hour = 10, 85
        values[missing_value] = np.nan
        values = np.delete(values, missing_hour, axis=0)
        timestamps = np.delete(timestamps, missing_hour)
        dataset = build_dataset(values, timestamps, max_gap=1)
        self.assertNotIn(missing_value, dataset.target_indices)
        for index in range(missing_value + 1, missing_value + 4):
            row = np.flatnonzero(dataset.target_indices == index)[0]
            self.assertEqual(dataset.weights[row], 0.75)
        for index in range(missing_hour, missing_hour + 3):
            self.assertNotIn(index, dataset.target_indices)

    def test_resampling_is_right_closed(self):
        frame = pd.DataFrame({"t": ["2020-01-01 08:00", "2020-01-01 08:01"], "v": [1, 100]})
        means, counts = hourly_mean(frame, "t", ["v"])
        self.assertEqual(means.index[0].isoformat(), "2020-01-01T15:00:00+00:00")
        np.testing.assert_array_equal(means.v, [1, 100])
        np.testing.assert_array_equal(counts.v, [1, 1])


class TemporalTests(unittest.TestCase):
    def test_manual_residual_gradients(self):
        backend = create_backend(dtype="float64")
        model = TemporalModel(backend, (8, 8), skips=(0, 1))
        reports = check_model(model, backend.normal((3, 18)), backend.normal((3, 6)), backend)
        self.assertTrue(all(report.passed for report in reports), reports)

    def test_checkpoint_and_forecast(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            frame = synthetic_history()
            frame.to_csv(path / "input.csv", index=False)
            result = train(path / "input.csv", path / "model.json",
                           TrainingConfig(epochs=20, batch_size=32, dropout=0))
            self.assertGreater(result["test"]["air_temperature_c"]["persistence_skill"], 0)
            for _ in range(2):
                result = forecast_csv(path / "input.csv", path / "model.json", 6)
                self.assertEqual(result.shape, (6, 7))
                self.assertTrue(np.isfinite(result.iloc[:, 1:]).all().all())
                self.assertEqual(result.timestamp.iloc[0], frame.timestamp.iloc[-1] + pd.Timedelta(hours=1))
            frame.drop(index=frame.index[-2]).to_csv(path / "input.csv", index=False)
            with self.assertRaises(ValueError):
                forecast_csv(path / "input.csv", path / "model.json", 6)


class AcquisitionTests(unittest.TestCase):
    def test_raw_replay_and_partial_invalid_readings(self):
        lines = [b"boot\n", b"--- AIR ---\n", b"Temperature: 22.2 C\n",
                 b"Temperature: 23 C\n", b"Humidity: 101 %RH\n", b"--- SOIL ---\n",
                 b"Moisture: 23.1 %\n", b"Temperature: 18.3 C\n",
                 b"EC: 42 uS/cm\n", b"pH: 7.1\n", b"===\n",
                 b"--- AIR ---\n", b"ERROR: SHT45 read failed.\n", b"\xff\n"]
        timestamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
        with TemporaryDirectory() as directory:
            path = Path(directory)
            capture(((timestamp, line) for line in lines), path / "capture")
            raw = path / "capture/raw.jsonl"
            self.assertEqual(list(replay(raw)), [(timestamp, line) for line in lines])
            capture(replay(raw), path / "replayed")
            readings = (path / "capture/readings.jsonl").read_text()
            self.assertEqual(readings, (path / "replayed/readings.jsonl").read_text())
            first, partial = map(json.loads, readings.splitlines())
            self.assertEqual(first["measurements"]["air_temperature_c"], 22.2)
            self.assertEqual(first["measurements"]["soil_ph"], 7.1)
            self.assertIn("invalid:relative_humidity_pct", first["flags"])
            self.assertIn("duplicate:air_temperature_c", first["flags"])
            self.assertIn("sensor_error:air", partial["flags"])


class SearchAndSpatialTests(unittest.TestCase):
    def test_search_budgets_and_normalization(self):
        settings = EvolutionConfig(population=3, generations=2, candidate_epochs=1)
        dataset = build_dataset(*history())
        score = validation_fitness(dataset, TrainingConfig(), settings)
        for search in (genetic_search, amplitude_search):
            best, generations = search(score, settings)
            self.assertEqual([len(generation) for generation in generations], [3, 3])
            self.assertTrue(np.isfinite(best.score))
        genome = QuantumGenome()
        genome.rotate(Genome((0,) * 10), Genome((1,) * 10), 0.1)
        np.testing.assert_allclose((genome.amplitudes ** 2).sum(axis=1), 1)

    def test_spatial_seed_fidelity(self):
        grid = SpatialGrid(np.arange(7), np.arange(5))
        result = reconstruct(grid, [[0, 0], [6, 4]], [2, 10], create_backend(),
                             SpatialConfig(iterations=200), hard_seeds=True)
        self.assertEqual(result.field.shape, (5, 7))
        self.assertEqual((result.field[0, 0], result.field[4, 6]), (2, 10))
        self.assertLess(result.energies[-1], result.energies[0])
