from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from config import TrainingConfig
from direct_forecast import predict, train
from main import synthetic_history
from Temporal.Direct import DIRECT_ARCHITECTURES, direct_features
from Temporal.DirectCheckpoint import load_direct
from Utils.Serialization import read_json


class SequenceForecastTests(unittest.TestCase):
    def test_checks_versioned_architectures_and_forecasts_without_historical_training_data(self):
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            history, session = directory / "history.csv", directory / "session.csv"
            series = synthetic_history(240)
            series.to_csv(history, index=False)
            series.tail(3).to_csv(session, index=False)
            settings = TrainingConfig(epochs=1, batch_size=64)
            checkpoints = {}
            for architecture in DIRECT_ARCHITECTURES:
                checkpoint = directory / f"{architecture}.json"
                checkpoints[architecture] = checkpoint
                report = train(history, checkpoint, settings, 3, 4, architecture=architecture)
                self.assertEqual(report["architecture"], architecture)
                self.assertTrue(np.isfinite(report["validation_loss"]).all())
                expected = "jupiter-direct-v1" if architecture == "gated" else "jupiter-direct-v2"
                self.assertEqual(read_json(checkpoint)["format"], expected)
            history.unlink()
            for architecture, checkpoint in checkpoints.items():
                prediction = predict(session, checkpoint)
                self.assertEqual(prediction.shape, (3, 6))
                self.assertTrue(np.isfinite(prediction.to_numpy()).all())
                self.assertEqual(load_direct(checkpoint).model.architecture, architecture)

    def test_sequence_features_keep_observations_in_chronological_order(self):
        history = np.arange(18).reshape(1, 3, 6)
        np.testing.assert_array_equal(direct_features(history, "attention"), np.arange(18)[None])
        self.assertFalse(np.allclose(direct_features(history, "gated"), np.arange(18)[None]))
