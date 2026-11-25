import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from config import TrainingConfig
from main import synthetic_history
from seasonal_study import compare
from Temporal.BlockIntervals import BlockIntervals, separated_indices
from Temporal.Seasonality import ConvexCorrection, HarmonicRegression, SeasonalLookup
from Temporal.SeasonalCheckpoint import forecast_saved


class SeasonalTests(unittest.TestCase):
    def setUp(self):
        frame = synthetic_history(960)
        self.values = frame.iloc[:, 1:].to_numpy()
        self.times = frame.timestamp.to_numpy(dtype="datetime64[ns]")

    def test_seasonal_history_lookup_never_uses_future_and_falls_back_on_gaps(self):
        origin = 200
        history = self.values[None, origin-2:origin+1]
        past = SeasonalLookup(self.values[:origin+1], self.times[:origin+1])
        whole = SeasonalLookup(self.values, self.times)
        for period in (24, 168):
            reference, missing = past.predict(self.times[origin:origin+1], history, 72, period)
            actual, flags = whole.predict(self.times[origin:origin+1], history, 72, period)
            self.assertFalse(missing.any())
            np.testing.assert_array_equal(actual, reference)
            np.testing.assert_array_equal(flags, missing)
        latest, missing = SeasonalLookup(history[0], self.times[origin-2:origin+1]).predict(
            self.times[origin:origin+1], history, 5
        )
        self.assertTrue(missing.all())
        np.testing.assert_array_equal(latest[0, 0], history[0, -1])

    def test_harmonic_and_blend_formulas(self):
        model = HarmonicRegression.fit(self.values[:480], self.times[:480], periods=((24, 1),),
                                       ridge=1e-10)
        np.testing.assert_allclose(model.predict(self.times[480:500])[:, 0], self.values[480:500, 0],
                                   atol=1e-6)
        anchor = np.broadcast_to(self.values[:1, None, :], (3, 2, 6))
        candidate = anchor + .01
        target = anchor + .005
        trained = ConvexCorrection.fit(anchor, candidate, target, shrinkage=0)
        np.testing.assert_allclose(trained.weight, .5)

    def test_spaced_marginal_and_trajectory_coverage(self):
        origins = self.times[:100]
        self.assertEqual(separated_indices(origins, 6).tolist(), list(range(0, 100, 6)))
        targets = np.zeros((100, 6, 6))
        errors = np.ones_like(targets)
        errors[:, -1] = 3
        intervals = BlockIntervals.fit(errors, targets, origins, .9)
        np.testing.assert_array_equal(intervals.trajectory_radius, 3)
        result = intervals.score(errors, targets, origins)
        self.assertEqual(result["verification_origins"], 17)
        np.testing.assert_array_equal(result["marginal_coverage"], 1)
        self.assertEqual(result["mean_interval_score"][0][0], 2)

    def test_independent_calibration_study_and_saved_inference(self):
        settings = TrainingConfig(epochs=2, hidden=(8,), batch_size=128, dropout=0)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "saved"
            result = compare(self.values, self.times, self.times[[400, 550, 750]], settings,
                             [42, 43], horizon=12, width=8, model_output=directory)
            final = forecast_saved(directory / "seasonal-ensemble.json", self.values[-3:], self.times[-3:])
        self.assertEqual(result["horizon"], 12)
        self.assertGreater(result["uncertainty"]["validation_blend"]["calibration_origins"], 10)
        self.assertEqual(final["point"].shape, (12, 6))
        self.assertTrue(final["used_current_state"].all())
        self.assertTrue((final["lower"] <= final["point"]).all())
