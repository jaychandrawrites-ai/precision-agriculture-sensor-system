import unittest

import numpy as np

from Data.RollingOrigin import rolling_folds
from main import synthetic_history
from rolling_station_study import evaluate


class RollingStationTests(unittest.TestCase):
    def setUp(self):
        data = synthetic_history(1300)
        self.values = data.iloc[:, 1:].to_numpy()
        self.times = data.timestamp.to_numpy(dtype="datetime64[ns]")

    def test_boundaries_exclude_future_and_omit_gaps(self):
        positions = np.r_[0:500, 501:1300]
        folds = list(rolling_folds(self.values[positions], self.times[positions],
                                   initial_hours=400, test_hours=300, horizon=24))
        self.assertEqual(len(folds), 3)
        first = folds[0]
        self.assertEqual(first.train_rows.stop, 400)
        self.assertEqual(first.potential, 277)
        self.assertLess(len(first.starts), first.potential)
        self.assertGreaterEqual(first.starts.min(), 400)

    def test_altering_later_fold_never_changes_first_fold(self):
        options = dict(initial_hours=400, test_hours=300, horizon=24, max_training_hours=290)
        a = evaluate(self.values, self.times, **options)
        changed = self.values.copy()
        changed[700:, 0] += 1
        b = evaluate(changed, self.times, **options)
        self.assertEqual(a["folds"][0]["scores_vs_daily"], b["folds"][0]["scores_vs_daily"])
        self.assertEqual(a["folds"][0]["training_hours"], 290)
        self.assertEqual(a["pool"]["daily"]["origins"], 3 * 277)
        self.assertLess(a["pool"]["daily"]["rmse"][:, 0].max(), 1e-9)
