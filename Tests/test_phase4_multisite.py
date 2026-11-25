import unittest

import numpy as np
import pandas as pd

from config import FEATURES
from multisite import aligned_snapshot


class MultisiteTests(unittest.TestCase):
    def test_snapshot_requires_common_origin_and_excludes_future_gps(self):
        index = pd.date_range("2024-01-01T08:00Z", periods=2, freq="h")
        forecast = pd.DataFrame(np.tile([25, 50, .3, .1, .08, 0], (2, 1)),
                                index=index, columns=FEATURES)
        def fixes(east):
            return pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01T06:45Z",
                                                               "2024-01-01T07:15Z"], utc=True),
                                 "received_at": pd.to_datetime(["2024-01-01T06:46Z",
                                                                 "2024-01-01T07:16Z"], utc=True),
                                 "latitude_deg": [0, 10], "longitude_deg": [east, east]})
        positions = {"a": fixes(0), "b": fixes(.0001)}
        snapshot, report = aligned_snapshot({"a": forecast, "b": forecast}, positions,
                                            feature=FEATURES[0], valid_at=index[-1])
        np.testing.assert_array_equal(snapshot.latitude_deg, [0, 0])
        self.assertEqual(report["lead_hours"], 2)
        self.assertTrue(snapshot.value_origin.eq("model_forecast").all())
        late = positions["b"].copy()
        late.loc[0, "received_at"] = late.loc[1, "received_at"]
        with self.assertRaisesRegex(ValueError, "fresh GPS"):
            aligned_snapshot({"a": forecast, "b": forecast}, {"a": positions["a"], "b": late},
                             feature=FEATURES[0], valid_at=index[-1])
        with self.assertRaisesRegex(ValueError, "common forecast origin"):
            shifted = forecast.iloc[1:]
            aligned_snapshot({"a": forecast, "b": shifted}, positions,
                             feature=FEATURES[0], valid_at=index[-1], max_gps_age_minutes=120)
