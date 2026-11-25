import unittest

import numpy as np
import pandas as pd

from Data.StationAnalysis import domain_shift, seasonal_profile
from Numerical.ODESolver import integrate


class StationTests(unittest.TestCase):
    def test_profile_preserves_missing_and_local_month(self):
        times = pd.DatetimeIndex(["2024-02-01T01:00Z", "2024-02-01T12:00Z"])
        frame = pd.DataFrame({"air_temperature_c": [38, np.nan]}, index=times)
        results = seasonal_profile(frame)["months"]
        self.assertEqual(results["1"]["air_temperature_hours_above_threshold"], 1)
        self.assertEqual(results["2"]["features"]["air_temperature_c"]["missing"], 1)
        self.assertIsNone(domain_shift(frame, frame)["air_temperature_c"]["mean_difference_reference_sd"])

    def test_adaptive_integrator_respects_stage_step_bound(self):
        result = integrate(lambda t, y: -y, [0.], 0., 2.0, step=0.1, max_step=.2)
        self.assertLessEqual(np.diff(result.times).max(), .2 + 1e-12)
        self.assertEqual(result.times[-1], 2)
