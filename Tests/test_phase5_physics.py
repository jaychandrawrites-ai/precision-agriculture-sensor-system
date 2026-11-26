import unittest

import numpy as np
import pandas as pd

from Data.AZMET import WeatherData
from Data.WeatherPhysics import ReferenceET
from Physics.Evapotranspiration.ETComparison import RadiationAssumptions, compare_et, hourly_extraterrestrial
from Physics.Evapotranspiration.FAO56 import (actual_pressure, atmospheric_pressure,
                                             net_radiation_hourly, reference_et_hourly)
from Physics.Evapotranspiration.SolarRadiation import extraterrestrial_daily


class HourlyETTests(unittest.TestCase):
    def test_fao56_hourly_worked_example(self):
        pressure = atmospheric_pressure(8.)
        vapor = actual_pressure(38., 52.)
        net = net_radiation_hourly(38., 2.45, 2.658, vapor)
        self.assertAlmostEqual(net, 1.749, places=2)
        self.assertAlmostEqual(reference_et_hourly(38., net, .1 * net, 3.3, vapor, pressure),
                               .63, places=2)
        vapor = actual_pressure(28., 90.)
        net = net_radiation_hourly(28., 0, 0, vapor, dark_cloud_ratio=.8)
        self.assertAlmostEqual(net, -.1, places=2)
        self.assertAlmostEqual(reference_et_hourly(28., net, .5 * net, 1.9, vapor, pressure),
                               0., places=2)
        with self.assertRaises(ValueError):
            net_radiation_hourly(25, 0, 0, 1)

    def test_solar_interval_units_and_missing_observations(self):
        start = pd.Timestamp("2020-06-03T00:00Z")
        ref = ReferenceET(32., 0)
        total = sum(hourly_extraterrestrial(start + pd.offsets.Hour(i), ref) for i in range(1, 25))
        self.assertAlmostEqual(total, extraterrestrial_daily(32., start.dayofyear), places=10)
        utc = pd.date_range(start, periods=50, freq="h")
        weather = pd.DataFrame({"air_temperature_c": 24 + np.sin(np.arange(50) * np.pi / 12),
                                "relative_humidity_pct": 45., "wind_m_s": 3., "shortwave_w_m2": 50.},
                               index=utc)
        weather.iloc[-1, weather.columns.get_loc("wind_m_s")] = np.nan
        output, report = compare_et(WeatherData(weather, pd.DataFrame(index=utc)),
                                    ref, RadiationAssumptions(0., 2., .8))
        self.assertTrue(np.isnan(output.iloc[-1, 1]))
        self.assertEqual(report["matched_hours"], 26)
        self.assertEqual(report["missing_hours"], 24)
