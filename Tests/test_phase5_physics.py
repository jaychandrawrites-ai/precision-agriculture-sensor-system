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

    def test_stratified_physical_uncertainty_keeps_exogenous_observations(self):
        from assumptions import Assumptions
        from physical_ensemble import ParameterRange, physical_ensemble, stratified_sample

        ref = ReferenceET(32.7, 0)
        timestamps = pd.date_range("2020-06-01T00:00Z", periods=36, freq="h")
        weather = pd.DataFrame({"air_temperature_c": 25. + 10*np.sin(np.arange(36) * np.pi / 12),
                                "relative_humidity_pct": 50., "shortwave_w_m2": 80.,
                                "wind_m_s": 2., "precipitation_mm": 0.}, index=timestamps)
        parameters = [ParameterRange("soil.crop_coefficient", .4, 1.2)]
        first = stratified_sample(parameters, 4, 42)
        np.testing.assert_array_equal(first, stratified_sample(parameters, 4, 42))
        scenarios = {"scenario_irrigation_mm": 0., "rain_interception_fraction": .05}
        bands, results = physical_ensemble(WeatherData(weather, pd.DataFrame(index=timestamps)),
                                           Assumptions(), ref, scenarios, parameters, draws=4)
        np.testing.assert_array_equal(bands["air_temperature_c_lower"], weather["air_temperature_c"])
        self.assertIsNone(results["features"]["air_temperature_c"]["spearman_correlation_of_mean"][
            "soil.crop_coefficient"])
        self.assertGreater(results["features"]["eta_mm_interval"]["spearman_correlation_of_mean"][
            "soil.crop_coefficient"], .8)

    def test_correlated_scenario_draws_preserve_marginal_strata(self):
        from physical_ensemble import ParameterRange, stratified_sample

        specs = [ParameterRange("soil.crop_coefficient", 0, 1),
                 ParameterRange("scenario.rain_interception_fraction", 0, 1)]
        values = stratified_sample(specs, 256, 19, latent_correlation=[[1, .9], [.9, 1]])
        self.assertGreater(np.corrcoef(values.T)[0, 1], .7)
        for column in values.T:
            np.testing.assert_array_equal(np.sort((column * 256).astype(int)), np.arange(256))
        with self.assertRaisesRegex(ValueError, "positive definite"):
            stratified_sample(specs, 4, 1, latent_correlation=[[1, 1], [1, 1]])
