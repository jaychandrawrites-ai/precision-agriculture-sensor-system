from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from assumptions import Assumptions
from azmet import prepare
from Data.AZMET import AZMETSchema, inspect_schema, read_azmet
from Data.Dataset import build_dataset
from Data.WeatherPhysics import CausalReferenceET, ReferenceET
from Examples.SyntheticPhase3 import historical_fixture
from Physics.Evapotranspiration.Hargreaves import reference_et_daily
from Physics.Evapotranspiration.SolarRadiation import extraterrestrial_daily, extraterrestrial_interval
from Physics.Leaf.EnergyBalance import LeafWeather, approximate_longwave
from Physics.Leaf.LeafWetness import LeafState, advance_leaf
from Physics.Soil.SoilWaterBalance import SoilBucket


class WeatherTests(unittest.TestCase):
    def test_schema_units_missing_values_and_provenance(self):
        with TemporaryDirectory() as tmp:
            source, config = historical_fixture(Path(tmp), 10)
            rows = pd.read_csv(source)
            rows.loc[0, ["temp_f", "humidity_fraction"]] = [77, 0.65]
            rows.loc[1, "temp_f"] = np.nan
            rows.loc[1, "rain_in"] = np.nan
            rows.loc[4, "humidity_fraction"] = 2
            rows.drop(index=2).to_csv(source, index=False)
            schema = AZMETSchema.from_dict(config["schema"])
            record = read_azmet(source, schema, source_label=config["source_label"], synthetic=True)
            self.assertEqual(inspect_schema(source)["row_count"], 9)
            self.assertEqual(record.provenance["source_sha256"], inspect_schema(source)["sha256"])
            self.assertEqual(record.provenance["source_type"], "synthetic_fixture")
            self.assertAlmostEqual(record.values.air_temperature_c.iloc[0], 25)
            self.assertEqual(record.values.relative_humidity_pct.iloc[0], 65)
            self.assertTrue(record.flags["filled:air_temperature_c"].iloc[1])
            self.assertTrue(np.isnan(record.values.precipitation_mm.iloc[1]))
            self.assertTrue(record.flags.missing_hour.iloc[2])
            self.assertTrue(np.isnan(record.values.air_temperature_c.iloc[2]))
            self.assertTrue(record.flags["invalid:relative_humidity_pct"].iloc[4])
            self.assertTrue(record.flags["filled:relative_humidity_pct"].iloc[4])

    def test_headerless_one_based_hour_and_leap_year(self):
        with TemporaryDirectory() as tmp:
            source, config = historical_fixture(Path(tmp), 2)
            names = ["year", "day", "hour", *pd.read_csv(source).columns[1:]]
            source.write_text("2020,366,24,77,0.6,0,1,0.1\n2021,1,1,78,0.65,0,1,0\n")
            mapping = {**config["schema"], "timestamp": None, "timezone": "America/Phoenix",
                       "calendar": {"year": "year", "day_of_year": "day", "hour": "hour",
                                    "hour_encoding": "1-24"}, "headerless_columns": names}
            result = read_azmet(source, AZMETSchema.from_dict(mapping), source_label="synthetic")
            self.assertEqual(result.values.index[0].isoformat(), "2021-01-01T07:00:00+00:00")
            self.assertEqual(result.values.index[1].isoformat(), "2021-01-01T08:00:00+00:00")
            self.assertAlmostEqual(result.values.precipitation_mm.iloc[0], 2.54)
            inspected = inspect_schema(source, headerless=True)
            self.assertEqual(inspected["row_count"], 2)
            self.assertEqual(len(inspected["columns"]), len(names))
            source.write_text("2021,366,24,77,0.6,0,1,0.1\n")
            with self.assertRaisesRegex(ValueError, "outside its declared range"):
                read_azmet(source, AZMETSchema.from_dict(mapping), source_label="synthetic")

    def test_historical_features_are_causal_and_split_statistics_remain_training_only(self):
        with TemporaryDirectory() as tmp:
            source, config = historical_fixture(Path(tmp))
            full, flags, report = prepare(source, config)
            self.assertEqual(report["provenance"]["feature_origin"]["lwi"], "simulated_leaf_ode")
            self.assertEqual(report["quality"]["flags"]["reference_et_unavailable"], 23)
            self.assertTrue(flags.soil_restart.iloc[23])
            timestamps = full.index.to_numpy(dtype="datetime64[ns]")
            dataset = build_dataset(full.to_numpy(), timestamps)
            training = dataset.targets[dataset.train]
            np.testing.assert_allclose(dataset.target_scaler.mean, training.mean(axis=0))
            np.testing.assert_array_less(full.eta_mm_interval.dropna(),
                                        full.eto_mm_interval.dropna() + 1e-9)
            raw = pd.read_csv(source)
            raw.iloc[:50].to_csv(source, index=False)
            prefix, _, _ = prepare(source, config)
            np.testing.assert_allclose(full.iloc[:50], prefix, equal_nan=True)

    def test_gap_restarts_temperature_history_and_solar_units_conserve_daily_total(self):
        config = ReferenceET(32.7, -412.5)
        start = pd.Timestamp("2020-06-02T00:00Z")
        reference = CausalReferenceET(config)
        for i in range(23):
            self.assertTrue(np.isnan(reference.advance(start + timedelta(hours=i), 20 + i % 12)))
        self.assertGreaterEqual(reference.advance(start + timedelta(hours=23), 32), 0)
        self.assertTrue(np.isnan(reference.advance(start + timedelta(hours=25), 32)))
        daily = extraterrestrial_daily(32.7, 155)
        np.testing.assert_allclose(daily, sum(extraterrestrial_interval(32.7, 155, hour, hour + 1)
                                              for hour in range(24)), atol=1e-10)
        self.assertAlmostEqual(reference_et_daily(25, 15, 35, 40),
                               0.0023 * 42.8 * np.sqrt(20) * 0.408 * 40)

    def test_soil_water_conservation_and_bounded_leaf_state(self):
        assumptions = Assumptions()
        bucket = SoilBucket(assumptions.soil)
        for rain in (100, 0, 0, 0):
            step = bucket.step(eto_mm=1.3, precipitation_mm=rain)
            self.assertAlmostEqual(step.residual_mm, 0, places=10)
            self.assertGreaterEqual(step.final_depletion_mm, 0)
            self.assertLessEqual(step.actual_et_mm, 1.3)
        weather = LeafWeather(25, 85, 0, approximate_longwave(25, 85), 1, 0.0001)
        state, ode = advance_leaf(LeafState(25, 0), weather, assumptions.leaf, 600)
        self.assertTrue(0 < state.wetness <= 1)
        self.assertTrue(np.isfinite(ode.states).all())
