import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from assumptions import Assumptions
from config import FEATURES, TrainingConfig
from Examples.SyntheticPhase3 import historical_fixture, session_fixture
from Physics.BalanceAudit import audit_crop_regime, audit_leaf_regime, audit_soil_regime, audited_leaf_step
from Physics.ForecastScenarios import load_site, scenario_forecasts
from Physics.Leaf.EnergyBalance import LeafWeather, approximate_longwave
from Physics.Leaf.LeafWetness import LeafState
from Physics.Soil.SoilWaterBalance import SoilBucket
from physical_ensemble import ParameterRange
from physical_forecast import run_experiment
from Pipeline import train_historical
from Utils.ModelLoader import load_forecaster
from Utils.Serialization import write_json


class PhysicalScenarioTests(unittest.TestCase):
    def test_historical_checkpoint_runs_on_joint_historical_and_field_scenarios(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, mapping = historical_fixture(root)
            model = root / "model.json"
            train_historical(source, mapping, model, TrainingConfig(hidden=(8,), epochs=1, batch_size=16))
            session, _, field = session_fixture(root, hours=46)
            forecaster = load_forecaster(model)
            issue_h = pd.Timestamp("2020-06-02T20:00Z")
            issue_f = pd.Timestamp("2020-07-02T21:00Z")
            ranges = [ParameterRange("soil.crop_coefficient", .5, 1.5),
                      ParameterRange("crop.critical_temperature_c", 20, 35)]
            historical = load_site(source, mapping, "historical", issue_h)
            predictions, history, report = scenario_forecasts(
                {"historical": historical}, forecaster, ranges, issue_h, steps=3, draws=4,
                latent_correlation=[[1., .7], [.7, 1.]]
            )
            self.assertEqual(predictions.shape, (4, 1, 3, 6))
            self.assertEqual(history.shape, (4, 1, 3, 6))
            self.assertTrue(report["sites"]["historical"]["baseline_checkpoint_physics_checked"])
            self.assertLess(np.ptp(history[:, 0, :, :2], axis=0).max(), 1e-12)
            self.assertGreater(np.ptp(history[:, 0, :, 4], axis=0).max(), 1e-4)
            self.assertTrue(report["sites"]["historical"]["local_derivatives_at_range_midpoints"])

            field_path = session / "readings.jsonl"
            second = session / "other-readings.jsonl"
            with field_path.open() as source_stream, second.open("w") as destination:
                for line in source_stream:
                    row = json.loads(line)
                    if "longitude_deg" in row["measurements"]:
                        row["measurements"]["longitude_deg"] = 0.00005
                    destination.write(json.dumps(row) + "\n")
            write_json(root / "field.json", field)
            write_json(root / "grid.json", {"east_m": [0, 3, 6], "north_m": [0, 3, 6],
                                            "origin_lat_lon": [0, 0], "features": list(FEATURES)})
            spec = {
                "checkpoint": "model.json", "issue_at": issue_f.isoformat(), "steps": 3,
                "draws": 4,
                "ranges": [{"path": "measurement.soil_vwc_bias", "lower": -.08, "upper": -.02},
                           {"path": "soil.field_capacity", "lower": .29, "upper": .32}],
                "sites": {"site-a": {"source": str(field_path), "configuration": "field.json", "kind": "field"},
                          "site-b": {"source": str(second), "configuration": "field.json", "kind": "field"}},
                "map": {"configuration": "grid.json", "leads": [1, 3]},
            }
            write_json(root / "spec.json", spec)
            result = run_experiment(root / "spec.json", root / "out")
            self.assertEqual(result["forecast_leads_hours"], [1, 2, 3])
            self.assertEqual(result["maps"]["3"]["draw_count"], 4)
            frame = pd.read_csv(root / "out" / "trajectories.csv")
            self.assertEqual(len(frame), 4 * 2 * 3)
            self.assertTrue((frame.groupby("draw_id").size() == 2 * 3).all())
            self.assertEqual(list(result["sites"]["site-a"]["history"]["rank_correlation_by_hour_feature"]),
                             ["measurement.soil_vwc_bias", "soil.field_capacity"])
            with np.load(root / "out" / "trajectories.npz", allow_pickle=False) as records:
                data = records["histories"]
                self.assertEqual(data.shape, (4, 2, 3, 6))
                self.assertLess(np.ptp(data[:, :, :, [0, 1, 2, 3]], axis=0).max(), 1e-12)
                self.assertGreater(np.ptp(data[:, :, :, 5], axis=0).max(), 1e-4)

    def test_integrated_water_and_energy_ledger_across_forcing_changes(self):
        assumptions = Assumptions()
        daytime = LeafWeather(32, 35, 600, approximate_longwave(32, 35), 1)
        night = LeafWeather(18, 100, 0, approximate_longwave(18, 100), .2)
        rain = LeafWeather(20, 90, 80, approximate_longwave(20, 90), .2, .003)
        _, cold = audited_leaf_step(LeafState(14, 0), night, assumptions.leaf, seconds=120)
        self.assertGreater(cold.condensation_kg_m2, 0)
        sunny, drying = audited_leaf_step(LeafState(25, .8), daytime, assumptions.leaf, seconds=120)
        self.assertGreater(drying.evaporation_kg_m2, 0)
        self.assertLess(sunny.wetness, .8)
        report = audit_leaf_regime(LeafState(23, .5), [daytime, night, rain], assumptions.leaf,
                                   seconds=600)
        self.assertLess(report["absolute_step_residual_max"]["water_kg_m2"], 1e-8)
        self.assertLess(report["absolute_step_residual_max"]["energy_j_m2"], 1e-6)
        self.assertGreater(report["balance_totals"]["rejected_water_kg_m2"], 0)
        self.assertLess(abs(report["fine_endpoint_minus_main"]["temperature_c"]), .01)
        soil = audit_soil_regime(SoilBucket(assumptions.soil, 89), [
            {"eto_mm": .5}, {"eto_mm": 1.2, "irrigation_mm": 120}, {"eto_mm": .1, "precipitation_mm": 6},
        ])
        self.assertTrue(soil["depletion_within_bounds"])
        self.assertGreater(soil["totals"]["drainage_mm"], 0)
        self.assertLess(soil["absolute_step_residual_max_mm"], 1e-9)
        accumulated = audit_crop_regime(0, [46, 47], assumptions.crop)
        cooled = audit_crop_regime(accumulated["final_index"], [21, 22], assumptions.crop)
        self.assertGreater(accumulated["final_index"], 0)
        self.assertLess(cooled["final_index"], accumulated["final_index"])
        self.assertLess(abs(cooled["residual_index"]), 1e-8)

    def test_observation_pairing_and_calibration_are_separate(self):
        from Physics.PairedValidation import ObservationContract, paired_validation

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model.csv"
            observed = root / "observed.csv"
            hours = pd.date_range("2020-01-01T00:00Z", periods=4, freq="h")
            pd.DataFrame({"timestamp": hours, "lwi": [.3, .3, .3, .3]}).to_csv(model, index=False)
            pd.DataFrame({"time": hours, "wetness": [40, 40, 80, np.nan]}).to_csv(observed, index=False)
            schema = ObservationContract("lwi", "percent", "time", "wetness",
                                         "synthetic", "synthetic test data")
            rows, report = paired_validation(model, observed, schema,
                                             validation_from="2020-01-01T02:00Z",
                                             calibrate_bias=True, synthetic=True)
            self.assertAlmostEqual(report["calibration"]["fitted_bias"], .1)
            self.assertAlmostEqual(report["validation"]["bias_corrected"]["rmse"], .4)
            self.assertEqual(len(rows), 1)
            self.assertEqual(report["alignment"]["unavailable_or_out_of_range_observation"], 1)
            with self.assertRaises(ValueError):
                ObservationContract("cotton_heat_stress", "index", "t", "x",
                                    "synthetic", "synthetic test data")
