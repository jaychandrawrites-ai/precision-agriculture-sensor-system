from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from config import TrainingConfig
from Data.Normalization import Standardizer
from Framework.Backend import create_backend
from main import synthetic_history
from reliability_forecast import assess_data, evaluate_policy, fit_profile
from Temporal.BlockIntervals import BlockIntervals
from Temporal.Direct import DirectForecast, make_direct_model
from Temporal.DirectCheckpoint import save_direct
from Temporal.HorizonVerification import InferenceArtifact, VerificationEnsemble, verify_leads
from Temporal.Quantile import QuantileForecaster
from Temporal.ReliabilityGate import ReliabilityProfile, ensemble_disagreement
from Temporal.SeasonalCheckpoint import save_ensemble, save_member
from Temporal.Seasonality import ConvexCorrection
from Temporal.SequenceForecast import SequenceSpec
from train import train


def _direct(backend, horizon, architecture="gated"):
    model = make_direct_model(backend, horizon, 4, architecture)
    size = 6 * (architecture.context if isinstance(architecture, SequenceSpec) else 3)
    inputs = Standardizer().fit(np.vstack((np.zeros(size), np.ones(size))))
    targets = Standardizer().fit(np.vstack((np.zeros(6), np.ones(6))))
    return DirectForecast(model, backend, inputs, targets)


class ForecastReliabilityTests(unittest.TestCase):
    def setUp(self):
        data = synthetic_history(1500)
        self.values = data.iloc[:, 1:].to_numpy()
        self.times = data.timestamp.to_numpy(dtype="datetime64[ns]")

    def test_calibration_only_context_and_joint_shift_abstention(self):
        profile = ReliabilityProfile.fit(self.values[:600], self.times[:600],
                                         station_ids=("test-station-a",), calibration_id="fixture",
                                         source_kind="historical", supported_steps=6,
                                         disagreements=np.full(25, .10))
        profile = ReliabilityProfile.from_record(profile.to_record())
        shifted = self.values[-3:].copy()
        shifted[-1, 0] = 60
        result = profile.assess(shifted, self.times[-3:] + np.timedelta64(180, "D"),
                                station_id="test-station-b", source_kind="field",
                                calibration_id="different", steps=6, disagreement=.20)
        self.assertEqual(result["decision"], "abstain")
        self.assertEqual({x["code"] for x in result["issues"]},
                         {"station_transfer", "domain_transfer", "calibration_mismatch", "seasonal_transfer",
                          "extreme_temperature", "distribution_shift", "ensemble_disagreement"})
        clean = profile.assess(self.values[-3:], self.times[-3:], steps=2, station_id="test-station-a",
                               calibration_id="fixture", disagreement=.20)
        self.assertIn("uncalibrated_ensemble", {item["code"] for item in clean["issues"]})

    def test_calibration_profile_bound_to_models_and_past_origins(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            frame = pd.DataFrame(self.values[:600], columns=synthetic_history(3).columns[1:])
            frame.insert(0, "timestamp", pd.DatetimeIndex(self.times[:600], tz="UTC"))
            frame.to_csv(path / "calibration.csv", index=False)
            models = []
            for i in range(2):
                backend = create_backend(seed=i)
                save_direct(path / f"fixture-{i}.json", _direct(backend, 6), 4)
                models.append(InferenceArtifact(path / f"fixture-{i}.json"))
            record = fit_profile(path / "calibration.csv", models, steps=6)
            with self.assertRaisesRegex(ValueError, "follow"):
                assess_data(self.values[:600], self.times[:600], record, models, steps=6)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                assess_data(self.values, self.times, record, models[::-1], steps=6)
            missing = self.values[-3:].copy()
            missing[-1, -1] = np.nan
            result = assess_data(missing, self.times[-3:], record, models, steps=6)
            self.assertEqual(result["decision"], "abstain")
            self.assertNotIn("point_forecast", result)
            unsupported = assess_data(self.values, self.times, record, models, steps=7)
            self.assertEqual(unsupported["decision"], "abstain")
            score = evaluate_policy(self.values, self.times, record, models, steps=1, stride=24)
            self.assertGreater(score["verification_origins"], 30)
            self.assertEqual(score["accepted_origins"], score["verification_origins"])
            self.assertEqual(score["all_mechanical_forecasts"]["rmse_by_lead_feature"].shape, (1, 6))

    def test_verify_leads_by_family(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            backend = create_backend(seed=8)
            timestamped = synthetic_history(140)
            timestamped.to_csv(folder / "training.csv", index=False)
            train(folder / "training.csv", folder / "recursive.json",
                  TrainingConfig(hidden=(4,), dropout=0., epochs=1, rollout_steps=6))
            save_direct(folder / "direct.json", _direct(backend, 72), 4)
            save_direct(folder / "sequence.json", _direct(backend, 9, SequenceSpec("gru", 4, context=5)), 4)
            QuantileForecaster(backend, np.zeros(6), np.ones(6), 5, width=4).save(folder / "quantile.json")
            seasonal = _direct(backend, 12)
            member = save_member(folder, 42, seasonal, 4)
            save_ensemble(folder, [member], ConvexCorrection(np.ones((12, 6))),
                          BlockIntervals(np.ones((12, 6)), np.ones(6), 16, 12, .9), 12)
            for filename, kind, limit in (("recursive.json", "recursive", 6), ("direct.json", "direct", 72),
                                          ("sequence.json", "sequence", 9), ("quantile.json", "quantile", 5),
                                          ("seasonal-ensemble.json", "seasonal_ensemble", 12)):
                with self.subTest(family=kind):
                    report = verify_leads(InferenceArtifact(folder / filename),
                                          self.values[-72:], self.times[-72:])
                    states = [item["status"] for item in report["leads"]]
                    self.assertEqual(report["family"], kind)
                    self.assertEqual(states, ["verified"] * limit + ["rejected_as_unsupported"] * (72-limit))
            bundle = VerificationEnsemble([InferenceArtifact(folder / "direct.json"),
                                           InferenceArtifact(folder / "quantile.json")])
            bundle_report = verify_leads(bundle, self.values[-72:], self.times[-72:])
            self.assertEqual(bundle_report["family"], "checkpoint_ensemble")
            self.assertEqual(bundle_report["leads"][4]["status"], "verified")
            self.assertEqual(bundle_report["leads"][5]["status"], "rejected_as_unsupported")
        self.assertEqual(ensemble_disagreement(np.zeros((2, 3, 6)), np.ones(6)), 0)
