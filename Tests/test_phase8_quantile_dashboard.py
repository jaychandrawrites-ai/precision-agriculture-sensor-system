"""Saved quantile model jobs and held-out research results over loopback."""
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Examples.SyntheticPhase3 import session_fixture
from config import FEATURES
from main import synthetic_history
from Tests import test_phase5_dashboard as helpers
from Tests.test_phase5_dashboard import dashboard, encoded


class QuantileDashboardTests(unittest.TestCase):
    _submit = helpers.DashboardResearchTests._submit
    _wait = helpers.DashboardResearchTests._wait

    def test_train_select_forecast_export(self):
        source = encoded(synthetic_history(300).to_csv(index=False).encode())
        with TemporaryDirectory() as directory, dashboard(Path(directory) / "dashboard") as request:
            fixture, schema, configuration = session_fixture(Path(directory))
            status, body = request("/api/jobs", {"kind": "quantile_train", "files": {"data": source},
                                                  "options": {"levels": [.9, .1]}})
            self.assertEqual(status, 400, body)
            model = self._submit(request, "quantile_train", {"data": source},
                                 {"horizon": 4, "epochs": 1, "levels": [.1, .5, .9]})
            job = self._wait(request, model)
            self.assertEqual(job["forecast_horizon"], 4)
            self.assertIn("quantile-model.json", job["artifacts"])
            status, body = request(f"/api/research/{model}")
            self.assertEqual(status, 200, body)
            research = json.loads(body)
            self.assertEqual(research["calibration"]["levels"], [.1, .5, .9])
            self.assertGreater(research["calibration"]["test_origins"], 0)
            status, artifact = request(f"/api/metrics/{model}")
            self.assertEqual(status, 200, artifact)
            rows = list(csv.DictReader(io.StringIO(artifact.decode())))
            self.assertEqual(len(rows), 4 * len(FEATURES) * 4)
            self.assertEqual({row["quantile_level"] for row in rows}, {"", "0.1", "0.5", "0.9"})
            forecast = self._submit(request, "forecast", {"data": source}, {"model": model, "steps": 2})
            self._wait(request, forecast)
            status, body = request(f"/api/forecast/{forecast}")
            self.assertEqual(status, 200, body)
            report = json.loads(body)
            self.assertEqual(len(report["timestamps"]), 2)
            for feature in FEATURES:
                levels = report["quantiles"][feature]
                self.assertEqual([line["level"] for line in levels], [.1, .5, .9])
                self.assertEqual(report["series"][feature], levels[1]["value"])
                self.assertTrue(all(low <= mid <= high
                                    for low, mid, high in zip(*(item["value"] for item in levels))))
            status, body = request("/api/jobs", {"kind": "forecast", "files": {"data": source},
                                                  "options": {"model": model, "steps": 5}})
            self.assertEqual(status, 400, body)

            recorded = self._submit(request, "replay", {
                "data": encoded((fixture / "raw.jsonl").read_bytes()), "config": encoded(schema),
            }, {})
            self._wait(request, recorded)
            field = self._submit(request, "field", {"config": encoded(configuration)},
                                 {"model": model, "session": recorded, "steps": 2})
            self._wait(request, field)
            status, body = request(f"/api/forecast/{field}")
            self.assertEqual(status, 200, body)
            field_series = json.loads(body)
            self.assertEqual(len(field_series["timestamps"]), 2)
            self.assertEqual(field_series["quantiles"][FEATURES[0]][1]["level"], .5)
            status, body = request(f"/api/artifacts/{field}/inference/report.json")
            self.assertEqual(status, 200, body)
            metadata = json.loads(body)["quantiles"]
            self.assertEqual(metadata["axes"], ["lead_hour", "feature", "quantile"])
            self.assertEqual(len(metadata["values"]), 2)

    def test_sequence_search_exports_selectable_models_and_forecasts(self):
        source = encoded(synthetic_history(300).to_csv(index=False).encode())
        with TemporaryDirectory() as directory, dashboard(Path(directory) / "dashboard") as request:
            model = self._submit(request, "search", {"data": source},
                                 {"sequence_search": True, "epochs": 1, "candidate_epochs": 1,
                                  "generations": 1, "population": 2})
            job = self._wait(request, model)
            self.assertIn("models/classical_ga.json", job["artifacts"])
            self.assertIn("models/random.json", job["artifacts"])
            forecast = self._submit(request, "forecast", {"data": source},
                                    {"model": model, "search_model": "random", "steps": 2})
            self._wait(request, forecast)
            status, body = request(f"/api/forecast/{forecast}")
            self.assertEqual(status, 200, body)
            self.assertEqual(len(json.loads(body)["timestamps"]), 2)

    def test_direct_controls_and_portable_blend_in_jobs(self):
        frame = synthetic_history(300)
        source = encoded(frame.to_csv(index=False).encode())
        with TemporaryDirectory() as directory, dashboard(Path(directory) / "dashboard") as request:
            members = []
            for width in (8, 12):
                job = self._submit(request, "direct_train", {"data": source}, {
                    "width": width, "epochs": 1, "horizon": 3, "direct_optimizer": "adagrad",
                    "scheduler": "plateau", "memory_bytes": 50000,
                })
                self._wait(request, job)
                status, data = request(f"/api/artifacts/{job}/direct-model.training.json")
                self.assertEqual(status, 200)
                report = json.loads(data)
                self.assertEqual(report["optimizer"], "adagrad")
                self.assertEqual(report["memory"]["plan"]["budget_bytes"], 50000)
                members.append(job)
            blend = self._submit(request, "blend", {"data": source}, {
                "members": members, "horizon": 3,
                "validation_start": frame.timestamp.iloc[260].tz_localize(None).isoformat(),
                "validation_end": frame.timestamp.iloc[290].tz_localize(None).isoformat(),
            })
            self._wait(request, blend)
            forecast = self._submit(request, "forecast", {"data": source}, {"model": blend, "steps": 2})
            self._wait(request, forecast)
            status, data = request(f"/api/forecast/{forecast}")
            self.assertEqual(status, 200)
            self.assertEqual(len(json.loads(data)["timestamps"]), 2)
            figure = self._submit(request, "figure", {}, {"figure_source": forecast})
            self._wait(request, figure)
            status, svg = request(f"/api/artifacts/{figure}/figure.svg")
            self.assertEqual(status, 200)
            self.assertIn(b"<svg", svg)
