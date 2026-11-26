"""Local job lifecycle and registered research CLI integration."""
import base64
from contextlib import contextmanager
import csv
from http.client import HTTPConnection
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
import unittest

from Dashboard.Jobs import Jobs
from Dashboard.Reports import sensor_series
from Dashboard.Server import DashboardServer
from Examples.SyntheticPhase3 import historical_fixture
from config import FEATURES
from main import synthetic_history


def encoded(payload):
    if isinstance(payload, dict):
        payload = json.dumps(payload).encode()
    return base64.b64encode(payload).decode()


@contextmanager
def dashboard(workspace):
    with DashboardServer(0, workspace) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()

        def request(path, data=None):
            client = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            try:
                headers = {"X-Jupiter-Token": server.token}
                client.request("GET" if data is None else "POST", path,
                               None if data is None else json.dumps(data), headers)
                response = client.getresponse()
                return response.status, response.read()
            finally:
                client.close()

        try:
            yield request
        finally:
            server.shutdown()
            worker.join(timeout=2)


class DashboardResearchTests(unittest.TestCase):
    def _submit(self, request, kind, files, options):
        status, body = request("/api/jobs", {"kind": kind, "files": files, "options": options})
        self.assertEqual(status, 202, body)
        return json.loads(body)["id"]

    def _wait(self, request, identifier):
        deadline = monotonic() + 35
        while monotonic() < deadline:
            status, body = request("/api/jobs")
            self.assertEqual(status, 200)
            job = next(item for item in json.loads(body) if item["id"] == identifier)
            if job["status"] not in {"queued", "running"}:
                detail = request(f"/api/artifacts/{identifier}/failure.log")[1]
                self.assertEqual(job["status"], "complete", detail)
                return job
            sleep(.15)
        self.fail("Dashboard job did not terminate")

    def test_prepared_dataset_survives_restart_and_exports_real_training_metrics(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            source, mapping = historical_fixture(base, hours=170)
            workspace = base / "dashboard"
            with dashboard(workspace) as request:
                created = self._submit(request, "prepare",
                                       {"data": encoded(source.read_bytes()), "config": encoded(mapping)}, {})
                job = self._wait(request, created)
                self.assertIn("prepared/hourly.csv", job["artifacts"])
                status, report = request(f"/api/dataset/{created}")
                self.assertEqual(status, 200, report)
                exports = json.loads(report)
                self.assertEqual(set(exports["distributions"]), set(FEATURES))
                self.assertEqual(set(exports["seasonal"]["months"]), {"5", "6"})
                self.assertFalse(job["can_export"])

            with dashboard(workspace) as request:
                status, jobs = request("/api/jobs")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(jobs)[0]["id"], created)
                model = self._submit(request, "train", {}, {"dataset": created, "epochs": 1})
                self.assertTrue(self._wait(request, model)["can_export"])
                status, report = request(f"/api/research/{model}")
                self.assertEqual(status, 200)
                rows = json.loads(report)["metrics"]
                self.assertTrue(rows)
                status, data = request(f"/api/metrics/{model}")
                self.assertEqual(status, 200)
                records = list(csv.DictReader(io.StringIO(data.decode())))
                self.assertEqual(len(records), len(rows))
                self.assertEqual(float(records[0]["rmse"]), rows[0]["rmse"])

    def test_benchmark_and_search_use_existing_cli_and_export_method_metrics(self):
        source = encoded(synthetic_history(480).to_csv(index=False).encode())
        with TemporaryDirectory() as directory, dashboard(Path(directory)) as request:
            benchmark = self._submit(request, "benchmark", {"data": source},
                                     {"horizon": 6, "epochs": 1, "seeds": [42], "synthetic": True})
            search = self._submit(request, "search", {"data": source},
                                  {"epochs": 1, "population": 2, "generations": 1,
                                   "candidate_epochs": 1, "seed": 42})
            self._wait(request, benchmark)
            self._wait(request, search)
            status, body = request(f"/api/research/{benchmark}")
            self.assertEqual(status, 200, body)
            result = json.loads(body)
            self.assertTrue(result["manifest"]["synthetic"])
            self.assertIn("direct", {row["method"] for row in result["metrics"]})
            status, body = request(f"/api/research/{search}")
            self.assertEqual(status, 200, body)
            result = json.loads(body)
            self.assertEqual(set(result["search"]), {"fixed", "classical_ga", "quantum_inspired", "random"})
            self.assertEqual(result["search"]["classical_ga"]["candidate_count"], 2)

    def test_recovery_locks_workspace_and_marks_jobs_interrupted(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            folder = path / ("1" * 32)
            folder.mkdir()
            record = {"version": 1, "kind": "train", "status": "running", "options": {},
                      "created_at": "2020-01-01T00:00+00:00"}
            (folder / "job.json").write_text(json.dumps(record), encoding="utf-8")
            (folder / "resume.json").touch()
            jobs = Jobs(path)
            try:
                self.assertEqual(jobs.snapshot()[0]["status"], "interrupted")
                self.assertTrue(jobs.snapshot()[0]["can_resume"])
                with self.assertRaisesRegex(ValueError, "already in use"):
                    Jobs(path)
            finally:
                jobs.close()

    def test_sensor_preview_contains_numeric_measurements_only(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "readings.jsonl"
            line = {"device_id": "private-name", "measurements": {
                "leaf_mv": 320, "air_temperature_c": 28, "latitude_deg": 30, "longitude_deg": -120,
            }}
            path.write_text((json.dumps(line) + "\n") * 125 + "partial", encoding="utf-8")
            result = sensor_series(path)
        self.assertEqual(set(result["series"]), {"leaf_mv", "air_temperature_c"})
        self.assertEqual(len(result["temperature_c"]), 120)
        self.assertEqual(result["series"]["leaf_mv"][0], 320)
