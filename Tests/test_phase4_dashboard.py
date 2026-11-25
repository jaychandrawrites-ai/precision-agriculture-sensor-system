import base64
from http.client import HTTPConnection
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
import unittest

from Dashboard.Server import DashboardServer
from main import synthetic_history


class DashboardTests(unittest.TestCase):
    def test_train_then_forecast_via_loopback_api(self):
        with TemporaryDirectory() as directory:
            server = DashboardServer(0, Path(directory), ports=("fixture-port",))
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()

            def request(method, path, data=None, token=True, host=None):
                client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                try:
                    headers = {"X-Jupiter-Token": server.token} if token else {}
                    if host is not None:
                        headers["Host"] = host
                    client.request(method, path, None if data is None else json.dumps(data), headers)
                    reply = client.getresponse()
                    return reply.status, reply.read()
                finally:
                    client.close()

            def wait(identifier):
                stop = monotonic() + 20
                while monotonic() < stop:
                    status, body = request("GET", "/api/jobs")
                    self.assertEqual(status, 200)
                    jobs = {job["id"]: job for job in json.loads(body)}
                    if jobs[identifier]["status"] not in ("queued", "running"):
                        return jobs[identifier]
                    sleep(.05)
                self.fail("Dashboard job did not terminate")

            try:
                self.assertEqual(request("GET", "/", host="example.invalid")[0], 403)
                status, html = request("GET", "/")
                self.assertEqual(status, 200)
                self.assertIn(b"Research console", html)
                data = base64.b64encode(synthetic_history().to_csv(index=False).encode()).decode()
                job = {"kind": "train", "files": {"data": data}, "options": {"epochs": 1}}
                self.assertEqual(request("POST", "/api/jobs", job, token=False)[0], 403)
                status, body = request("POST", "/api/jobs", job)
                self.assertEqual(status, 202)
                training = json.loads(body)["id"]
                self.assertEqual(wait(training)["status"], "complete")
                status, body = request("POST", "/api/jobs", {"kind": "forecast", "files": {"data": data},
                                                           "options": {"model": training, "steps": 4}})
                self.assertEqual(status, 202)
                forecast = json.loads(body)["id"]
                self.assertEqual(wait(forecast)["status"], "complete")
                status, csv = request("GET", f"/api/artifacts/{forecast}/forecast.csv")
                self.assertEqual(status, 200)
                self.assertEqual(len(csv.splitlines()), 5)
                self.assertEqual(request("GET", f"/api/artifacts/{forecast}/../input.csv")[0], 404)
                self.assertEqual(request("POST", "/api/jobs", {"kind": "live", "files": {},
                                                              "options": {"port": "/arbitrary"}})[0], 400)
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)
