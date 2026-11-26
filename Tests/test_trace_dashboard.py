"""Exercise the actual loopback survey routes with a synthetic serial receiver."""
from http.client import HTTPConnection
import json
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep
import unittest

from Dashboard.Server import DashboardServer


class SyntheticLink:
    def __init__(self, packets):
        self.packets = packets

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read_until(self, size):
        try:
            value = self.packets.get(timeout=.02)
        except Empty:
            return b""
        if isinstance(value, OSError):
            raise value
        return value


class TraceDashboardTests(unittest.TestCase):
    def test_five_sites_revisit_reconnect_and_actual_api(self):
        with TemporaryDirectory() as workspace:
            server = DashboardServer(0, workspace, ports=("synthetic",))
            packets, opens = Queue(), []
            receiver = server.trace_receiver

            def connector(port, baud):
                opens.append((port, baud))
                return SyntheticLink(packets)

            receiver.connector = connector
            receiver.retry_seconds = .03
            runner = Thread(target=server.serve_forever, daemon=True)
            runner.start()

            def request(path, body=None):
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
                try:
                    connection.request("GET" if body is None else "POST", path,
                                       None if body is None else json.dumps(body),
                                       {"X-Jupiter-Token": server.token, "Content-Type": "application/json"})
                    reply = connection.getresponse()
                    contents = reply.read()
                    self.assertLess(reply.status, 300, contents)
                    return contents
                finally:
                    connection.close()

            def wait_for(predicate):
                deadline = monotonic() + 5
                while monotonic() < deadline:
                    result = json.loads(request(endpoint))
                    if predicate(result):
                        return result
                    sleep(.025)
                self.fail("Timed out waiting for synthetic receiver")

            try:
                html = request("/trace").decode()
                self.assertIn('id="trace-stop"', html)
                self.assertIn('src="/trace.js"', html)
                survey = json.loads(request("/api/trace", {"label": "Five synthetic sites"}))
                endpoint = "/api/trace/" + survey["id"]
                request(endpoint + "/connect", {"port": "synthetic", "mapping": {
                    "fields": {"air_temperature_c": {"key": "temp", "unit": "C"},
                               "latitude_deg": {"key": "lat", "unit": "deg"},
                               "longitude_deg": {"key": "lon", "unit": "deg"}}
                }})
                wait_for(lambda x: x["receiver"]["state"] == "connected")
                for index in range(5):
                    request(endpoint + "/start", {"label": f"Location {index}", "coordinates": [0, index * .001]})
                    packets.put(json.dumps({"temp": 20 + index, "lat": 0, "lon": index * .001}).encode() + b"\n")
                    status = wait_for(lambda x: x["survey"]["locations"][-1]["segments"][0]["sample_count"] == 1)
                    self.assertNotIn("latitude_deg", status["receiver"]["latest"]["measurements"])
                    self.assertNotIn("longitude_deg", status["survey"]["locations"][-1])
                    stopped = json.loads(request(endpoint + "/stop", {}))
                    self.assertEqual(stopped["receiver"]["state"], "connected")
                self.assertEqual(len(opens), 1)
                first = stopped["survey"]["locations"][0]["id"]
                request(endpoint + "/start", {"location_id": first})
                packets.put(OSError("synthetic disconnect"))
                recovered = wait_for(lambda x: len(opens) >= 2 and x["receiver"]["state"] == "connected")
                self.assertEqual(recovered["survey"]["locations"][0]["segments"][-1]["status"], "interrupted")
                request(endpoint + "/finish", {})
                archive = json.loads(request(endpoint + "/export?format=json"))
                self.assertEqual(len(archive["survey"]["locations"]), 5)
                self.assertEqual(len(archive["survey"]["locations"][0]["segments"]), 2)
            finally:
                server.shutdown()
                server.server_close()
                runner.join(timeout=2)
