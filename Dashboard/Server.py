"""Loopback-only API for the Jupiter research dashboard."""
import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import shutil
from urllib.parse import urlsplit

import numpy as np

from Acquisition.AITrace import TraceStore
from Acquisition.ReceiverPool import ReceiverPool
from Dashboard.Jobs import Jobs
from Dashboard.TraceAPI import trace_get, trace_post
from Utils.Serialization import read_json
from Dashboard.Reports import dataset_report, forecast_series, metrics_csv, research_report, sensor_series


STATIC = Path(__file__).with_name("static")
MAX_REQUEST = 33 * 1024 * 1024
ROUTES = {"/diagnostics": ("diagnostics.html", "text/html"),
          "/diagnostics.js": ("diagnostics.js", "text/javascript"),
          "/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
          "/style.css": ("style.css", "text/css"),
          "/trace": ("trace.html", "text/html"), "/trace.js": ("trace.js", "text/javascript")}


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, port, root, ports=()):
        super().__init__(("127.0.0.1", port), Handler)
        try:
            self.trace = TraceStore(Path(root) / "surveys")
            self.trace_receivers = ReceiverPool(self.trace, ports)
            self.trace_receiver = self.trace_receivers.default
            self.jobs = Jobs(root, ports=ports)
        except BaseException:
            super().server_close()
            raise
        self.token = secrets.token_urlsafe(24)
        self.authority = f"127.0.0.1:{self.server_port}"

    def server_close(self):
        super().server_close()
        self.trace_receivers.close()
        self.jobs.close()


class Handler(BaseHTTPRequestHandler):
    def _headers(self, status, size, content_type, download):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.send_header("Content-Length", str(size))
        self.end_headers()

    def _reply(self, status, payload, content_type="application/json", download=None):
        self._headers(status, len(payload), content_type, download)
        self.wfile.write(payload)

    def _file(self, artifact, content_type):
        with artifact.open("rb") as stream:
            self._headers(200, artifact.stat().st_size, content_type, artifact.name)
            shutil.copyfileobj(stream, self.wfile)

    def _json(self, status, payload):
        self._reply(status, json.dumps(payload).encode())

    def _valid_host(self):
        return self.headers.get("Host") == self.server.authority

    def do_GET(self):
        if not self._valid_host():
            return self._json(403, {"error": "Use the printed loopback URL"})
        path = urlsplit(self.path).path
        if path in ROUTES:
            name, kind = ROUTES[path]
            payload = (STATIC / name).read_bytes()
            if name in ("index.html", "trace.html"):
                payload = payload.replace(b"DASHBOARD_TOKEN", self.server.token.encode())
            return self._reply(200, payload, kind + "; charset=utf-8")
        if path == "/api/trace" or path.startswith("/api/trace/"):
            return trace_get(self)
        if path == "/api/ports":
            return self._json(200, self.server.jobs.ports)
        if path.startswith("/api/live/"):
            try:
                file = self.server.jobs.live_path(path.rsplit("/", 1)[-1])
            except KeyError:
                return self._json(404, {"error": "Live session not found"})
            return self._json(200, sensor_series(file))
        for route, handler in (("dataset", self._dataset), ("forecast", self._forecast),
                               ("research", self._research), ("metrics", self._metrics)):
            if path.startswith(f"/api/{route}/"):
                try:
                    return handler(path.rsplit("/", 1)[-1])
                except (KeyError, ValueError, OSError):
                    return self._json(404, {"error": "Result not available"})
        if path.startswith("/api/maps/"):
            try:
                identifier = path.rsplit("/", 1)[-1]
                try:
                    file = self.server.jobs.artifact(identifier, "map/field.npz")
                except KeyError:
                    file = self.server.jobs.artifact(identifier, "map/forecast-field.npz")
                info = read_json(self.server.jobs.artifact(identifier, "map/report.json"))
                with np.load(file, allow_pickle=False) as data:
                    maps = {}
                    for key in ("variational", "idw", "kriging", "mean", "std", "spread", "lower",
                                "upper", "split_rhat", "ess"):
                        if key in data:
                            grid = np.asarray(data[key])
                            maps[key] = np.where(np.isfinite(grid), grid, None).tolist()
                return self._json(200, {"layers": maps, "metadata": info})
            except (KeyError, ValueError, OSError):
                return self._json(404, {"error": "Map not found"})
        if path == "/api/jobs":
            return self._json(200, self.server.jobs.snapshot())
        if path.startswith("/api/artifacts/"):
            parts = path.split("/", 4)
            try:
                _, _, _, identifier, name = parts
                artifact = self.server.jobs.artifact(identifier, name)
            except (KeyError, ValueError):
                return self._json(404, {"error": "Artifact not found"})
            kind = {".json": "application/json", ".csv": "text/csv", ".log": "text/plain",
                    ".png": "image/png", ".svg": "image/svg+xml"}.get(
                artifact.suffix, "application/octet-stream"
            )
            return self._file(artifact, kind)
        self._json(404, {"error": "Route not found"})

    def _dataset(self, identifier):
        report = self.server.jobs.artifact(identifier, "prepared/report.json")
        hourly = self.server.jobs.artifact(identifier, "prepared/hourly.csv")
        return self._json(200, dataset_report(hourly, report))

    def _forecast(self, identifier):
        try:
            path = self.server.jobs.artifact(identifier, "forecast.csv")
        except KeyError:
            path = self.server.jobs.artifact(identifier, "inference/forecast.csv")
        return self._json(200, forecast_series(path))

    def _research(self, identifier):
        kind, path = self.server.jobs.research_report(identifier)
        return self._json(200, research_report(path, kind))

    def _metrics(self, identifier):
        kind, path = self.server.jobs.research_report(identifier)
        return self._reply(200, metrics_csv(path, kind), "text/csv; charset=utf-8", "metrics.csv")

    def do_POST(self):
        origin = self.headers.get("Origin")
        token = self.headers.get("X-Jupiter-Token", "")
        if (not self._valid_host() or origin not in (None, f"http://{self.server.authority}")
                or not hmac.compare_digest(token.encode(), self.server.token.encode())):
            return self._json(403, {"error": "Request is outside the dashboard session"})
        try:
            path = urlsplit(self.path).path
            if path == "/api/trace" or path.startswith("/api/trace/"):
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST:
                    raise ValueError("Invalid survey request size")
                return trace_post(self, json.loads(self.rfile.read(size)))
            if self.path == "/api/jobs":
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST:
                    raise ValueError("Invalid upload size")
                request = json.loads(self.rfile.read(size))
                files = {key: base64.b64decode(value, validate=True)
                         for key, value in request["files"].items()}
                identifier = self.server.jobs.submit(request["kind"], files, request.get("options", {}))
                return self._json(202, {"id": identifier})
            if self.path.startswith("/api/resume/"):
                self.server.jobs.resume(self.path.rsplit("/", 1)[-1])
                return self._json(202, {"status": "resume queued"})
            if self.path.startswith("/api/cancel/"):
                self.server.jobs.cancel(self.path.rsplit("/", 1)[-1])
                return self._json(202, {"status": "cancellation requested"})
            return self._json(404, {"error": "Route not found"})
        except ValueError as error:
            return self._json(400, {"error": str(error)[:200]})
        except (TypeError, KeyError, AttributeError, OSError):
            return self._json(400, {"error": "Invalid job options, upload, or job identifier"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--import-training-run", type=Path, action="append", default=[])
    parser.add_argument("--allow-port", action="append", default=[])
    args = parser.parse_args()
    with DashboardServer(args.port, args.workspace, args.allow_port) as server:
        for source in args.import_training_run:
            server.jobs.import_training_run(source)
        print(f"Open http://{server.authority}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
