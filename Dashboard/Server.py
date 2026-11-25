"""Loopback-only API for the Jupiter research dashboard."""
import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

import numpy as np
from pathlib import Path
import secrets
from urllib.parse import urlsplit

from Dashboard.Jobs import Jobs


STATIC = Path(__file__).with_name("static")
MAX_REQUEST = 33 * 1024 * 1024
ROUTES = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
          "/style.css": ("style.css", "text/css")}


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, port, root, ports=()):
        super().__init__(("127.0.0.1", port), Handler)
        self.jobs = Jobs(root, ports=ports)
        self.token = secrets.token_urlsafe(24)
        self.authority = f"127.0.0.1:{self.server_port}"

    def server_close(self):
        super().server_close()
        self.jobs.close()


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status, payload, content_type="application/json", download=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

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
            if name == "index.html":
                payload = payload.replace(b"DASHBOARD_TOKEN", self.server.token.encode())
            return self._reply(200, payload, kind + "; charset=utf-8")
        if path == "/api/ports":
            return self._json(200, self.server.jobs.ports)
        if path.startswith("/api/live/"):
            try:
                file = self.server.jobs.live_path(path.rsplit("/", 1)[-1])
            except KeyError:
                return self._json(404, {"error": "Live session not found"})
            rows = []
            if file.is_file():
                with file.open("rb") as stream:
                    size = file.stat().st_size
                    stream.seek(max(0, size - 65536))
                    if size > 65536:
                        stream.readline()
                    for line in stream:
                        try:
                            row = json.loads(line)
                            value = row["measurements"].get("air_temperature_c")
                            if isinstance(value, (int, float)) and np.isfinite(value):
                                rows.append(value)
                        except (ValueError, KeyError, TypeError):
                            continue
            return self._json(200, {"temperature_c": rows[-120:]})
        if path.startswith("/api/maps/"):
            try:
                file = self.server.jobs.artifact(path.rsplit("/", 1)[-1], "map/field.npz")
                with np.load(file, allow_pickle=False) as data:
                    maps = {key: data[key].tolist() for key in ("variational", "std") if key in data}
            except (KeyError, ValueError):
                return self._json(404, {"error": "Map not found"})
            return self._json(200, maps)
        if path == "/api/jobs":
            return self._json(200, self.server.jobs.snapshot())
        if path.startswith("/api/artifacts/"):
            parts = path.split("/", 4)
            try:
                _, _, _, identifier, name = parts
                artifact = self.server.jobs.artifact(identifier, name)
            except (KeyError, ValueError):
                return self._json(404, {"error": "Artifact not found"})
            kind = {".json": "application/json", ".csv": "text/csv", ".log": "text/plain"}.get(
                artifact.suffix, "application/octet-stream"
            )
            return self._reply(200, artifact.read_bytes(), kind, artifact.name)
        self._json(404, {"error": "Route not found"})

    def do_POST(self):
        origin = self.headers.get("Origin")
        token = self.headers.get("X-Jupiter-Token", "")
        if (not self._valid_host() or origin not in (None, f"http://{self.server.authority}")
                or not hmac.compare_digest(token.encode(), self.server.token.encode())):
            return self._json(403, {"error": "Request is outside the dashboard session"})
        try:
            if self.path == "/api/jobs":
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST:
                    raise ValueError("Invalid upload size")
                request = json.loads(self.rfile.read(size))
                files = {key: base64.b64decode(value, validate=True)
                         for key, value in request["files"].items()}
                identifier = self.server.jobs.submit(request["kind"], files, request.get("options", {}))
                return self._json(202, {"id": identifier})
            if self.path.startswith("/api/cancel/"):
                self.server.jobs.cancel(self.path.rsplit("/", 1)[-1])
                return self._json(202, {"status": "cancellation requested"})
            return self._json(404, {"error": "Route not found"})
        except (ValueError, TypeError, KeyError, AttributeError):
            return self._json(400, {"error": "Invalid job options, upload, or job identifier"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-port", action="append", default=[])
    args = parser.parse_args()
    with DashboardServer(args.port, args.workspace, args.allow_port) as server:
        print(f"Open http://{server.authority}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
