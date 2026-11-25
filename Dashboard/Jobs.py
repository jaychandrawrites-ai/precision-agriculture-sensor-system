"""Bounded local subprocess jobs using only registered scientific entry points."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Lock
from uuid import uuid4

from Utils.Serialization import atomic_output, decode_json


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = {
    "inspect": ("data",),
    "train": ("data",),
    "forecast": ("data",),
    "replay": ("data", "config"),
    "spatial": ("data", "config"),
    "live": (),
}
ARTIFACTS = {
    "inspect": {"report.json"},
    "train": {"report.json", "model.json"},
    "forecast": {"forecast.csv"},
    "replay": {"session/readings.jsonl"},
    "spatial": {"map/report.json", "map/field.npz"},
    "live": {"session/readings.jsonl", "session/raw.jsonl"},
}
MAX_DECODED = 12 * 1024 * 1024


@dataclass
class Job:
    identifier: str
    kind: str
    folder: Path
    status: str = "queued"
    cancel: Event = field(default_factory=Event)


class Jobs:
    def __init__(self, root, *, capacity=64, ports=()):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if capacity < 1:
            raise ValueError("Job capacity must be positive")
        self.capacity = capacity
        self.ports = tuple(dict.fromkeys(ports))
        self.store = {}
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _model(self, value):
        if not isinstance(value, str) or value not in self.store:
            raise ValueError("Choose a model from a completed training job")
        job = self.store[value]
        if job.kind != "train" or job.status != "complete":
            raise ValueError("Selected training job has no completed checkpoint")
        return job.folder / "model.json"

    @staticmethod
    def _number(raw, label, default, bounds):
        value = raw.get(label, default)
        if type(value) is not int or not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"{label} must be an integer in {bounds}")
        return value

    def _check_map(self, payload):
        config = decode_json(payload)
        width, height = len(config["east_m"]), len(config["north_m"])
        if not 2 <= width <= 150 or not 2 <= height <= 150 or width * height > 10000:
            raise ValueError("Dashboard grid exceeds the spatial job budget")
        self._number(config.get("solver", {}), "iterations", 500, (1, 2000))
        if "sampling" in config:
            sampling = config["sampling"]
            draws = self._number(sampling, "draws", 100, (2, 1000))
            chains = self._number(sampling, "chain_count", 4, (1, 8))
            burn = self._number(sampling, "burn_in", 100, (0, 2000))
            thin = self._number(sampling, "thin", 2, (1, 10))
            if chains * (draws * thin + burn) * width * height > 12_000_000:
                raise ValueError("Dashboard sampling exceeds the spatial job budget")

    def submit(self, kind, files, options):
        if not isinstance(kind, str) or kind not in SPEC or set(files) != set(SPEC[kind]):
            raise ValueError("Missing files or unsupported job")
        if any(type(data) is not bytes or not data or len(data) > MAX_DECODED for data in files.values()):
            raise ValueError("Input files must be nonempty and at most 12 MiB each")
        if kind == "spatial":
            self._check_map(files["config"])
        port = options.get("port")
        if kind == "live" and (not isinstance(port, str) or port not in self.ports):
            raise ValueError("Select an explicitly allowed local serial port")
        with self.lock:
            if len(self.store) >= self.capacity:
                raise ValueError("Dashboard job limit reached")
            model = self._model(options.get("model")) if kind == "forecast" else None
            epochs = self._number(options, "epochs", 30, (1, 300))
            steps = self._number(options, "steps", 6, (1, 168))
            identifier = uuid4().hex
            folder = self.root / identifier
            folder.mkdir(exist_ok=False, mode=0o700)
            source = folder / ("input.jsonl" if kind == "replay" else "input.csv")
            config = folder / "config.json"
            for name, data in files.items():
                with atomic_output(source if name == "data" else config, binary=True) as file:
                    file.write(data)
            jobs = {
                "inspect": ("azmet.py", "inspect", source),
                "train": ("train.py", source, folder / "model.json", "--epochs", epochs),
                "forecast": ("forecast.py", source, model, "--steps", steps),
                "replay": ("acquire.py", "--replay", source, "--packet-schema", config, folder / "session"),
                "spatial": ("Pipeline.py", "map", source, config, folder / "map"),
                "live": ("acquire.py", "--port", port, "--baud",
                         self._number(options, "baud", 9600, (300, 1000000)),
                         "--reconnect-attempts", 3, folder / "session"),
            }
            job = Job(identifier, kind, folder)
            self.store[identifier] = job
            self.executor.submit(self._run, job, jobs[kind])
            return identifier

    def _run(self, job, args):
        with self.lock:
            if job.cancel.is_set():
                job.status = "cancelled"
                return
            job.status = "running"
        output_name = "forecast.csv" if job.kind == "forecast" else "report.json"
        try:
            with (job.folder / output_name).open("wb") as output, (job.folder / "failure.log").open("wb") as error:
                process = subprocess.Popen([sys.executable, *(str(value) for value in args)], cwd=REPOSITORY,
                                           stdout=output, stderr=error, stdin=subprocess.DEVNULL,
                                           shell=False)
                try:
                    while process.poll() is None:
                        if job.cancel.wait(.2):
                            process.terminate()
                            try:
                                process.wait(timeout=4)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            break
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            with self.lock:
                job.status = ("cancelled" if job.cancel.is_set()
                              else "complete" if process.returncode == 0 else "failed")
        except OSError:
            with self.lock:
                job.status = "failed"

    def cancel(self, identifier):
        with self.lock:
            job = self.store[identifier]
            if job.status in ("queued", "running"):
                job.cancel.set()

    def snapshot(self):
        with self.lock:
            return [{"id": job.identifier, "kind": job.kind, "status": job.status,
                     "artifacts": sorted(name for name in ARTIFACTS[job.kind]
                                         if (job.folder / name).is_file())
                     if job.status == "complete" or job.kind == "live" and job.status == "cancelled" else
                     ["failure.log"] if job.status == "failed" else []} for job in reversed(self.store.values())]

    def artifact(self, identifier, name):
        with self.lock:
            if not re.fullmatch(r"[0-9a-f]{32}", identifier):
                raise KeyError(identifier)
            job = self.store[identifier]
            if (name == "failure.log" and job.status == "failed"
                    or (job.status == "complete" or job.kind == "live" and job.status == "cancelled")
                    and name in ARTIFACTS[job.kind]):
                path = job.folder / name
                if path.is_file():
                    return path
        raise KeyError(name)

    def live_path(self, identifier):
        with self.lock:
            job = self.store[identifier]
            if job.kind != "live":
                raise KeyError(identifier)
            return job.folder / "session/readings.jsonl"

    def close(self):
        with self.lock:
            for job in self.store.values():
                if job.status in ("queued", "running"):
                    job.cancel.set()
        self.executor.shutdown()
