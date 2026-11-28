"""Bounded local subprocess jobs using only registered scientific entry points."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
from threading import Event, Lock
from uuid import uuid4

from Utils.Serialization import atomic_output, decode_json, read_json, write_json


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = {
    "inspect": ("data",),
    "prepare": ("data", "config"),
    "train": ("data",),
    "forecast": ("data",),
    "blend": ("data",),
    "figure": (),
    "replay": ("data", "config"),
    "spatial": ("data", "config"),
    "live": (),
    "health": (),
    "field": ("config",),
    "benchmark": ("data",),
    "search": ("data",),
    "direct_train": ("data",),
    "quantile_train": ("data",),
    "adaptive_spatial": ("data", "config"),
    "ensemble_spatial": ("data", "config"),
    "joint": (),
}
ARTIFACTS = {
    "inspect": {"report.json"},
    "prepare": {"prepared/hourly.csv", "prepared/flags.csv", "prepared/report.json"},
    "train": {"report.json", "model.json", "diagnostics/metrics.json", "diagnostics/evaluation.npz",
              "diagnostics/training.png", "diagnostics/actual-predicted.png", "diagnostics/horizon.png",
              "diagnostics/residuals.png", "diagnostics/parameters.png", "diagnostics/internal.png",
              "diagnostics/evolution.json", "diagnostics/evolution.png",
              "diagnostics/spectral.png", "diagnostics/spectral.npz"},
    "forecast": {"forecast.csv"},
    "blend": {"blend.json"},
    "figure": {"figure.png", "figure.svg"},
    "replay": {"session/readings.jsonl"},
    "spatial": {"map/report.json", "map/field.npz"},
    "live": {"session/readings.jsonl", "session/raw.jsonl"},
    "health": {"report.json"},
    "field": {"inference/forecast.csv", "inference/coverage.csv", "inference/report.json"},
    "benchmark": {"report.json"},
    "search": {"report.json"},
    "direct_train": {"direct-model.json", "direct-model.training.json"},
    "quantile_train": {"quantile-model.json", "report.json"},
    "adaptive_spatial": {"adaptive-map.json"},
    "ensemble_spatial": {"map/report.json", "map/forecast-field.npz"},
    "joint": {"report.json"},
}
# Restrict shareable bundles to artifacts which don't carry packet/device identifiers,
# spatial coordinates or field settings.
RESEARCH_ARTIFACTS = {
    "train": {"report.json"},
    "benchmark": {"report.json"},
    "search": {"report.json"},
    "quantile_train": {"report.json"},
}
SEARCH_MODELS = ("fixed", "random", "classical_ga", "quantum_inspired", "hybrid_amplitude")
ARTIFACTS["search"].update(f"models/{method}.json" for method in SEARCH_MODELS)

BLEND_INPUTS = {"train": "model.json", "direct_train": "direct-model.json",
                "quantile_train": "quantile-model.json", "search": "models/classical_ga.json"}

FIGURE_SOURCE = {
    "train": ("training", "report.json"),
    "direct_train": ("training", "direct-model.training.json"),
    "forecast": ("forecast", "forecast.csv"),
    "field": ("forecast", "inference/forecast.csv"),
    "search": ("pareto", "report.json"),
}

DATASET_INPUTS = {"train", "direct_train", "quantile_train", "benchmark", "search"}
IN_PROGRESS = {"queued", "running"}
FINAL = {"complete", "failed", "cancelled", "interrupted"}
ID = re.compile(r"[0-9a-f]{32}")
MAX_DECODED = 12 * 1024 * 1024


@dataclass
class Job:
    identifier: str
    kind: str
    folder: Path
    options: dict
    created_at: str
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
        self.lease = self._claim_workspace()
        try:
            self._restore()
            self.executor = ThreadPoolExecutor(max_workers=1)
        except BaseException:
            self.lease.close()
            raise

    def import_training_run(self, source):
        """Register a local, completed CLI training run without retraining."""
        from hashlib import sha256
        import shutil
        from Utils.ModelLoader import load_forecaster

        source = Path(source)
        model = load_forecaster(source / "model.json")
        metrics = source / "diagnostics/metrics.json"
        report = read_json(metrics)
        if report.get("checkpoint_sha256") != model.checkpoint_sha256:
            raise ValueError("Diagnostic report and checkpoint hashes differ")
        identifier = sha256(("cli-training:" + model.checkpoint_sha256).encode()).hexdigest()[:32]
        with self.lock:
            if identifier in self.store:
                return identifier
            if len(self.store) >= self.capacity:
                raise ValueError("Dashboard workspace job limit reached")
            folder = self.root / identifier
            folder.mkdir()
            try:
                (folder / "diagnostics").mkdir()
                for name in ARTIFACTS["train"] - {"report.json"}:
                    original = source / name
                    if original.is_file():
                        shutil.copyfile(original, folder / name)
                if sha256((folder / "model.json").read_bytes()).hexdigest() != model.checkpoint_sha256:
                    raise ValueError("Checkpoint changed during import")
                shutil.copyfile(metrics, folder / "report.json")
                options = {}
                if model.spectral is not None and model.spectral.residual is not None:
                    options["label"] = f"QSEN v3 ({identifier[:8]})"
                job = Job(identifier, "train", folder, options, datetime.now(timezone.utc).isoformat(), "complete")
                self._save(job)
                self.store[identifier] = job
            except BaseException:
                shutil.rmtree(folder)
                raise
        return identifier

    def _claim_workspace(self):
        """Exclude a second running server while allowing recovery after a crash."""
        path = self.root / ".dashboard.lock"
        if path.is_symlink():
            raise ValueError("Dashboard workspace lock must be a regular local file")
        lease = path.open("a+b")
        try:
            lease.seek(0)
            if not lease.read(1):
                lease.write(b"\0")
                lease.flush()
            lease.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lease.close()
            raise ValueError("Dashboard workspace is already in use") from exc
        return lease

    def _restore(self):
        restored = []
        for folder in self.root.iterdir():
            if not ID.fullmatch(folder.name) or folder.is_symlink() or not folder.is_dir():
                continue
            receipt = folder / "job.json"
            if receipt.is_symlink() or not receipt.is_file():
                continue
            try:
                record = read_json(receipt)
                if (record["version"] != 1 or record["kind"] not in SPEC
                        or record["status"] not in IN_PROGRESS | FINAL
                        or not isinstance(record["options"], dict)
                        or not isinstance(record["created_at"], str)):
                    continue
                job = Job(folder.name, record["kind"], folder, record["options"],
                          record["created_at"], record["status"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            # A subprocess from an earlier server must never restart automatically.
            if job.status in IN_PROGRESS:
                self._status(job, "interrupted")
            restored.append(job)
        self.store = {job.identifier: job for job in sorted(restored, key=lambda job: job.created_at)}

    def _save(self, job):
        write_json(job.folder / "job.json", {"version": 1, "kind": job.kind,
                   "options": job.options, "created_at": job.created_at, "status": job.status})

    def _status(self, job, status):
        job.status = status
        self._save(job)

    @staticmethod
    def _contained_file(job, name):
        path = job.folder / name
        if path.resolve() == path and path.is_file():
            return path
        raise KeyError(name)

    def _linked(self, value, kinds, artifact):
        if not isinstance(value, str) or value not in self.store:
            raise ValueError("Choose a saved source from the job list")
        job = self.store[value]
        if job.kind not in kinds or not self._available(job, artifact):
            raise ValueError("The selected job has no available source file")
        return self._contained_file(job, artifact)

    @staticmethod
    def _number(raw, label, default, bounds):
        value = raw.get(label, default)
        if type(value) is not int or not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"{label} must be an integer between {bounds[0]} and {bounds[1]}")
        return value

    @staticmethod
    def _choice(raw, label, default, allowed):
        value = raw.get(label, default)
        if value not in allowed:
            raise ValueError(f"Choose a supported {label}")
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

    def _options(self, kind, raw):
        settings = {}
        if kind in DATASET_INPUTS and raw.get("dataset"):
            self._linked(raw["dataset"], {"prepare"}, "prepared/hourly.csv")
            settings["dataset"] = raw["dataset"]
        if kind in {"forecast", "field"}:
            identifier = raw.get("model")
            candidate = self.store.get(identifier) if isinstance(identifier, str) else None
            if candidate is not None and candidate.kind == "blend":
                settings["model_artifact"] = "blend.json"
                self._linked(identifier, {"blend"}, "blend.json")
                limit = candidate.options["horizon"]
                settings["steps"] = self._number(raw, "steps", limit, (1, limit))
            elif candidate is not None and candidate.kind == "search":
                method = self._choice(raw, "search_model", "classical_ga", SEARCH_MODELS)
                settings["model_artifact"] = f"models/{method}.json"
                self._linked(identifier, {"search"}, settings["model_artifact"])
                settings["steps"] = self._number(raw, "steps", 6, (1, 6))
            elif candidate is not None and candidate.kind == "quantile_train":
                self._linked(identifier, {"quantile_train"}, "quantile-model.json")
                limit = candidate.options["horizon"]
                settings["steps"] = self._number(raw, "steps", limit, (1, limit))
            elif candidate is not None and candidate.kind == "direct_train":
                self._linked(identifier, {"direct_train"}, "direct-model.json")
                if kind == "field":
                    settings["steps"] = self._number(raw, "steps", 6, (1, 72))
            else:
                self._linked(identifier, {"train"}, "model.json")
                settings["steps"] = self._number(raw, "steps", 6, (1, 168))
            settings["model"] = identifier
        if kind in {"health", "field"}:
            self._linked(raw.get("session"), {"live", "replay"}, "session/readings.jsonl")
            settings["session"] = raw["session"]
        if kind in DATASET_INPUTS:
            settings["epochs"] = self._number(raw, "epochs", 30, (1, 300 if kind == "train" else 50))
        if kind == "train":
            settings["diagnostic_every"] = self._number(raw, "diagnostic_every", 0, (0, 300))
            settings["optimizer"] = self._choice(raw, "optimizer", "adam", ("adam", "sgd"))
            settings["loss"] = self._choice(raw, "loss", "huber", ("huber", "mse"))
        if kind == "figure":
            from config import FEATURES

            identifier = raw.get("figure_source")
            candidate = self.store.get(identifier) if isinstance(identifier, str) else None
            if candidate is None or candidate.kind not in FIGURE_SOURCE:
                raise ValueError("Choose a completed training, forecast or sequence search job")
            figure_kind, artifact = FIGURE_SOURCE[candidate.kind]
            self._linked(identifier, {candidate.kind}, artifact)
            settings.update(figure_source=identifier, figure_kind=figure_kind, artifact=artifact,
                            feature=self._choice(raw, "feature", FEATURES[0], FEATURES))
        if kind == "blend":
            from datetime import datetime

            members = raw.get("members")
            if (not isinstance(members, list) or not 2 <= len(members) <= 6
                    or not all(isinstance(x, str) for x in members) or len(set(members)) != len(members)):
                raise ValueError("Choose two to six distinct saved checkpoints")
            for identifier in members:
                candidate = self.store.get(identifier)
                if candidate is None or candidate.kind not in BLEND_INPUTS:
                    raise ValueError("Unknown ensemble member")
                self._linked(identifier, {candidate.kind}, BLEND_INPUTS[candidate.kind])
            try:
                start, end = (datetime.fromisoformat(raw[key]) for key in ("validation_start", "validation_end"))
            except (KeyError, ValueError, TypeError) as error:
                raise ValueError("Enter an ordered validation period in UTC") from error
            if start.tzinfo is not None or end.tzinfo is not None or start >= end:
                raise ValueError("Enter an ordered validation period in UTC")
            settings.update(members=members, validation_start=start.isoformat(), validation_end=end.isoformat(),
                            horizon=self._number(raw, "horizon", 6, (1, 72)))
        if kind == "direct_train":
            from Temporal.Direct import DIRECT_ARCHITECTURES

            settings["architecture"] = self._choice(raw, "architecture", "gated", DIRECT_ARCHITECTURES)
            settings["width"] = self._number(raw, "width", 32, (1, 128))
            settings["horizon"] = self._number(raw, "horizon", 6, (1, 72))
            settings["direct_optimizer"] = self._choice(raw, "direct_optimizer", "adam",
                                                        ("adam", "adamw", "sgd", "rmsprop", "adagrad"))
            settings["scheduler"] = self._choice(raw, "scheduler", "cosine", ("cosine", "plateau"))
            settings["memory_bytes"] = self._number(raw, "memory_bytes", 0, (0, 2**30))
        if kind == "quantile_train":
            levels = raw.get("levels", [.05, .5, .95])
            if (not isinstance(levels, list) or not 2 <= len(levels) <= 9
                    or any(type(level) not in (int, float) or not 0 < level < 1 for level in levels)
                    or any(a >= b for a, b in zip(levels, levels[1:]))):
                raise ValueError("Use two to nine strictly increasing quantile levels between 0 and 1")
            settings["levels"] = levels
            settings["horizon"] = self._number(raw, "horizon", 6, (1, 72))
            settings["seed"] = self._number(raw, "seed", 42, (0, 2**32 - 1))
        if kind == "joint":
            settings["epochs"] = self._number(raw, "epochs", 160, (1, 500))
            settings["seed"] = self._number(raw, "seed", 82, (0, 2**32 - 1))
        if kind == "benchmark":
            settings["horizon"] = self._number(raw, "horizon", 12, (1, 72))
            seeds = raw.get("seeds", [42, 43, 44])
            if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 5
                    or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
                    or len(set(seeds)) != len(seeds)):
                raise ValueError("Use one to five distinct nonnegative integer seeds below 2³²")
            settings["seeds"] = seeds
            synthetic = raw.get("synthetic", False)
            if type(synthetic) is not bool:
                raise ValueError("Mark the dataset as synthetic or observational")
            settings["synthetic"] = synthetic
        if kind == "search":
            sequence = raw.get("sequence_search", False)
            if type(sequence) is not bool:
                raise ValueError("Sequence search must be a boolean")
            settings["sequence_search"] = sequence
            settings["population"] = self._number(raw, "population", 4, (2, 8))
            settings["generations"] = self._number(raw, "generations", 3, (1, 10))
            settings["candidate_epochs"] = self._number(raw, "candidate_epochs", 6, (1, 20))
            settings["seed"] = self._number(raw, "seed", 42, (0, 2**32 - 1))
            if settings["population"] * settings["generations"] * settings["candidate_epochs"] > 240:
                raise ValueError("Evolution budget is limited to 240 candidate epochs per method")
        if kind == "inspect":
            headerless = raw.get("headerless", False)
            if type(headerless) is not bool:
                raise ValueError("Expected a headerless indicator")
            settings["headerless"] = headerless
            settings["delimiter"] = self._choice(raw, "delimiter", ",", (",", ";", "\t"))
        if kind == "live":
            settings["port"] = self._choice(raw, "port", None, self.ports)
            settings["baud"] = self._number(raw, "baud", 9600, (300, 1000000))
        return settings

    def _command(self, job):
        folder, kind, opts = job.folder, job.kind, job.options
        data = folder / ("input.jsonl" if kind == "replay" else "input.csv")
        if "dataset" in opts:
            data = self._linked(opts["dataset"], {"prepare"}, "prepared/hourly.csv")
        config = folder / "config.json"
        if kind == "inspect":
            args = ["azmet.py", "inspect", data, "--delimiter", opts["delimiter"]]
            return args + (["--headerless"] if opts["headerless"] else [])
        if kind == "prepare":
            return ["azmet.py", "prepare", data, config, folder / "prepared"]
        if kind == "train":
            command = ["train.py", data, folder / "model.json", "--epochs", opts["epochs"],
                    "--optimizer", opts["optimizer"], "--loss", opts["loss"],
                    "--resume-state", folder / "resume.json", "--diagnostics", folder / "diagnostics", "--diagnostic-every", opts.get("diagnostic_every", 0)]
            if "dataset" in opts:
                command += ["--preparation-report", self._linked(opts["dataset"], {"prepare"}, "prepared/report.json")]
            return command
        if kind == "benchmark":
            args = ["forecast_benchmark.py", data, folder / "report.json", "--epochs", opts["epochs"],
                    "--horizon", opts["horizon"], "--seeds", *opts["seeds"]]
            return args + (["--synthetic"] if opts["synthetic"] else [])
        if kind == "search":
            args = ["experiments.py", data, "--epochs", opts["epochs"], "--random-baseline",
                    "--population", opts["population"], "--generations", opts["generations"],
                    "--candidate-epochs", opts["candidate_epochs"], "--seed", opts["seed"],
                    "--search-directory", folder / "search"]
            if opts.get("sequence_search"):
                args += ["--sequence-space", "--export-selected", folder / "models"]
            return args
        if kind == "figure":
            candidate = self.store[opts["figure_source"]]
            source = self._linked(candidate.identifier, {candidate.kind}, opts["artifact"])
            return ["research_figures.py", opts["figure_kind"], source, folder / "figure",
                    "--feature", opts["feature"]]
        if kind == "blend":
            args = ["blend_forecast.py", data, folder / "blend.json", "--start", opts["validation_start"],
                    "--end", opts["validation_end"], "--steps", opts["horizon"]]
            for identifier in opts["members"]:
                candidate = self.store[identifier]
                args += ["--member", self._linked(identifier, {candidate.kind}, BLEND_INPUTS[candidate.kind]),
                         "--name", identifier]
            return args
        if kind == "direct_train":
            args = ["direct_forecast.py", "train", data, folder / "direct-model.json", "--epochs", opts["epochs"],
                    "--horizon", opts["horizon"], "--width", opts["width"], "--architecture", opts["architecture"],
                    "--optimizer", opts.get("direct_optimizer", "adam"), "--scheduler", opts.get("scheduler", "cosine"),
                    "--progress", folder / "direct-progress.json"]
            return args + (["--memory-bytes", opts["memory_bytes"]] if opts.get("memory_bytes") else [])
        if kind == "quantile_train":
            levels = [item for level in opts["levels"] for item in ("--level", level)]
            return ["quantile_forecast.py", data, folder / "quantile-model.json", folder / "report.json",
                    "--horizon", opts["horizon"], "--epochs", opts["epochs"], "--seed", opts["seed"], *levels]
        if kind == "joint":
            return ["-m", "Examples.SyntheticJoint", folder / "report.json", "--epochs", opts["epochs"],
                    "--seed", opts["seed"]]
        if kind == "adaptive_spatial":
            return ["adaptive_map.py", data, config, folder / "adaptive-map.json"]
        if kind == "ensemble_spatial":
            return ["spatial_forecast.py", data, config, folder / "map"]
        if kind == "replay":
            return ["acquire.py", "--replay", data, "--packet-schema", config, folder / "session"]
        if kind == "spatial":
            return ["Pipeline.py", "map", data, config, folder / "map"]
        if kind == "live":
            return ["acquire.py", "--port", opts["port"], "--baud", opts["baud"],
                    "--reconnect-attempts", 3, folder / "session"]
        if kind == "forecast":
            candidate = self.store[opts["model"]]
            if candidate.kind == "quantile_train":
                model = self._linked(opts["model"], {"quantile_train"}, "quantile-model.json")
                return ["quantile_forecast.py", "predict", data, model, folder / "forecast.csv",
                        "--steps", opts["steps"]]
            if "steps" not in opts:
                model = self._linked(opts["model"], {"direct_train"}, "direct-model.json")
                return ["direct_forecast.py", "predict", data, model, folder / "forecast.csv"]
            model = self._linked(opts["model"], {candidate.kind}, opts.get("model_artifact", "model.json"))
            return ["forecast.py", data, model, "--steps", opts["steps"]]
        source = self._linked(opts["session"], {"live", "replay"}, "session/readings.jsonl")
        if kind == "health":
            return ["Pipeline.py", "session-health", source, folder / "report.json"]
        candidate = self.store[opts["model"]]
        artifact = {"direct_train": "direct-model.json", "quantile_train": "quantile-model.json"}.get(
            candidate.kind, "model.json")
        model = self._linked(opts["model"], {candidate.kind}, opts.get("model_artifact", artifact))
        return ["Pipeline.py", "forecast-session", source, config, folder / "inference",
                "--checkpoint", model, "--steps", opts["steps"]]

    def submit(self, kind, files, options):
        if not isinstance(kind, str) or kind not in SPEC:
            raise ValueError("Unsupported job")
        expected = set(SPEC[kind])
        if kind in DATASET_INPUTS and options.get("dataset"):
            expected.remove("data")
        if set(files) != expected:
            raise ValueError("Supply exactly the inputs needed for this operation")
        if any(type(data) is not bytes or not data or len(data) > MAX_DECODED for data in files.values()):
            raise ValueError("Input files must be nonempty and at most 12 MiB each")
        if kind in {"spatial", "ensemble_spatial"}:
            self._check_map(files["config"])
        if kind == "adaptive_spatial":
            settings = decode_json(files["config"]).get("refinement", {})
            self._number(settings, "max_depth", 5, (0, 8))
            self._number(settings, "max_leaves", 1024, (1, 2048))
        with self.lock:
            if len(self.store) >= self.capacity:
                raise ValueError("Dashboard workspace job limit reached")
            settings = self._options(kind, options)
            identifier = uuid4().hex
            folder = self.root / identifier
            folder.mkdir(exist_ok=False, mode=0o700)
            source = folder / ("input.jsonl" if kind == "replay" else "input.csv")
            for name, data in files.items():
                with atomic_output(source if name == "data" else folder / "config.json", binary=True) as file:
                    file.write(data)
            job = Job(identifier, kind, folder, settings, datetime.now(timezone.utc).isoformat())
            self._save(job)
            self.store[identifier] = job
            self.executor.submit(self._run, job, self._command(job))
            return identifier

    def _run(self, job, args):
        with self.lock:
            if job.cancel.is_set():
                self._status(job, "cancelled")
                return
            self._status(job, "running")
        if (job.kind == "forecast" and "steps" in job.options
                and self.store[job.options["model"]].kind != "quantile_train"):
            output_name = "forecast.csv"
        elif job.kind in {"inspect", "train", "search"}:
            output_name = "report.json"
        else:
            output_name = "stdout.log"
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
                terminal = ("cancelled" if job.cancel.is_set()
                            else "complete" if process.returncode == 0 else "failed")
                self._status(job, terminal)
        except OSError as exc:
            with (job.folder / "failure.log").open("a", encoding="utf-8") as error:
                error.write(f"\nDashboard could not run this job: {exc}\n")
            with self.lock:
                self._status(job, "failed")

    def cancel(self, identifier):
        with self.lock:
            job = self.store[identifier]
            if job.status in IN_PROGRESS:
                job.cancel.set()

    def resume(self, identifier):
        with self.lock:
            job = self.store[identifier]
            if not self._can_resume(job):
                raise ValueError("A stopped training run or search with a saved checkpoint is required")
            args = self._command(job)
            job.cancel.clear()
            self._status(job, "queued")
            self.executor.submit(self._run, job, args)

    @staticmethod
    def _can_resume(job):
        if job.status not in FINAL - {"complete"}:
            return False
        if job.kind == "train":
            return (job.folder / "resume.json").is_file()
        if job.kind == "search":
            return any((job.folder / "search").glob("*.json"))
        return False

    @staticmethod
    def _progress(job):
        if job.status not in IN_PROGRESS:
            return None
        if job.kind == "train":
            state = job.folder / "resume.json"
            if state.is_file():
                record = read_json(state)
                return {"stage": "epoch", "completed": record["epoch"],
                        "total": record["recipe"]["settings"]["epochs"]}
        if job.kind == "direct_train":
            state = job.folder / "direct-progress.json"
            if state.is_file():
                return {"stage": "epoch", **read_json(state)}
        if job.kind == "search":
            done = {}
            for name in ("classical_ga", "quantum_inspired", "random", "hybrid_amplitude"):
                state = job.folder / "search" / f"{name}.json"
                if state.is_file():
                    done[name] = len(read_json(state)["history"])
            return {"stage": "search", "generations": done, "total": job.options["generations"]}
        return None

    @staticmethod
    def _available(job, name):
        return (name == "failure.log" and job.status == "failed"
                or name in ARTIFACTS[job.kind]
                and (job.status == "complete" or job.kind == "live"
                     and job.status in {"cancelled", "interrupted"}))

    def snapshot(self):
        with self.lock:
            records = []
            for job in reversed(self.store.values()):
                artifacts = []
                for name in ARTIFACTS[job.kind] | {"failure.log"}:
                    if self._available(job, name):
                        try:
                            self._contained_file(job, name)
                        except KeyError:
                            continue
                        artifacts.append(name)
                records.append({"id": job.identifier, "kind": job.kind, "status": job.status,
                                "created_at": job.created_at, "label": job.options.get("label"),
                                "can_resume": self._can_resume(job),
                                "forecast_horizon": job.options.get("horizon")
                                if job.kind in {"direct_train", "quantile_train", "blend"} else None,
                                "progress": self._progress(job), "artifacts": sorted(artifacts),
                                "can_export": bool(RESEARCH_ARTIFACTS.get(job.kind, set()) & set(artifacts))})
            return records

    def artifact(self, identifier, name):
        with self.lock:
            job = self.store[identifier]
            if self._available(job, name):
                return self._contained_file(job, name)
        raise KeyError(name)

    def research_report(self, identifier):
        with self.lock:
            job = self.store[identifier]
            if job.status != "complete" or job.kind not in RESEARCH_ARTIFACTS:
                raise KeyError(identifier)
            return job.kind, self._contained_file(job, "report.json")

    def live_path(self, identifier):
        with self.lock:
            job = self.store[identifier]
            if job.kind not in {"live", "replay"}:
                raise KeyError(identifier)
            file = job.folder / "session/readings.jsonl"
            if file.resolve() != file:
                raise KeyError(identifier)
            return file

    def close(self):
        with self.lock:
            for job in self.store.values():
                if job.status in IN_PROGRESS:
                    job.cancel.set()
        self.executor.shutdown()
        self.lease.close()
