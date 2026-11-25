"""Immutable, content-addressed catalogs of locally prepared hourly datasets."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from azmet import prepare
from config import FEATURES
from Data.AZMET import AZMETSchema
from Data.StationAnalysis import compare_reference_et
from Data.Studies import distribution_profile
from Utils.Serialization import atomic_output, read_json, write_json


FORMAT = "jupiter-dataset-v1"
ARTIFACTS = ("hourly.csv", "flags.csv", "report.json", "mapping.json")
SOURCE_MODULES = ("Data", "Physics", "Numerical")
IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")
VERSION = re.compile(r"[0-9a-f]{64}\Z")


def _checksum(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_json(record):
    return sha256(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def _producer_hash():
    root = Path(__file__).resolve().parents[1]
    sources = [root / name for name in ("azmet.py", "assumptions.py", "config.py")]
    for module in SOURCE_MODULES:
        sources.extend(sorted((root / module).rglob("*.py")))
    return _digest_json({"numpy": np.__version__, "pandas": pd.__version__,
                         "sources": {path.relative_to(root).as_posix(): _checksum(path)
                                     for path in sources}})


@dataclass(frozen=True)
class DatasetReference:
    station: str
    version: str

    def __post_init__(self):
        if not IDENTIFIER.fullmatch(self.station) or not VERSION.fullmatch(self.version):
            raise ValueError("Invalid station or version identifier")

    @classmethod
    def parse(cls, token):
        fields = token.split("/")
        if len(fields) != 2:
            raise ValueError("Use STATION/VERSION as printed by the registry")
        return cls(*fields)

    def __str__(self):
        return f"{self.station}/{self.version}"


def _identity(station, source_hash, mapping_hash, producer):
    return {"format": FORMAT, "station_id": station, "source_sha256": source_hash,
            "mapping_sha256": mapping_hash, "producer_sha256": producer}


class DatasetRegistry:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def _path(self, reference):
        path = self.root / reference.station / reference.version
        if path.resolve() != path:
            raise ValueError("Registry versions cannot be symbolic links")
        return path

    def import_archive(self, source, mapping):
        """Prepare one archive atomically; identical imports reuse a verified version."""
        station = mapping.get("station_id")
        if not isinstance(station, str) or not IDENTIFIER.fullmatch(station):
            raise ValueError("Mapping requires a short lowercase station_id")
        schema = AZMETSchema.from_dict(mapping["schema"])
        timezone = schema.timezone or "UTC"
        ZoneInfo(timezone)
        payload = Path(source).read_bytes()
        source_hash = sha256(payload).hexdigest()
        expected = mapping.get("source_sha256")
        if expected is not None and expected != source_hash:
            raise ValueError("Archive hash differs from the verified mapping")
        identity = _identity(station, source_hash, _digest_json(mapping), _producer_hash())
        for version in self.references(station=station):
            recorded = read_json(self._path(version) / "manifest.json")
            if all(recorded.get(key) == value for key, value in identity.items()):
                self.verify(version)
                return version
        station_directory = self.root / station
        if station_directory.resolve() != station_directory:
            raise ValueError("Registry stations cannot be symbolic links")
        station_directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".import-", dir=station_directory) as temporary:
            staging = Path(temporary)
            original = staging / "source"
            original.write_bytes(payload)
            features, flags, report = prepare(original, mapping)
            report["distribution"] = distribution_profile(features, timezone=timezone)
            if reference_column := mapping.get("station_eto_mm_column"):
                report["station_et_comparison"] = compare_reference_et(
                    original, schema, features, reference_column)
            features.to_csv(staging / "hourly.csv", lineterminator="\n")
            flags.to_csv(staging / "flags.csv", lineterminator="\n")
            write_json(staging / "report.json", report)
            write_json(staging / "mapping.json", mapping)
            original.unlink()
            artifact_hashes = {name: _checksum(staging / name) for name in ARTIFACTS}
            reference = DatasetReference(station, _digest_json({**identity, "artifacts": artifact_hashes}))
            destination = self._path(reference)
            manifest = {**identity, "version": reference.version, "timezone": timezone,
                        "source_url": mapping.get("source_url"), "features": list(FEATURES),
                        "synthetic": bool(mapping.get("synthetic", False)),
                        "artifacts": artifact_hashes}
            write_json(staging / "manifest.json", manifest)
            try:
                staging.rename(destination)
            except OSError:
                if not destination.exists():
                    raise
                self.verify(reference)
        return reference

    def verify(self, reference, source=None):
        folder = self._path(reference)
        manifest = read_json(folder / "manifest.json")
        expected = _identity(reference.station, manifest["source_sha256"],
                             manifest["mapping_sha256"], manifest["producer_sha256"])
        if (any(manifest.get(key) != value for key, value in expected.items())
                or manifest["version"] != reference.version
                or _digest_json({**expected, "artifacts": manifest["artifacts"]}) != reference.version
                or manifest["features"] != list(FEATURES)):
            raise ValueError("Dataset manifest does not match its version")
        if set(manifest["artifacts"]) != set(ARTIFACTS):
            raise ValueError("Manifest artifact list is incomplete")
        for name in ARTIFACTS:
            path = folder / name
            if path.is_symlink() or _checksum(path) != manifest["artifacts"][name]:
                raise ValueError(f"Dataset artifact changed: {name}")
        mapping = read_json(folder / "mapping.json")
        if (_digest_json(mapping) != manifest["mapping_sha256"]
                or manifest["timezone"] != (AZMETSchema.from_dict(mapping["schema"]).timezone or "UTC")
                or manifest["synthetic"] != bool(mapping.get("synthetic", False))
                or manifest["source_url"] != mapping.get("source_url")):
            raise ValueError("Version mapping hash changed")
        if source is not None and _checksum(source) != manifest["source_sha256"]:
            raise ValueError("Supplied source does not match the archived source hash")
        return manifest

    def references(self, *, station=None):
        if not self.root.is_dir():
            return []
        return [DatasetReference(folder.name, version.name)
                for folder in sorted(self.root.iterdir()) if folder.is_dir()
                and not folder.is_symlink() and IDENTIFIER.fullmatch(folder.name)
                and (station is None or folder.name == station)
                for version in sorted(folder.iterdir()) if version.is_dir()
                and not version.is_symlink() and VERSION.fullmatch(version.name)]

    def load(self, reference):
        manifest = self.verify(reference)
        folder = self._path(reference)
        features = pd.read_csv(folder / "hourly.csv", index_col="timestamp")
        features.index = pd.to_datetime(features.index, format="ISO8601", utc=True)
        if (list(features.columns) != manifest["features"] or features.empty
                or features.index.has_duplicates or not features.index.is_monotonic_increasing
                or not (features.index == features.index.floor("h")).all()
                or not (np.diff(features.index.to_numpy(dtype="datetime64[ns]")) == np.timedelta64(1, "h")).all()):
            raise ValueError("Prepared data does not satisfy the hourly feature contract")
        return features, manifest

    def export(self, reference, destination, *, start=None, end=None):
        features, _ = self.load(reference)
        for boundary, following in ((start, True), (end, False)):
            if boundary is None:
                continue
            timestamp = pd.Timestamp(boundary)
            if timestamp.tzinfo is None:
                raise ValueError("Specify explicit timezone offsets for export boundaries")
            index = features.index
            features = features[(index >= timestamp) if following else (index < timestamp)]
        if features.empty:
            raise ValueError("No hours in the requested half-open interval")
        with atomic_output(destination) as stream:
            features.to_csv(stream, lineterminator="\n")
