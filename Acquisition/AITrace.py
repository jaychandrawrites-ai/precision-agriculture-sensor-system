"""Durable, local multi-location acquisition with independently finalized segments."""
import base64
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
import re
import sqlite3
from uuid import uuid4


SCHEMA = """
CREATE TABLE IF NOT EXISTS survey (
    id TEXT PRIMARY KEY, label TEXT NOT NULL, created_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS location (
    id TEXT PRIMARY KEY, survey_id TEXT NOT NULL REFERENCES survey(id), label TEXT NOT NULL,
    latitude_deg REAL, longitude_deg REAL, coordinate_source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS segment (
    id TEXT PRIMARY KEY, location_id TEXT NOT NULL REFERENCES location(id),
    started_at TEXT NOT NULL, ended_at TEXT, status TEXT NOT NULL, calibration TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS packet (
    id INTEGER PRIMARY KEY, segment_id TEXT NOT NULL REFERENCES segment(id),
    received_at TEXT NOT NULL, raw BLOB, parsed TEXT, CHECK(raw IS NOT NULL OR parsed IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS packet_segment ON packet(segment_id, id);
CREATE INDEX IF NOT EXISTS location_survey ON location(survey_id);
CREATE INDEX IF NOT EXISTS segment_location ON segment(location_id);
"""


def _time(value=None):
    stamp = datetime.now(timezone.utc) if value is None else value
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp)
    if not isinstance(stamp, datetime) or stamp.utcoffset() is None:
        raise ValueError("Recorded timestamps must include a UTC offset")
    return stamp.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _label(value):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 120:
        raise ValueError("Use a nonempty label of at most 120 characters")
    return value.strip()


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("Invalid survey, location or segment identifier")
    return value


def _coordinates(coords):
    if coords is None:
        return None, None, "unavailable"
    if (not isinstance(coords, (list, tuple)) or len(coords) != 2
            or not all(type(x) in (int, float) and math.isfinite(x) for x in coords)
            or not -90 <= coords[0] <= 90 or not -180 <= coords[1] <= 180):
        raise ValueError("Provide a valid manual latitude and longitude pair")
    return *coords, "manual"


def _reading(parsed, timestamp):
    if parsed is None:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("measurements"), dict):
        raise ValueError("A parsed record needs a measurements object")
    flags = parsed.get("flags", [])
    if (not isinstance(flags, (tuple, list)) or not all(isinstance(flag, str) for flag in flags)
            or not all(isinstance(key, str) and type(value) in (int, float) and math.isfinite(value)
                       for key, value in parsed["measurements"].items())):
        raise ValueError("Measurements must be finite numbers and quality flags must be strings")
    output = {**parsed, "received_at": _time(parsed.get("received_at", timestamp)), "flags": flags}
    if output.get("observed_at") is not None:
        output["observed_at"] = _time(output["observed_at"])
    return _json(output)


class TraceStore:
    """One application owns a workspace. SQLite commits each transition and packet."""
    def __init__(self, root, *, recover=True):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "trace.sqlite3"
        with self._database() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            if recover:
                db.execute("""UPDATE segment SET status='interrupted', ended_at=MAX(started_at, COALESCE(
                              (SELECT MAX(received_at) FROM packet WHERE segment_id=segment.id), started_at))
                              WHERE status='recording'""")

    @contextmanager
    def _database(self, write=False):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                if write:
                    db.execute("BEGIN IMMEDIATE")
                yield db
        finally:
            db.close()

    def create(self, label):
        identifier = uuid4().hex
        with self._database(write=True) as db:
            db.execute("INSERT INTO survey VALUES (?, ?, ?, NULL)", (identifier, _label(label), _time()))
        return self.status(identifier)

    def list_surveys(self):
        with self._database() as db:
            return [dict(row) for row in db.execute("SELECT * FROM survey ORDER BY created_at, rowid")]

    def _survey(self, db, identifier):
        row = db.execute("SELECT * FROM survey WHERE id=?", (_id(identifier),)).fetchone()
        if row is None:
            raise KeyError("Survey not found")
        return dict(row)

    def _active(self, db, identifier):
        return db.execute("""SELECT s.id, s.location_id FROM segment s JOIN location l ON l.id=s.location_id
                             WHERE l.survey_id=? AND s.status='recording'""", (identifier,)).fetchone()

    def status(self, identifier):
        with self._database() as db:
            db.execute("BEGIN")
            return self._status(db, identifier)

    def _status(self, db, identifier):
        survey = self._survey(db, identifier)
        rows = db.execute("""SELECT l.* FROM location l WHERE survey_id=? ORDER BY rowid""", (identifier,))
        locations = [dict(row) for row in rows]
        active_segment = active_location = None
        for location in locations:
            segments = db.execute("""SELECT s.*, COUNT(p.raw) AS raw_count, COUNT(p.parsed) AS sample_count
                                     FROM segment s LEFT JOIN packet p ON p.segment_id=s.id
                                     WHERE s.location_id=? GROUP BY s.id ORDER BY s.rowid""", (location["id"],))
            location["segments"] = [dict(row) for row in segments]
            for segment in location["segments"]:
                segment.pop("location_id")
                segment["calibration"] = json.loads(segment["calibration"])
                if segment["status"] == "recording":
                    active_segment, active_location = segment["id"], location["id"]
        return {**survey, "locations": locations, "active_segment_id": active_segment,
                "active_location_id": active_location}

    def start(self, identifier, label=None, location_id=None, coordinates=None, calibration=None, *, recorded_at=None):
        if calibration is not None and not isinstance(calibration, dict):
            raise ValueError("Calibration metadata must be an object")
        with self._database(write=True) as db:
            survey = self._survey(db, identifier)
            if survey["finished_at"] or self._active(db, identifier):
                raise ValueError("Finish the active segment before starting another; finished surveys are read-only")
            if location_id is None:
                location_id = uuid4().hex
                db.execute("INSERT INTO location VALUES (?, ?, ?, ?, ?, ?)",
                           (location_id, identifier, _label(label), *_coordinates(coordinates)))
            else:
                if label is not None or coordinates is not None:
                    raise ValueError("Revisiting preserves the original location label and coordinates")
                row = db.execute("SELECT id FROM location WHERE id=? AND survey_id=?",
                                 (_id(location_id), identifier)).fetchone()
                if row is None:
                    raise KeyError("Location not found in this survey")
            db.execute("INSERT INTO segment VALUES (?, ?, ?, NULL, 'recording', ?)",
                       (uuid4().hex, location_id, _time(recorded_at), _json(calibration or {})))
        return self.status(identifier)

    def _stop(self, db, identifier, interrupted, recorded_at=None):
        active = self._active(db, identifier)
        if active is not None:
            db.execute("UPDATE segment SET status=?, ended_at=? WHERE id=?",
                       ("interrupted" if interrupted else "complete", _time(recorded_at), active["id"]))

    def stop(self, identifier, interrupted=False, *, recorded_at=None):
        with self._database(write=True) as db:
            self._survey(db, identifier)
            self._stop(db, identifier, interrupted, recorded_at)
        return self.status(identifier)

    def finish(self, identifier):
        with self._database(write=True) as db:
            self._survey(db, identifier)
            self._stop(db, identifier, False)
            db.execute("UPDATE survey SET finished_at=COALESCE(finished_at, ?) WHERE id=?", (_time(), identifier))
        return self.status(identifier)

    def ingest(self, identifier, raw, received_at, parsed=None):
        if raw is not None and not isinstance(raw, bytes):
            raise ValueError("Raw packets must be bytes")
        if raw is None and parsed is None:
            raise ValueError("Expected raw bytes or a parsed reading")
        stamp = _time(received_at)
        text = _reading(parsed, stamp)
        with self._database(write=True) as db:
            self._survey(db, identifier)
            active = self._active(db, identifier)
            if active is None:
                return False
            db.execute("INSERT INTO packet(segment_id, received_at, raw, parsed) VALUES (?, ?, ?, ?)",
                       (active["id"], stamp, raw, text))
        return True

    def segment_readings(self, identifier, segment_id):
        with self._database() as db:
            segment = db.execute("""SELECT s.id FROM segment s JOIN location l ON l.id=s.location_id
                                    WHERE s.id=? AND l.survey_id=?""", (_id(segment_id), _id(identifier))).fetchone()
            if segment is None:
                raise KeyError("Segment not found in this survey")
            return [json.loads(row[0]) for row in db.execute(
                "SELECT parsed FROM packet WHERE segment_id=? AND parsed IS NOT NULL ORDER BY id", (segment_id,))]

    def export(self, identifier, format="json", location_id=None, segment_id=None):
        with self._database() as db:
            db.execute("BEGIN")
            survey = self._status(db, identifier)
            locations = []
            for location in survey["locations"]:
                if location_id is not None and location_id != location["id"]:
                    continue
                segments = []
                for segment in location["segments"]:
                    if segment_id is not None and segment_id != segment["id"]:
                        continue
                    records = []
                    for row in db.execute("SELECT received_at, raw, parsed FROM packet WHERE segment_id=? ORDER BY id",
                                          (segment["id"],)):
                        raw = row["raw"]
                        records.append({"received_at": row["received_at"],
                                        "raw_b64": None if raw is None else base64.b64encode(raw).decode("ascii"),
                                        "reading": None if row["parsed"] is None else json.loads(row["parsed"])})
                    segments.append({**segment, "records": records})
                if segments or segment_id is None:
                    locations.append({**location, "segments": segments})
            if not locations and (location_id is not None or segment_id is not None):
                raise KeyError("Location or segment not found in this survey")
        survey["locations"] = locations
        if format == "json":
            archive = {"format": "jupiter-ai-trace-v1", "survey": survey}
            archive["sha256"] = sha256(_json(archive).encode()).hexdigest()
            return (_json(archive) + "\n").encode()
        if format == "raw":
            return "".join(_json({"survey_id": identifier, "location_id": l["id"], "segment_id": s["id"],
                                 "received_at": r["received_at"], "line_b64": r["raw_b64"]}) + "\n"
                           for l in locations for s in l["segments"] for r in s["records"]
                           if r["raw_b64"] is not None).encode()
        if format != "csv":
            raise ValueError("Unknown export format")
        return self._csv(identifier, locations)

    @staticmethod
    def _csv_cell(value):
        if isinstance(value, str) and (value.lstrip().startswith(("=", "+", "-", "@"))
                                       or value.startswith(("\t", "\r", "\n"))):
            return "'" + value
        return value

    @staticmethod
    def _csv(identifier, locations):
        rows = []
        fields = ["survey_id", "location_id", "location_label", "segment_id", "received_at",
                  "observed_at", "device_id", "packet_id", "flags", "coordinate_source",
                  "manual_latitude_deg", "manual_longitude_deg"]
        measures = set()
        for location in locations:
            for segment in location["segments"]:
                for record in segment["records"]:
                    reading = record["reading"]
                    if reading is None:
                        continue
                    values = reading["measurements"]
                    measures.update(f"measurement_{key}" for key in values)
                    label = TraceStore._csv_cell(location["label"])
                    rows.append({
                        "survey_id": identifier, "location_id": location["id"], "location_label": label,
                        "segment_id": segment["id"], "received_at": reading["received_at"],
                        **{key: TraceStore._csv_cell(reading.get(key)) for key in ("observed_at", "device_id", "packet_id")},
                        "flags": _json(reading["flags"]), "coordinate_source": location["coordinate_source"],
                        "manual_latitude_deg": location["latitude_deg"], "manual_longitude_deg": location["longitude_deg"],
                        **{f"measurement_{key}": value for key, value in values.items()},
                    })
        output = StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields + sorted(measures))
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue().encode()

    def restore(self, payload):
        value = json.loads(payload) if isinstance(payload, (bytes, str)) else payload
        if not isinstance(value, dict) or value.get("format") != "jupiter-ai-trace-v1":
            raise ValueError("Unsupported survey archive")
        body = {key: item for key, item in value.items() if key != "sha256"}
        if sha256(_json(body).encode()).hexdigest() != value.get("sha256"):
            raise ValueError("Survey archive checksum mismatch")
        survey = value["survey"]
        identifier = _id(survey["id"])
        try:
            with self._database(write=True) as db:
                db.execute("INSERT INTO survey VALUES (?, ?, ?, ?)",
                           (identifier, _label(survey["label"]), _time(survey["created_at"]),
                            None if survey["finished_at"] is None else _time(survey["finished_at"])))
                for location in survey["locations"]:
                    coords = (None if location["coordinate_source"] == "unavailable"
                              else [location["latitude_deg"], location["longitude_deg"]])
                    location_id = _id(location["id"])
                    db.execute("INSERT INTO location VALUES (?, ?, ?, ?, ?, ?)",
                               (location_id, identifier, _label(location["label"]), *_coordinates(coords)))
                    for segment in location["segments"]:
                        status = segment["status"]
                        if status not in {"recording", "complete", "interrupted"}:
                            raise ValueError("Invalid recording state")
                        segment_id = _id(segment["id"])
                        started = _time(segment["started_at"])
                        ended = (_time(segment["ended_at"]) if segment["ended_at"] else
                                 max((r["received_at"] for r in segment["records"]), default=started))
                        db.execute("INSERT INTO segment VALUES (?, ?, ?, ?, ?, ?)",
                                   (segment_id, location_id, started, _time(ended),
                                    "interrupted" if status == "recording" else status,
                                    _json(segment["calibration"])))
                        for record in segment["records"]:
                            stamp = _time(record["received_at"])
                            raw = record["raw_b64"]
                            db.execute("INSERT INTO packet(segment_id,received_at,raw,parsed) VALUES (?,?,?,?)",
                                       (segment_id, stamp, None if raw is None else base64.b64decode(raw, validate=True),
                                        _reading(record["reading"], stamp)))
        except sqlite3.IntegrityError as exc:
            raise ValueError("Archive has duplicate identifiers or invalid records") from exc
        return self.status(identifier)
