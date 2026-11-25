"""Integrity reports for local raw logs and deterministic recovery sources."""
import base64
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from Utils.Serialization import decode_json, read_json


class RawArchiveError(ValueError):
    """The raw archive is malformed outside the explicitly recoverable tail."""


@dataclass(frozen=True)
class ArchiveFile:
    records: int
    bytes: int
    sha256: str
    incomplete_final_bytes: int = 0


def _receive_time(raw):
    stamp = datetime.fromisoformat(raw)
    if stamp.utcoffset() is None:
        raise ValueError("Raw record timestamps require a UTC offset")
    return stamp


def decode_raw_record(line):
    record = decode_json(line)
    if not isinstance(record, dict) or not {"received_at", "line_b64"} <= record.keys():
        raise ValueError("Raw record requires received_at and line_b64")
    stamp = _receive_time(record["received_at"])
    payload = base64.b64decode(record["line_b64"], validate=True)
    return stamp, payload


def _records(stream, accept_torn_tail=False, digest=None, stats=None):
    for number, row in enumerate(stream, 1):
        if digest is not None:
            digest.update(row)
        if stats is not None:
            stats[1] += len(row)
        try:
            record = decode_raw_record(row)
        except (ValueError, TypeError, RecursionError) as exc:
            if accept_torn_tail and not row.endswith(b"\n"):
                # File iteration leaves out the terminator only for its last record.
                if stats is not None:
                    stats[2] = len(row)
                return
            raise RawArchiveError(f"Invalid raw log record at line {number}") from exc
        if stats is not None:
            stats[0] += 1
        yield record


def raw_records(path, *, accept_torn_tail=False, expected_sha256=None):
    """Read records without rewriting bytes and optionally enforce the source digest."""
    digest = sha256() if expected_sha256 is not None else None
    with Path(path).open("rb") as stream:
        yield from _records(stream, accept_torn_tail, digest)
    if digest is not None and digest.hexdigest() != expected_sha256:
        raise RawArchiveError("Source raw archive changed after inspection")


def inspect_raw(path, *, accept_torn_tail=False):
    """Count accepted records and bytes, hashing every byte including an ignored tail."""
    digest = sha256()
    stats = [0, 0, 0]
    with Path(path).open("rb") as stream:
        for _ in _records(stream, accept_torn_tail, digest, stats):
            pass
    return ArchiveFile(stats[0], stats[1], digest.hexdigest(), stats[2])


def inspect_readings(path):
    """Summarize bytes and validate the required structure of derived JSONL rows."""
    digest = sha256()
    count = size = 0
    with Path(path).open("rb") as stream:
        for number, row in enumerate(stream, 1):
            digest.update(row)
            size += len(row)
            try:
                entry = decode_json(row)
                _receive_time(entry["received_at"])
                if not isinstance(entry["measurements"], dict) or not isinstance(entry["flags"], list):
                    raise ValueError("Invalid derived readings record")
            except (ValueError, TypeError, KeyError, RecursionError) as exc:
                raise RawArchiveError(f"Invalid readings record at line {number}") from exc
            count += 1
    return ArchiveFile(count, size, digest.hexdigest())


def inspect_session(directory, *, accept_torn_tail=False):
    """Verify the exact on-disk archives against a closed capture's integrity record."""
    directory = Path(directory)
    raw = asdict(inspect_raw(directory / "raw.jsonl", accept_torn_tail=accept_torn_tail))
    state_path = directory / "integrity.json"
    expected = None
    findings = []
    if state_path.is_file():
        try:
            expected = read_json(state_path)
            if not isinstance(expected, dict):
                raise ValueError("Expected a JSON object")
        except (ValueError, RecursionError):
            expected = None
            findings.append("invalid_integrity_manifest")
    readings = None
    if expected is None:
        if not findings:
            findings.append("missing_integrity_manifest")
    elif expected.get("version") != 1:
        findings.append("unsupported_integrity_version")
    elif expected.get("status") not in ("complete", "stopped"):
        findings.append(f"capture_status:{expected.get('status', 'unknown')}")
    if raw["incomplete_final_bytes"]:
        findings.append("incomplete_final_raw_record")
    if expected is not None and expected.get("raw") is not None and raw != expected["raw"]:
        findings.append("raw_manifest_mismatch")
    try:
        readings = asdict(inspect_readings(directory / "readings.jsonl"))
    except (OSError, RawArchiveError):
        findings.append("missing_or_invalid_readings")
    else:
        if expected is not None and expected.get("readings") is not None and readings != expected["readings"]:
            findings.append("readings_manifest_mismatch")
    if expected is not None and expected.get("status") in ("complete", "stopped"):
        if expected.get("raw") is None or expected.get("readings") is None:
            findings.append("missing_integrity_fields")
    return {"version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": expected.get("status") if expected else None,
            "verified": not findings, "findings": findings, "raw": raw, "readings": readings}
