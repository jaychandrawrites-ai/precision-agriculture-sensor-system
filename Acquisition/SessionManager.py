"""Create acquisition sessions with atomic, auditable lifecycle records."""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from Acquisition.Archives import inspect_raw, inspect_readings
from Acquisition.DataLogger import write_json as write_line
from Utils.Serialization import write_json


@contextmanager
def session(output, *, metadata=None, stop=None):
    directory = Path(output)
    directory.mkdir()
    manifest = directory / "integrity.json"
    state = {"version": 1, "status": "open",
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(manifest, state)
    try:
        if metadata is not None:
            with (directory / "session.json").open("x", encoding="utf-8") as file:
                write_line(file, {**metadata, "version": 1})
        with (directory / "raw.jsonl").open("x", encoding="utf-8") as raw:
            with (directory / "readings.jsonl").open("x", encoding="utf-8") as readings:
                yield raw, readings
    except BaseException as exc:
        # A derived-output failure must not prevent recovery from the raw source.
        state.update(status="failed", error_type=type(exc).__name__,
                     ended_at_utc=datetime.now(timezone.utc).isoformat())
        try:
            write_json(manifest, state)
        except OSError:
            pass
        raise
    else:
        state.update(status="stopped" if stop is not None and stop.is_set() else "complete",
                     ended_at_utc=datetime.now(timezone.utc).isoformat(),
                     raw=asdict(inspect_raw(directory / "raw.jsonl")),
                     readings=asdict(inspect_readings(directory / "readings.jsonl")))
        write_json(manifest, state)
