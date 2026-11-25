"""Create a fresh output folder for each acquisition session."""
from contextlib import contextmanager
from pathlib import Path

from Acquisition.DataLogger import write_json


@contextmanager
def session(output, *, metadata=None):
    directory = Path(output)
    directory.mkdir()
    if metadata is not None:
        with (directory / "session.json").open("x", encoding="utf-8") as file:
            write_json(file, {"version": 1, **metadata})
    with (directory / "raw.jsonl").open("x", encoding="utf-8") as raw:
        with (directory / "readings.jsonl").open("x", encoding="utf-8") as readings:
            yield raw, readings
