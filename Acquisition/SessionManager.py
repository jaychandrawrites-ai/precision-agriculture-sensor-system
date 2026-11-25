"""Create a fresh output folder for each acquisition session."""
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def session(output):
    directory = Path(output)
    directory.mkdir()
    with (directory / "raw.jsonl").open("x", encoding="utf-8") as raw:
        with (directory / "readings.jsonl").open("x", encoding="utf-8") as readings:
            yield raw, readings
