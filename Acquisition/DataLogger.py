"""Raw byte-preserving logging and deterministic replay."""
import base64
from datetime import datetime, timezone
import json
from pathlib import Path

from Acquisition.Archives import raw_records


def write_json(file, record):
    file.write(json.dumps(record, allow_nan=False) + "\n")
    file.flush()


def log_raw(file, timestamp, line):
    write_json(file, {"received_at": timestamp.isoformat(),
                      "line_b64": base64.b64encode(line).decode("ascii")})


def replay(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        yield from raw_records(path)
        return
    with path.open("rb") as file:
        for line in file:
            yield datetime.now(timezone.utc), line
