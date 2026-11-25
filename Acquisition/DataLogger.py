"""Raw byte-preserving logging and deterministic replay."""
import base64
from datetime import datetime, timezone
import json


def write_json(file, record):
    file.write(json.dumps(record, allow_nan=False) + "\n")
    file.flush()


def log_raw(file, timestamp, line):
    write_json(file, {"received_at": timestamp.isoformat(),
                      "line_b64": base64.b64encode(line).decode("ascii")})


def replay(path):
    with open(path, "rb") as file:
        for line in file:
            timestamp = datetime.now(timezone.utc)
            if path.suffix == ".jsonl":
                entry = json.loads(line)
                timestamp = datetime.fromisoformat(entry["received_at"])
                line = base64.b64decode(entry["line_b64"], validate=True)
            yield timestamp, line
