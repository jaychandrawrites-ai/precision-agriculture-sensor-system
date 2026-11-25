"""Summarize timestamp, sensor coverage and sequence diagnostics from stored sessions."""
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path


def session_health(path):
    counts = defaultdict(Counter)
    series = defaultdict(list)
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            name = row.get("device_id") or "unidentified"
            summary = counts[name]
            summary["packets"] += 1
            for flag in row["flags"]:
                if flag.startswith("sequence_gap:"):
                    summary["sequence_gaps_detected"] += 1
                    summary["estimated_missing_packets"] += int(flag.split(":")[1])
                else:
                    summary[f"flag:{flag}"] += 1
            if row["measurements"]:
                summary["packets_with_measurements"] += 1
                for measurement in row["measurements"]:
                    summary[f"reading:{measurement}"] += 1
            stamp = datetime.fromisoformat(row["received_at"])
            if stamp.utcoffset() is None:
                raise ValueError("Session timestamps require UTC offsets")
            series[name].append(stamp)
    return {name: {"counts": dict(count), "first_receipt": min(series[name]).isoformat(),
                   "last_receipt": max(series[name]).isoformat()}
            for name, count in counts.items()}
