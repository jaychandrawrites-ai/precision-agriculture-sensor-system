"""Explicit unit conversions for manual CSV surveys and deterministic raw replay."""
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from io import StringIO
from itertools import chain
import csv
from uuid import uuid4

from Acquisition.AITrace import _json, _label, _time
from Acquisition.DataLogger import replay
from Acquisition.MappedPackets import MappedPacketParser, PacketSchema
from Acquisition.PacketParser import PacketParser


def import_csv(store, payload, label, mapping):
    """Import separate segments using named columns and the verified packet unit registry."""
    text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
    rows = csv.DictReader(StringIO(text))
    if not rows.fieldnames or len(rows.fieldnames) != len(set(rows.fieldnames)):
        raise ValueError("CSV headers must be present and unique")
    location_col = mapping["location"]
    segment_col = mapping["segment"]
    time_col = mapping.get("received_at")
    observed_col = mapping.get("observed_at")
    if not time_col and not observed_col:
        raise ValueError("Map an explicit receipt or observation time column")
    fields = mapping["fields"]
    expected = {location_col, segment_col, *(spec["column"] for spec in fields.values())}
    expected.update(col for col in (time_col, observed_col, mapping.get("device_id")) if col)
    if not expected <= set(rows.fieldnames):
        raise ValueError("Manual mapping refers to columns absent from the CSV")
    schema = PacketSchema.from_dict({"fields": {key: {"key": key, "unit": spec["unit"]}
                                               for key, spec in fields.items()}})
    parser = MappedPacketParser(schema)
    stamp, survey_id = _time(), uuid4().hex
    archive = {"format": "jupiter-ai-trace-v1", "survey": {
        "id": survey_id, "label": _label(label), "created_at": stamp, "finished_at": stamp, "locations": [],
    }}
    locations, segments = {}, {}
    for index, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Ragged CSV at row {index}")
        place, split = _label(row[location_col]), _label(row[segment_col])
        if place not in locations:
            locations[place] = {"id": uuid4().hex, "label": place, "latitude_deg": None,
                                "longitude_deg": None, "coordinate_source": "unavailable", "segments": []}
        if (place, split) not in segments:
            segment = {"id": uuid4().hex, "status": "complete", "started_at": None, "ended_at": None,
                       "calibration": {"source": "manual_csv", "original_segment": split,
                                       "field_mapping": fields}, "records": []}
            segments[place, split] = segment
            locations[place]["segments"].append(segment)
        segment = segments[place, split]
        received = _time(row[time_col]) if time_col else stamp
        packet = {name: row[spec["column"]] or None for name, spec in fields.items()}
        parsed = asdict(parser.feed(_json(packet), datetime.fromisoformat(received)))
        parsed["received_at"] = received
        parsed["observed_at"] = _time(row[observed_col]) if observed_col and row[observed_col] else None
        if mapping.get("device_id"):
            parsed["device_id"] = row[mapping["device_id"]] or None
        parsed["flags"] = [*parsed["flags"], "manual_csv", *([] if time_col else ["receipt_is_import_time"])]
        segment["started_at"] = min(received, segment["started_at"] or received)
        segment["ended_at"] = max(received, segment["ended_at"] or received)
        segment["records"].append({"received_at": received, "raw_b64": None, "reading": parsed})
    if not locations:
        raise ValueError("CSV contains no data rows")
    archive["survey"]["locations"] = list(locations.values())
    archive["sha256"] = sha256(_json(archive).encode()).hexdigest()
    return store.restore(archive)


def replay_segment(store, survey_id, source, *, label=None, location_id=None, schema=None):
    parser = PacketParser() if schema is None else MappedPacketParser(PacketSchema.from_dict(schema))
    records = iter(replay(source))
    first = next(records, None)
    if first is None:
        raise ValueError("Replay source contains no packets")
    last = first[0]
    store.start(survey_id, label=label, location_id=location_id, recorded_at=last,
                calibration={"ingestion_source": "raw_replay"})
    try:
        for received, raw in chain((first,), records):
            if received < last:
                raise ValueError("Raw replay receipt timestamps must be nondecreasing")
            last = received
            # Persist raw bytes before attempting to parse them.
            store.ingest(survey_id, raw, received)
            parsed = parser.feed(raw.decode("utf-8", errors="replace"), received)
            if parsed is not None:
                store.ingest(survey_id, None, received, store_record(parsed))
        last_reading = None if last is None else parser.finish(last)
        if last_reading is not None:
            store.ingest(survey_id, None, last, store_record(last_reading))
    except BaseException:
        store.stop(survey_id, interrupted=True, recorded_at=last)
        raise
    return store.stop(survey_id, recorded_at=last)


def store_record(reading):
    result = asdict(reading)
    result["received_at"] = _time(reading.received_at)
    result["observed_at"] = _time(reading.observed_at) if reading.observed_at else None
    return result
