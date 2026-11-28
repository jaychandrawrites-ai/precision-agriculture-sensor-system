"""Explicit unit conversions for manual CSV surveys and deterministic raw replay."""
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from io import StringIO
from itertools import chain
import csv
from uuid import uuid4
from zoneinfo import ZoneInfo
from collections import Counter

from Acquisition.StationaryGPS import stationary_location

from Acquisition.AquaMap import AquaMapParser
from Acquisition.AITrace import _json, _label, _time
from Acquisition.DataLogger import replay
from Acquisition.MappedPackets import MappedPacketParser, PacketSchema
from Acquisition.PacketParser import PacketParser



def _csv_time(value, timezone):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        if timezone is None:
            raise ValueError("Naive CSV timestamps require an explicit timezone")
        # Reject ambiguous folds and nonexistent wall clocks.
        first = stamp.replace(tzinfo=timezone)
        second = stamp.replace(tzinfo=timezone, fold=1)
        if first.utcoffset() != second.utcoffset():
            raise ValueError("Ambiguous or nonexistent local timestamp. Supply an explicit offset.")
        stamp = first
    return _time(stamp.isoformat())

def _csv_archive(payload, label, mapping):
    """Import separate segments using named columns and the verified packet unit registry."""
    text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
    rows = csv.DictReader(StringIO(text))
    if not rows.fieldnames or len(rows.fieldnames) != len(set(rows.fieldnames)):
        raise ValueError("CSV headers must be present and unique")
    location_col = mapping.get("location")
    segment_col = mapping.get("segment")
    if not location_col and not mapping.get("location_label"):
        raise ValueError("Supply a location column or a location label")
    zone = ZoneInfo(mapping["timezone"]) if mapping.get("timezone") else None
    time_col = mapping.get("received_at")
    observed_col = mapping.get("observed_at")
    if not time_col and not observed_col:
        raise ValueError("Map an explicit receipt or observation time column")
    fields = mapping["fields"]
    expected = {spec["column"] for spec in fields.values()}
    expected.update(col for col in (location_col, segment_col) if col)
    expected.update(col for col in (time_col, observed_col, mapping.get("device_id"), mapping.get("point")) if col)
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
        place = _label(row[location_col] if location_col else mapping["location_label"])
        split = _label(row[segment_col] if segment_col else "CSV visit")
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
        received = _csv_time(row[time_col], zone) if time_col else stamp
        packet = {name: row[spec["column"]] or None for name, spec in fields.items()}
        parsed = asdict(parser.feed(_json(packet), datetime.fromisoformat(received)))
        parsed["received_at"] = received
        parsed["observed_at"] = _csv_time(row[observed_col], zone) if observed_col and row[observed_col] else None
        if mapping.get("device_id"):
            parsed["device_id"] = row[mapping["device_id"]] or None
        if mapping.get("point"):
            parsed["source_point"] = row[mapping["point"]]
        parsed["flags"] = [*parsed["flags"], "manual_csv", *([] if time_col else ["receipt_is_import_time"])]
        segment["started_at"] = min(received, segment["started_at"] or received)
        segment["ended_at"] = max(received, segment["ended_at"] or received)
        segment["records"].append({"received_at": received, "raw_b64": None, "reading": parsed})
    if not locations:
        raise ValueError("CSV contains no data rows")
    reports = []
    for location in locations.values():
        for segment in location["segments"]:
            records = [item["reading"] for item in segment["records"]]
            clock = "observed_at" if observed_col else "received_at"
            counts = Counter(item[clock] for item in records if item[clock] is not None)
            ordered = sorted(map(datetime.fromisoformat, counts))
            gaps = [(b-a).total_seconds() for a, b in zip(ordered, ordered[1:])]
            points = [(item["measurements"].get("latitude_deg"), item["measurements"].get("longitude_deg"))
                      for item in records]
            gps = stationary_location(points, **mapping.get("gps", {}))
            for item in records:
                if counts.get(item[clock], 0) > 1:
                    item["flags"].append("duplicate_csv_timestamp")
            diagnostics = {"records": len(records), "duplicate_timestamp_rows": sum(n for n in counts.values() if n > 1),
                           "max_gap_seconds": max(gaps, default=0), "gps": gps,
                           "source_sha256": sha256(payload if isinstance(payload, bytes) else payload.encode()).hexdigest(),
                           "data_use": "inference_only"}
            segment["calibration"]["csv_quality"] = diagnostics
            if ordered:
                reports.append((ordered[0], ordered[-1], diagnostics))
    for i, (start, end, quality) in enumerate(reports):
        quality["overlapping_visits"] = sum(other_start <= end and start <= other_end
                                             for j, (other_start, other_end, _) in enumerate(reports) if i != j)
    archive["survey"]["locations"] = list(locations.values())
    archive["sha256"] = sha256(_json(archive).encode()).hexdigest()
    return archive


def import_csv(store, payload, label, mapping):
    return store.restore(_csv_archive(payload, label, mapping))


_CLEANED_FIELDS = {
    "latitude_deg": ("latitude", "deg"), "longitude_deg": ("longitude", "deg"),
    "gps_satellites": ("gps_sats", "count"), "soil_moisture_pct": ("soil_moisture", "%"),
    "soil_temperature_c": ("soil_temp", "C"), "soil_ec_us_cm": ("ec", "uS/cm"),
    "soil_ph": ("ph", "pH"), "air_temperature_c": ("air_temp", "C"),
    "relative_humidity_pct": ("air_humidity", "%"), "leaf_mv": ("leaf_wetness", "mV"),
    "radio_rssi_dbm": ("radio_rssi", "dBm"), "radio_snr_db": ("radio_snr", "dB"),
}


def import_csv_batch(store, files, label, timezone, *, preview=False):
    """Validate every file before publishing one combined, inference-only survey."""
    from pathlib import Path
    from zoneinfo import ZoneInfoNotFoundError

    if not timezone or not 1 <= len(files) <= 32:
        raise ValueError("Supply an explicit timezone and one to 32 compatible CSVs")
    combined = None
    labels = set()
    visits = []
    for filename, payload, alias in files:
        name = _label(alias or Path(filename).stem)
        if name in labels:
            raise ValueError("Use a distinct location label for each file")
        labels.add(name)
        mapping = {"timezone": timezone, "observed_at": "timestamp", "location_label": name, "point": "point",
                   "fields": {key: {"column": column, "unit": unit}
                              for key, (column, unit) in _CLEANED_FIELDS.items()}}
        try:
            archive = _csv_archive(payload, label, mapping)
        except ZoneInfoNotFoundError as error:
            raise ValueError("Unknown IANA timezone") from error
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        location = archive["survey"]["locations"][0]
        segment = location["segments"][0]
        quality = segment["calibration"]["csv_quality"]
        readings = [row["reading"] for row in segment["records"]]
        stamps = sorted(row["observed_at"] for row in readings if row["observed_at"])
        quality.update(observed_start=stamps[0] if stamps else None,
                       observed_end=stamps[-1] if stamps else None,
                       missing_fields=dict(Counter(flag.removeprefix("missing:") for row in readings
                                                   for flag in row["flags"] if flag.startswith("missing:"))))
        visits.append((name, quality))
        if combined is None:
            combined = archive
        else:
            combined["survey"]["locations"].append(location)
    for name, quality in visits:
        start, end = quality["observed_start"], quality["observed_end"]
        quality["overlapping_visits"] = sum(
            other != name and start is not None and item["observed_start"] is not None
            and item["observed_start"] <= end and start <= item["observed_end"]
            for other, item in visits)
    if preview:
        return [{**quality, "location_label": name,
                 "gps": {key: value for key, value in quality["gps"].items() if key != "location"}}
                for name, quality in visits]
    del combined["sha256"]
    combined["sha256"] = sha256(_json(combined).encode()).hexdigest()
    return store.restore(combined)


def replay_segment(store, survey_id, source, *, label=None, location_id=None, schema=None):
    parser = (AquaMapParser() if schema == "aquamap_rx_csv" else
              PacketParser() if schema is None else MappedPacketParser(PacketSchema.from_dict(schema)))
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
