"""Recover a session into a separate directory without changing its source."""
from dataclasses import asdict
from pathlib import Path

from Acquisition.AcquisitionController import capture
from Acquisition.Archives import RawArchiveError, inspect_raw, raw_records
from Acquisition.MappedPackets import MappedPacketParser, PacketSchema
from Acquisition.PacketParser import PacketParser
from Utils.Serialization import read_json


def _parser(source, packet_schema):
    stored = source / "session.json"
    metadata = read_json(stored) if stored.is_file() else {}
    selection = packet_schema if packet_schema is not None else metadata.get("parser")
    if selection == "checked_in_arduino_text":
        return PacketParser(), selection
    if isinstance(selection, dict):
        schema = PacketSchema.from_dict(selection)
        return MappedPacketParser(schema), asdict(schema)
    raise ValueError("Specify a packet schema or explicitly select the checked-in text parser")


def recover_session(source, output, *, packet_schema=None, accept_torn_tail=False):
    source, output = Path(source), Path(output)
    raw = source / "raw.jsonl"
    parser, schema = _parser(source, packet_schema)
    summary = inspect_raw(raw, accept_torn_tail=accept_torn_tail)
    manifest_path = source / "integrity.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("version") != 1:
            raise RawArchiveError("Source integrity manifest has an unsupported format")
        if manifest.get("status") in ("complete", "stopped") and manifest.get("raw") != asdict(summary):
            raise RawArchiveError("Completed session raw archive disagrees with its integrity manifest")
    if not summary.records:
        raise RawArchiveError("Raw session contains no complete records to recover")
    metadata = {"transport": "verified_raw_recovery", "parser": schema,
                "source_raw": asdict(summary)}
    capture(raw_records(raw, accept_torn_tail=accept_torn_tail, expected_sha256=summary.sha256),
            output, parser=parser, metadata=metadata)
    return metadata
