"""Capture the checked-in Arduino text format or replay a saved session."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import signal
from threading import Event

from Acquisition.AcquisitionController import capture
from Acquisition.DataLogger import replay
from Acquisition.MappedPackets import MappedPacketParser, PacketSchema
from Acquisition.SerialReader import serial_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--port")
    source.add_argument("--replay", type=Path)
    parser.add_argument("--packet-schema", type=Path)
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    stop = Event()

    def request_stop(*_):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    schema = None
    if args.packet_schema:
        schema = PacketSchema.from_dict(json.loads(args.packet_schema.read_text(encoding="utf-8")))
    parser = MappedPacketParser(schema) if schema is not None else None
    metadata = {"transport": "serial" if args.port else "replay",
                "parser": asdict(schema) if schema is not None else "checked_in_arduino_text"}
    if args.port:
        capture(serial_lines(args.port, args.baud, stop), args.output, parser=parser,
                metadata=metadata)
    else:
        capture(replay(args.replay), args.output, stop, parser=parser, metadata=metadata)


if __name__ == "__main__":
    main()
