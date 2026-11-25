"""Capture raw serial lines before parsing them into reading blocks."""
from Acquisition.DataLogger import log_raw, write_json
from Acquisition.PacketParser import PacketParser
from Acquisition.SessionManager import session


def _save_reading(file, reading):
    write_json(file, {"received_at": reading.received_at.isoformat(),
                      "measurements": reading.measurements, "flags": reading.flags})


def capture(lines, output, stop=None):
    parser = PacketParser()
    timestamp = None
    with session(output) as (raw, readings):
        try:
            for timestamp, line in lines:
                if stop is not None and stop.is_set():
                    break
                log_raw(raw, timestamp, line)
                reading = parser.feed(line.decode("utf-8", errors="replace"), timestamp)
                if reading is not None:
                    _save_reading(readings, reading)
        finally:
            if timestamp is not None:
                partial = parser.finish(timestamp)
                if partial is not None:
                    _save_reading(readings, partial)
