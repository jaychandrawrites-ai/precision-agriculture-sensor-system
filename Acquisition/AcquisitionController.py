"""Capture raw serial lines before parsing them into reading blocks."""
from Acquisition.DataLogger import log_raw, write_json
from Acquisition.PacketParser import PacketParser
from Acquisition.SessionManager import session


def _save_reading(file, reading):
    record = {"received_at": reading.received_at.isoformat(),
              "measurements": reading.measurements, "flags": reading.flags}
    if reading.observed_at is not None:
        record["observed_at"] = reading.observed_at.isoformat()
    if reading.device_id is not None:
        record["device_id"] = reading.device_id
    if reading.packet_id is not None:
        record["packet_id"] = reading.packet_id
    write_json(file, record)


def capture(lines, output, stop=None, *, parser=None, metadata=None, session_stop=None):
    parser = PacketParser() if parser is None else parser
    timestamp = None
    lifecycle = stop if session_stop is None else session_stop
    with session(output, metadata=metadata, stop=lifecycle) as (raw, readings):
        try:
            for received_at, line in lines:
                if stop is not None and stop.is_set():
                    break
                log_raw(raw, received_at, line)
                timestamp = received_at
                reading = parser.feed(line.decode("utf-8", errors="replace"), timestamp)
                if reading is not None:
                    _save_reading(readings, reading)
        finally:
            if timestamp is not None:
                partial = parser.finish(timestamp)
                if partial is not None:
                    _save_reading(readings, partial)
