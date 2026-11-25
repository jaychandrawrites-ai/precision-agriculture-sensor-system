"""Serial line stream with read timeouts and stop support."""
from datetime import datetime, timezone


def serial_lines(port, baud, stop):
    import serial

    pending = bytearray()
    with serial.Serial(port, baudrate=baud, timeout=0.5) as connection:
        while not stop.is_set():
            chunk = connection.read_until(size=8192)
            if not chunk:
                continue
            pending.extend(chunk)
            if pending.endswith(b"\n") or len(pending) >= 8192:
                yield datetime.now(timezone.utc), bytes(pending)
                pending.clear()
        if pending:
            yield datetime.now(timezone.utc), bytes(pending)
