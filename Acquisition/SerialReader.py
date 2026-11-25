"""Serial byte capture with bounded retries and graceful stop."""
from datetime import datetime, timezone


def serial_lines(port, baud, stop, *, reconnect_attempts=0, reconnect_delay=1.0):
    import serial

    if not isinstance(reconnect_attempts, int) or reconnect_attempts < 0 or reconnect_delay < 0:
        raise ValueError("Invalid serial reconnection settings")
    pending = bytearray()
    remaining = reconnect_attempts
    while not stop.is_set():
        try:
            with serial.Serial(port, baudrate=baud, timeout=0.5) as connection:
                while not stop.is_set():
                    chunk = connection.read_until(size=8192)
                    if not chunk:
                        continue
                    pending.extend(chunk)
                    if pending.endswith(b"\n") or len(pending) >= 8192:
                        yield datetime.now(timezone.utc), bytes(pending)
                        pending.clear()
                break
        except serial.SerialException:
            # Preserve a partial fragment, but never combine across connections.
            if pending:
                yield datetime.now(timezone.utc), bytes(pending)
                pending.clear()
            if remaining == 0:
                raise
            remaining -= 1
            if stop.wait(reconnect_delay):
                break
    if pending:
        yield datetime.now(timezone.utc), bytes(pending)
