"""AI Trace receiver lifecycle, independent of location recording."""
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, RLock, Thread

from Acquisition.MappedPackets import GPS_FIELDS, MappedPacketParser, PacketSchema
from Acquisition.PacketParser import PacketParser
from Acquisition.TraceDiagnostics import TraceDiagnostics, calibration_readout, units_for
from Acquisition.TraceImport import replay_segment, store_record


def _serial_connection(port, baud):
    import serial

    return serial.Serial(port, baudrate=baud, timeout=.4)


def _recoverable(error):
    if isinstance(error, OSError):
        return True
    # pyserial's SerialException is an OSError, including device-open errors.
    return False


class TraceReceiver:
    """Monitor a serial stream while independently starting and stopping recordings."""

    def __init__(self, store, ports=(), connector=None, retry_seconds=1.):
        self.store = store
        self.ports = frozenset(ports)
        self.connector = connector or _serial_connection
        self.retry_seconds = retry_seconds
        self._lock = RLock()
        self._worker = None
        self._stop = None
        self._survey = None
        self._schema = None
        self._parser = None
        self._recording = False
        self._state = "disconnected"
        self._port = self._baud = None
        self._error = None
        self._latest = None
        self._bytes = 0
        self._packets = 0
        self._last_receipt = None
        self._pending = bytearray()
        self._monitor = TraceDiagnostics()
        self._calibration = {}
        self._segment_last_receipt = None

    def _reset_parser(self):
        self._parser = PacketParser() if self._schema is None else MappedPacketParser(self._schema)

    def connect(self, survey_id, port, baud=9600, mapping=None):
        if port not in self.ports:
            raise ValueError("Select a serial port enabled when the dashboard was launched")
        if type(baud) is not int or not 300 <= baud <= 2_000_000:
            raise ValueError("Invalid serial baud rate")
        schema = PacketSchema.from_dict(mapping) if mapping is not None else None
        with self._lock:
            self.store.status(survey_id)
            if self._worker is not None and self._worker.is_alive():
                raise ValueError("Disconnect the current receiver before connecting another")
            if self._recording:
                raise ValueError("Stop the recording before connecting a receiver")
            self._survey = survey_id
            self._schema = schema
            self._reset_parser()
            self._port, self._baud, self._state, self._error = port, baud, "connecting", None
            self._monitor = TraceDiagnostics()
            self._last_receipt = None
            self._bytes, self._packets, self._latest = 0, 0, None
            self._stop = Event()
            self._worker = Thread(target=self._run, args=(self._stop, port, baud),
                                  daemon=True, name="jupiter-trace-serial")
            self._worker.start()
            return self.diagnostics(survey_id)

    def _run(self, stop, port, baud):
        while not stop.is_set():
            try:
                with self.connector(port, baud) as link:
                    with self._lock:
                        if stop.is_set():
                            break
                        self._state, self._error = "connected", None
                    while not stop.is_set():
                        with self._lock:
                            raw = link.read_until(size=8192 - len(self._pending))
                            self._bytes += len(raw)
                            self._pending.extend(raw)
                            if self._pending.endswith(b"\n") or len(self._pending) >= 8192:
                                self.feed(self._survey, bytes(self._pending), datetime.now(timezone.utc), count_bytes=False)
                                self._pending.clear()
                        stop.wait(.001)
            except (OSError, ValueError, ImportError) as exc:
                with self._lock:
                    self._stop_recording(interrupted=True)
                    self._pending.clear()
                    self._reset_parser()
                    self._latest = None
                    recoverable = _recoverable(exc)
                    self._monitor.error(message=str(exc), timestamp=datetime.now(timezone.utc), recoverable=recoverable)
                    if not recoverable:
                        self._state, self._error = "failed", str(exc)
                        break
                    self._state, self._error = "reconnecting", str(exc)
                stop.wait(self.retry_seconds)
        with self._lock:
            self._stop_recording()
            self._pending.clear()
            if self._state != "failed":
                self._state = "disconnected"

    def start(self, survey_id, *, label=None, location_id=None, coordinates=None, **metadata):
        with self._lock:
            if self._survey is not None and survey_id != self._survey and (self._state != "disconnected" or self._recording):
                raise ValueError("Receiver belongs to another survey")
            if self._recording:
                raise ValueError("Stop the current location before starting another")
            if self._state in ("reconnecting", "connecting", "failed"):
                raise ValueError("Wait for the receiver to connect or disconnect it before replaying")
            result = self.store.start(survey_id, label=label, location_id=location_id,
                                      coordinates=coordinates, **metadata)
            self._pending.clear()
            self._segment_last_receipt = None
            self._calibration = metadata.get("calibration") or {}
            self._survey, self._recording = survey_id, True
            if self._schema is None:
                self._reset_parser()
            return result

    def _stop_recording(self, interrupted=False):
        if self._recording:
            if self._pending:
                self.store.ingest(self._survey, bytes(self._pending), datetime.now(timezone.utc), None)
                self._pending.clear()
            if self._schema is None and self._segment_last_receipt is not None:
                reading = self._parser.finish(self._segment_last_receipt)
                if reading is not None:
                    parsed = store_record(reading)
                    self._readout(parsed, self._segment_last_receipt)
                    self.store.ingest(self._survey, None, self._segment_last_receipt, parsed)
            result = self.store.stop(self._survey, interrupted=interrupted)
            self._recording = False
            if self._schema is None:
                self._reset_parser()
            return result
        return None

    def stop(self, survey_id):
        with self._lock:
            if self._survey != survey_id or not self._recording:
                return self.store.status(survey_id)
            return self._stop_recording()

    def finish(self, survey_id):
        with self._lock:
            self.stop(survey_id)
            result = self.store.finish(survey_id)
        if self._survey == survey_id:
            self.disconnect(survey_id)
        return result

    def feed(self, survey_id, raw, received_at, *, count_bytes=True):
        with self._lock:
            if survey_id != self._survey:
                raise ValueError("Receiver belongs to another survey")
            if self._parser is None:
                self._reset_parser()
            if count_bytes:
                self._bytes += len(raw)
            self._packets += 1
            self._last_receipt = received_at
            if self._recording:
                self._segment_last_receipt = received_at
                self.store.ingest(survey_id, raw, received_at)
            reading = self._parser.feed(raw.decode("utf-8", errors="replace"), received_at)
            parsed = store_record(reading) if reading is not None else None
            if parsed is not None:
                self._readout(parsed, received_at)
                if self._recording:
                    self.store.ingest(survey_id, None, received_at, parsed)

    def _readout(self, parsed, received_at):
        self._monitor.reading(parsed, received_at)
        measurements = parsed["measurements"]
        self._latest = {
            "received_at": received_at.isoformat(),
            "observed_at": parsed["observed_at"],
            "measurements": {key: value for key, value in measurements.items() if key not in GPS_FIELDS},
            "gps_available": GPS_FIELDS <= measurements.keys(),
            "gps_validity": ("valid" if GPS_FIELDS <= measurements.keys() else "invalid"
                             if any(flag.startswith(("invalid:gps", "incomplete:gps"))
                                    for flag in parsed["flags"]) else "not_reported"),
            "flags": parsed["flags"],
            "device_id": parsed["device_id"],
            "boot_id": parsed.get("boot_id"),
        }

    def replay(self, survey_id, payload, *, label=None, location_id=None, mapping=None):
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise ValueError("Disconnect the live receiver before replaying")
            if self._recording:
                raise ValueError("Stop the active segment before replaying another segment")
            with TemporaryDirectory() as folder:
                path = Path(folder) / "capture.jsonl"
                path.write_bytes(payload)
                return replay_segment(self.store, survey_id, path, label=label,
                                      location_id=location_id, schema=mapping)

    def disconnect(self, survey_id=None):
        with self._lock:
            if survey_id is not None and self._survey != survey_id:
                return self.diagnostics(survey_id)
            worker = self._worker
            if self._stop is not None:
                self._stop.set()
        if worker is not None:
            worker.join(timeout=2)
            if worker.is_alive():
                raise ValueError("Receiver is still stopping")
        with self._lock:
            self._stop_recording()
            self._state = "disconnected"
            self._survey, self._port, self._baud = None, None, None
            self._calibration = {}
            self._schema, self._parser, self._latest = None, None, None
            return self.diagnostics(survey_id)

    def diagnostics(self, survey_id=None):
        with self._lock:
            if survey_id is not None and self._survey != survey_id:
                return {"state": "disconnected", "bytes_received": 0, "packets_received": 0,
                        "latest": None, "error": None}
            return {"state": self._state, "port": self._port, "baud": self._baud,
                    "bytes_received": self._bytes, "packets_received": self._packets,
                    "recording": self._recording, "pending_frame_bytes": len(self._pending),
                    "last_receipt": self._last_receipt.isoformat() if self._last_receipt else None,
                    "latest": self._latest, "error": self._error, "configured_units": units_for(self._schema),
                    "calibration": calibration_readout(self._calibration,
                                                       (self._latest or {}).get("measurements", {})),
                    **self._monitor.snapshot()}
