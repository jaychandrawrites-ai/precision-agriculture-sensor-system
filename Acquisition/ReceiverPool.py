"""Route independent surveys to exclusive serial receiver workers."""
from threading import RLock

from Acquisition.TraceReceiver import TraceReceiver


class ReceiverPool:
    def __init__(self, store, ports):
        self.store, self.ports = store, ports
        self.default = TraceReceiver(store, ports)
        self.workers = {}
        self.lock = RLock()

    def for_survey(self, identifier):
        with self.lock:
            return self.workers.get(identifier, self.default)

    def connect(self, identifier, port, baud, mapping):
        with self.lock:
            if any(other != identifier and worker.diagnostics().get("port") == port
                   and worker.diagnostics()["state"] != "disconnected"
                   for other, worker in self.workers.items()):
                raise ValueError("Serial port is already assigned to another survey")
            worker = self.workers.get(identifier)
            if worker is None:
                worker = TraceReceiver(self.store, self.ports, connector=self.default.connector)
            result = worker.connect(identifier, port, baud, mapping)
            self.workers[identifier] = worker
            return result

    def close(self):
        with self.lock:
            for worker in self.workers.values():
                worker.disconnect()
            self.default.disconnect()

    def group(self, action):
        """Apply per-worker controls independently and return partial failures."""
        from uuid import uuid4

        if action not in {"start", "stop"}:
            raise ValueError("Unknown receiver group action")
        with self.lock:
            workers = tuple(self.workers.items())
        group_id = uuid4().hex if action == "start" else None
        results = {}
        for identifier, worker in workers:
            try:
                if action == "start":
                    if worker.diagnostics(identifier)["state"] != "connected":
                        raise ValueError("Receiver is not connected")
                    worker.start(identifier, label="Group recording",
                                 calibration={"acquisition_group": group_id})
                else:
                    worker.stop(identifier)
                results[identifier] = {"ok": True}
            except (ValueError, KeyError, OSError) as error:
                results[identifier] = {"ok": False, "error": str(error)}
        return {"group_id": group_id, "receivers": results}

    def aligned_export(self, seconds=5):
        """Long-form UTC alignment keeps every observed packet and its offset."""
        from datetime import datetime, timezone
        import csv
        import io
        import json

        if type(seconds) is not int or not 1 <= seconds <= 3600:
            raise ValueError("Alignment interval must be 1–3600 seconds")
        with self.lock:
            workers = tuple(self.workers)
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["survey_id", "location_id", "segment_id", "group_id", "node_id",
                         "received_at", "observed_at", "utc_bin_start", "receipt_offset_seconds",
                         "measurements_json", "flags_json"])
        for identifier in workers:
            for location in self.store.status(identifier)["locations"]:
                for segment in location["segments"]:
                    group_id = segment["calibration"].get("acquisition_group")
                    for row in self.store.segment_readings(identifier, segment["id"]):
                        stamp = datetime.fromisoformat(row["received_at"]).timestamp()
                        start = int(stamp // seconds) * seconds
                        writer.writerow([identifier, location["id"], segment["id"], group_id,
                                         row.get("device_id"), row["received_at"], row.get("observed_at"),
                                         datetime.fromtimestamp(start, timezone.utc).isoformat(), stamp-start,
                                         json.dumps(row["measurements"]), json.dumps(row["flags"])])
        return stream.getvalue().encode()
