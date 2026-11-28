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
