from contextlib import contextmanager
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from time import monotonic, sleep
import unittest

from Acquisition.AITrace import TraceStore
from Acquisition.ReceiverPool import ReceiverPool


class ReceiverPoolTests(unittest.TestCase):
    def test_independent_simulated_workers_and_exclusive_ports(self):
        queues = {name: Queue() for name in ("first", "second")}

        @contextmanager
        def connect(port, baud):
            queue = queues[port]

            class Link:
                def read_until(self, size):
                    try:
                        return queue.get(timeout=.01)
                    except Empty:
                        return b""

            yield Link()

        def wait(predicate):
            deadline = monotonic() + 3
            while not predicate() and monotonic() < deadline:
                sleep(.01)
            self.assertTrue(predicate())

        with TemporaryDirectory() as directory:
            store = TraceStore(directory)
            surveys = [store.create(name)["id"] for name in queues]
            pool = ReceiverPool(store, queues)
            pool.default.connector = connect
            try:
                pool.connect(surveys[0], "first", 115200, "aquamap_rx_csv")
                with self.assertRaisesRegex(ValueError, "assigned"):
                    pool.connect(surveys[1], "first", 115200, "aquamap_rx_csv")
                pool.connect(surveys[1], "second", 115200, "aquamap_rx_csv")
                workers = [pool.for_survey(survey) for survey in surveys]
                for counter in range(1, 6):
                    for node, queue in enumerate(queues.values(), 1):
                        queue.put(f"RX,-90,3,AQ,{node},{counter},0,0,30,24,450,6.8,22,55,350\n".encode())
                for worker, survey in zip(workers, surveys):
                    wait(lambda: worker.diagnostics(survey)["native_readiness"]["ready"])
                group = pool.group("start")
                self.assertTrue(all(item["ok"] for item in group["receivers"].values()))
                workers[0].disconnect(surveys[0])
                self.assertTrue(workers[1].diagnostics()["recording"])
                queues["second"].put(b"RX,-90,3,AQ,2,6,0,0,30,24,450,6.8,22,55,350\n")
                wait(lambda: workers[1].diagnostics()["packets_received"] == 6)
                active = store.status(surveys[1])["active_segment_id"]
                wait(lambda: bool(store.segment_readings(surveys[1], active)))
                self.assertEqual(store.segment_readings(surveys[1], active)[0]["device_id"], "2")
                aligned = pool.aligned_export(5).decode()
                self.assertIn(group["group_id"], aligned)
                self.assertIn("receipt_offset_seconds", aligned)
                pool.group("stop")
                self.assertFalse(workers[1].diagnostics()["recording"])
            finally:
                pool.close()
