import unittest

from Acquisition.Sequence import SequenceTracker


class SequenceTests(unittest.TestCase):
    def test_per_node_wrap_gap_replay_and_restart(self):
        tracker = SequenceTracker(16)
        self.assertEqual(tracker.update("a", "boot1", 15), ())
        self.assertEqual(tracker.update("b", "boot1", 15), ())
        self.assertEqual(tracker.update("a", "boot1", 2), ("sequence_gap:2",))
        self.assertEqual(tracker.update("a", "boot1", 2), ("duplicate_sequence",))
        self.assertEqual(tracker.update("a", "boot1", 1), ("out_of_order_sequence",))
        self.assertEqual(tracker.update("a", "boot2", 0), ("device_restart",))
        self.assertEqual(tracker.update("a", "boot1", 3), ("previous_boot_packet",))
        self.assertEqual(tracker.update("a", "boot2", False), ("invalid:sequence",))

    def test_reconnect_preserves_partial_bytes_without_concatenating_links(self):
        import sys
        from threading import Event
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from Acquisition.SerialReader import serial_lines

        class SerialException(Exception):
            pass

        first = MagicMock()
        first.__enter__.return_value.read_until.side_effect = [b"partial", SerialException()]
        second = MagicMock()
        stop = Event()

        def receive(*args, **kwargs):
            stop.set()
            return b"another\n"

        second.__enter__.return_value.read_until.side_effect = receive
        mock = SimpleNamespace(Serial=MagicMock(side_effect=[first, second]),
                               SerialException=SerialException)
        with patch.dict(sys.modules, {"serial": mock}):
            captured = [packet for _, packet in serial_lines("fixture", 9600, stop,
                                                            reconnect_attempts=1, reconnect_delay=0)]
        self.assertEqual(captured, [b"partial", b"another\n"])
