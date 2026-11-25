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
