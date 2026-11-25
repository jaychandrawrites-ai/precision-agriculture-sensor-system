"""Explicit modular sequence tracking within each identified node boot."""
from collections import OrderedDict


class SequenceTracker:
    def __init__(self, modulus, capacity=4096):
        if not isinstance(modulus, int) or modulus < 3 or capacity < 1:
            raise ValueError("Sequence modulus and capacity must be positive")
        self.modulus, self.capacity = modulus, capacity
        self.previous = OrderedDict()

    def update(self, device, boot, counter):
        if (device is None or boot is None or isinstance(counter, bool) or not isinstance(counter, int)
                or not 0 <= counter < self.modulus):
            return ("invalid:sequence",)
        reference = self.previous.get(device)
        if reference is not None and reference[0] == boot:
            distance = (counter - reference[1]) % self.modulus
            if distance == 0:
                return ("duplicate_sequence",)
            if 2 * distance >= self.modulus:
                return ("out_of_order_sequence",)
            flags = (f"sequence_gap:{distance - 1}",) if distance > 1 else ()
        else:
            if reference is not None and boot in reference[2]:
                return ("previous_boot_packet",)
            flags = ("device_restart",) if reference is not None else ()
        older = () if reference is None else reference[2]
        if reference is not None and reference[0] != boot:
            older = (*older[-7:], reference[0])
        self.previous[device] = (boot, counter, older)
        self.previous.move_to_end(device)
        if len(self.previous) > self.capacity:
            self.previous.popitem(last=False)
        return flags
