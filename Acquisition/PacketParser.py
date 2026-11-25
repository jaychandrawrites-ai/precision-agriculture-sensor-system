"""Streaming parser for the AIR and SOIL blocks in src/main.cpp."""
import math
import re

from Acquisition.SensorData import SensorData


FIELDS = {
    "AIR": {"Temperature": ("air_temperature_c", "C", -60, 65),
            "Humidity": ("relative_humidity_pct", "%RH", 0, 100)},
    "SOIL": {"Temperature": ("soil_temperature_c", "C", -60, 85),
             "Moisture": ("soil_moisture_pct", "%", 0, 100),
             "EC": ("soil_ec_us_cm", "uS/cm", 0, 65535),
             "pH": ("soil_ph", "", 0, 14)},
}
READING = re.compile(r"([^:]+):\s*(\S+)\s*(.*)")
EXPECTED = {entry[0] for fields in FIELDS.values() for entry in fields.values()}


class PacketParser:
    def __init__(self):
        self.section = None
        self.measurements = {}
        self.seen = set()
        self.flags = []

    def finish(self, received_at):
        if self.section is None:
            return None
        missing = EXPECTED - self.measurements.keys()
        flags = self.flags + [f"missing:{field}" for field in sorted(missing)]
        packet = SensorData(received_at, self.measurements, tuple(flags))
        self.__init__()
        return packet

    def feed(self, text, received_at):
        line = text.strip()
        if line == "--- AIR ---":
            previous = self.finish(received_at)
            self.section = "AIR"
            return previous
        if self.section is None:
            return None
        if line == "--- SOIL ---":
            self.section = "SOIL"
        elif line and set(line) == {"="}:
            return self.finish(received_at)
        elif line.startswith("ERROR:"):
            self.flags.append(f"sensor_error:{self.section.lower()}")
        elif line:
            match = READING.fullmatch(line)
            field = FIELDS[self.section].get(match[1].strip()) if match else None
            if field is None:
                self.flags.append("unrecognized_line")
                return None
            name, unit, minimum, maximum = field
            if name in self.seen:
                self.flags.append(f"duplicate:{name}")
                return None
            self.seen.add(name)
            try:
                value = float(match[2])
            except ValueError:
                value = math.nan
            if match[3] != unit or not minimum <= value <= maximum:
                self.flags.append(f"invalid:{name}")
            else:
                self.measurements[name] = value
        return None
