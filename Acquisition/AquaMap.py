"""Decode the supplied AquaMap base station's USB serial CSV."""
from dataclasses import replace
import json
import math

from Acquisition.MappedPackets import MappedPacketParser, PacketSchema
from Acquisition.SensorData import SensorData


FIELDS = (
    ("latitude_deg", "deg"), ("longitude_deg", "deg"), ("soil_moisture_pct", "%"),
    ("soil_temperature_c", "C"), ("soil_ec_us_cm", "uS/cm"), ("soil_ph", "pH"),
    ("air_temperature_c", "C"), ("relative_humidity_pct", "%"), ("leaf_mv", "mV"),
)
NATIVE = PacketSchema.from_dict({
    "fields": {name: {"key": name, "unit": unit} for name, unit in FIELDS},
    "device_id_key": "node", "packet_id_key": "sample", "label": "aquamap_rx_csv",
})


class AquaMapParser(MappedPacketParser):
    def __init__(self):
        super().__init__(NATIVE)
        self.last = {}

    def feed(self, text, received_at):
        text = text.strip()
        if not text or text.startswith("#") or text.startswith("RX,rssi,snr,"):
            return None
        parts = text.split(",")
        if len(parts) != 15 or parts[0] != "RX" or parts[3] != "AQ":
            return SensorData(received_at, {}, ("invalid_aquamap_frame",))
        try:
            rssi, snr = int(parts[1]), float(parts[2])
            node, sample = int(parts[4]), int(parts[5])
            if node < 0 or sample < 0 or not math.isfinite(snr):
                raise ValueError
        except ValueError:
            return SensorData(received_at, {}, ("invalid_aquamap_header",))
        flags = ()
        previous = self.last.get(node)
        if previous is not None and sample < previous:
            flags = ("sequence_regression", "device_restart")
            self.seen = type(self.seen)((k, v) for k, v in self.seen.items() if k[0] != str(node))
        record = dict(node=node, sample=sample)
        record.update({name: value for (name, _), value in zip(FIELDS, parts[6:]) if value})
        reading = super().feed(json.dumps(record), received_at)
        if "duplicate_packet" not in reading.flags:
            self.last[node] = sample
        return replace(reading, flags=reading.flags + flags,
                       measurements={**reading.measurements, "radio_rssi_dbm": rssi, "radio_snr_db": snr})
