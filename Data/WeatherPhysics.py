"""Shared causal hourly reference ET convention."""
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
import math

import pandas as pd

from Physics.Evapotranspiration.Hargreaves import causal_interval_proxy


@dataclass(frozen=True)
class ReferenceET:
    latitude_deg: float
    solar_offset_minutes: float
    history_hours: int = 24

    def __post_init__(self):
        if not -90 < self.latitude_deg < 90 or not -720 <= self.solar_offset_minutes <= 840:
            raise ValueError("Invalid latitude or fixed solar offset from UTC")
        if not isinstance(self.history_hours, int) or self.history_hours < 24:
            raise ValueError("Reference ET requires at least 24 hourly temperatures")


class CausalReferenceET:
    def __init__(self, config):
        self.config = config
        self.history = deque(maxlen=config.history_hours)
        self.previous = None

    def advance(self, timestamp, temperature):
        timestamp = pd.Timestamp(timestamp)
        if timestamp.tzinfo is None or pd.isna(timestamp):
            raise ValueError("Reference ET requires an aware timestamp")
        timestamp = timestamp.tz_convert("UTC")
        if self.previous is not None:
            if timestamp <= self.previous:
                raise ValueError("Reference ET timestamps must strictly increase")
            if timestamp - self.previous != timedelta(hours=1):
                self.history.clear()
        self.previous = timestamp
        if not math.isfinite(temperature) or not -60 <= temperature <= 65:
            self.history.clear()
            return math.nan
        self.history.append(float(temperature))
        if len(self.history) < self.config.history_hours:
            return math.nan
        # Use a fixed offset from UTC, independent of civil daylight saving time.
        end = timestamp + timedelta(minutes=self.config.solar_offset_minutes)
        start = end - timedelta(hours=1)
        midnight = start.normalize() + timedelta(days=1)
        edges = (start, midnight, end) if end > midnight else (start, end)
        return sum(self._interval(a, b) for a, b in zip(edges, edges[1:]))

    def _interval(self, start, end):
        midnight = start.normalize()
        first = (start - midnight).total_seconds() / 3600
        last = (end - midnight).total_seconds() / 3600
        return causal_interval_proxy(self.history, self.config.latitude_deg, start.dayofyear,
                                     first, last, self.config.history_hours)
