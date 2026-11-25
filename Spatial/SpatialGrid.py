"""Local metric coordinates and regular grids indexed by [north, east]."""
from dataclasses import dataclass

import numpy as np


def local_coordinates(latitude_longitude, origin):
    points = np.asarray(latitude_longitude, dtype=float)
    latitude, longitude = origin
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or not -90 < latitude < 90 or not -180 <= longitude <= 180
            or (np.abs(points) > [90, 180]).any()):
        raise ValueError("Invalid GPS coordinates")
    delta_lon = (points[:, 1] - longitude + 180) % 360 - 180
    north = 6371008.8 * np.deg2rad(points[:, 0] - latitude)
    east = 6371008.8 * np.deg2rad(delta_lon) * np.cos(np.deg2rad(latitude))
    return np.column_stack((east, north))


@dataclass(frozen=True)
class SpatialGrid:
    east: np.ndarray
    north: np.ndarray

    def __post_init__(self):
        for name in ("east", "north"):
            axis = np.asarray(getattr(self, name), dtype=float)
            if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
                raise ValueError("Grid axes must contain at least two finite coordinates")
            spacing = np.diff(axis)
            if (spacing <= 0).any() or not np.allclose(spacing, spacing[0]):
                raise ValueError("Grid coordinates must be evenly spaced and increasing")
            object.__setattr__(self, name, axis)

    @property
    def shape(self):
        return len(self.north), len(self.east)

    def points(self):
        east, north = np.meshgrid(self.east, self.north)
        return np.column_stack((east.ravel(), north.ravel()))
