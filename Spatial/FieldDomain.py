"""Conservative polygon domains and face-connected masks in local metres."""
from dataclasses import dataclass

import numpy as np

from Spatial.AdaptiveGrid import _polygon_mask


def _ring(values):
    ring = np.asarray(values, dtype=float)
    if ring.ndim != 2 or ring.shape[1] != 2 or len(ring) < 3 or not np.isfinite(ring).all():
        raise ValueError("Polygon rings require at least three finite east/north vertices")
    if np.array_equal(ring[0], ring[-1]):
        ring = ring[:-1]
    next_point = np.roll(ring, -1, axis=0)
    if (len(ring) < 3 or len(np.unique(ring, axis=0)) != len(ring)
            or abs(np.sum(ring[:, 0] * next_point[:, 1] - ring[:, 1] * next_point[:, 0])) == 0):
        raise ValueError("Polygon rings must be simple and have nonzero area")
    return ring.copy()


def _orientation(a, b, c):
    delta, other = b - a, c - a
    return delta[0] * other[1] - delta[1] * other[0]


def _segment_intersection(a, b, c, d):
    if np.any(np.maximum(np.minimum(a, b), np.minimum(c, d)) > np.minimum(np.maximum(a, b), np.maximum(c, d))):
        return False
    # Check signs without multiplying cross products.
    first, second = _orientation(a, b, c), _orientation(a, b, d)
    third, fourth = _orientation(c, d, a), _orientation(c, d, b)
    return min(first, second) <= 0 <= max(first, second) and min(third, fourth) <= 0 <= max(third, fourth)


def _edges(ring):
    return tuple(zip(ring, np.roll(ring, -1, axis=0)))


def _edge_in_open_box(a, b, box):
    lo, hi = 0., 1.
    for start, delta, lower, upper in zip(a, b - a, box[:2], box[2:]):
        if delta == 0:
            if not lower < start < upper:
                return False
            continue
        interval = sorted(((lower - start) / delta, (upper - start) / delta))
        lo, hi = max(lo, interval[0]), min(hi, interval[1])
        if lo >= hi:
            return False
    return lo < hi


@dataclass(frozen=True)
class FieldDomain:
    polygon: np.ndarray
    holes: tuple = ()

    def __post_init__(self):
        rings = (_ring(self.polygon), *(_ring(hole) for hole in self.holes))
        for ring in rings:
            edges = _edges(ring)
            for i, (a, b) in enumerate(edges):
                for j in range(i + 2, len(edges)):
                    if i == 0 and j == len(edges) - 1:
                        continue
                    if _segment_intersection(a, b, *edges[j]):
                        raise ValueError("Polygon rings must be simple")
        for i, ring in enumerate(rings[1:], 1):
            if not _polygon_mask(ring, rings[0]).all():
                raise ValueError("Exclusion holes must be inside the field polygon")
            for j, previous in enumerate(rings[:i]):
                if any(_segment_intersection(*a, *b) for a in _edges(ring) for b in _edges(previous)):
                    raise ValueError("Polygon rings cannot touch or intersect")
                if j and (_polygon_mask(ring, previous).any() or _polygon_mask(previous, ring).any()):
                    raise ValueError("Exclusion holes cannot overlap")
        for ring in rings:
            ring.setflags(write=False)
        object.__setattr__(self, "polygon", rings[0])
        object.__setattr__(self, "holes", rings[1:])

    def contains(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Expected finite east/north coordinate pairs")
        inside = _polygon_mask(points, self.polygon)
        for hole in self.holes:
            inside &= ~_polygon_mask(points, hole)
        return inside

    def classify_box(self, bounds):
        """Return inside, outside, or mixed for a positive-area rectangle."""
        west, south, east, north = bounds
        center = np.array([[(west + east) / 2, (south + north) / 2]])
        for ring in (self.polygon, *self.holes):
            if any(_edge_in_open_box(a, b, bounds) for a, b in _edges(ring)):
                return "mixed"
        return "inside" if self.contains(center)[0] else "outside"

    def visible(self, left, right):
        """Conservative straight-line visibility between in-domain points."""
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        if not self.contains(np.stack((a, b, (a + b) / 2))).all():
            return False
        direction = b - a
        for c, d in _edges(self.polygon):
            if _segment_intersection(a, b, c, d):
                edge = d - c
                denominator = direction[0]*edge[1] - direction[1]*edge[0]
                if denominator != 0:
                    offset = c - a
                    fraction = (offset[0]*edge[1] - offset[1]*edge[0]) / denominator
                    if 0 < fraction < 1:
                        return False
        return not any(_segment_intersection(a, b, c, d)
                       for ring in self.holes for c, d in _edges(ring))


def regular_masks(grid, domain):
    """Return active nodes and visibility-gated horizontal and vertical faces."""
    points = grid.points().reshape(*grid.shape, 2)
    active = domain.contains(points.reshape(-1, 2)).reshape(grid.shape)
    east = active[:, :-1] & active[:, 1:]
    north = active[:-1] & active[1:]
    for mask, dx, dy in ((east, 1, 0), (north, 0, 1)):
        for row, col in zip(*np.nonzero(mask)):
            mask[row, col] = domain.visible(points[row, col], points[row + dy, col + dx])
    return active, east, north
