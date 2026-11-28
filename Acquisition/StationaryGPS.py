"""Dominant stationary GPS cluster in local metres, without assuming fix quality."""
from collections import defaultdict
import math
from statistics import median

from Spatial.SpatialGrid import local_coordinates


def stationary_location(points, *, radius_m=15., minimum_samples=5, minimum_fraction=.7, sentinels=()):
    if (not math.isfinite(radius_m) or radius_m <= 0 or type(minimum_samples) is not int
            or minimum_samples < 2 or not .5 < minimum_fraction <= 1):
        raise ValueError("Invalid stationary GPS thresholds")
    excluded = {tuple(item) for item in sentinels}
    valid, rejected = [], 0
    for point in points:
        try:
            lat, lon = map(float, point)
            if not (-90 < lat < 90 and -180 <= lon <= 180) or (lat, lon) in excluded:
                raise ValueError
        except (TypeError, ValueError):
            rejected += 1
            continue
        valid.append((lat, lon))
    result = dict(valid_samples=len(valid), rejected_observations=rejected,
                  dominant_fraction=0., spread_m=None, location=None)
    if not valid:
        return result
    xy = local_coordinates(valid, valid[0])
    cells = defaultdict(list)
    for i, (x, y) in enumerate(xy):
        cells[math.floor(x/radius_m), math.floor(y/radius_m)].append(i)
    best = []
    for (a, b), indices in cells.items():
        neighbors = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     for j in cells.get((a+dx, b+dy), ())]
        for i in indices:
            cluster = [j for j in neighbors if math.dist(xy[i], xy[j]) <= radius_m]
            if len(cluster) > len(best):
                best = cluster
    lat0, lon0 = valid[0]
    lat = median(valid[i][0] for i in best)
    lon = (lon0 + median((valid[i][1]-lon0+180) % 360 - 180 for i in best) + 180) % 360 - 180
    spread = max(math.hypot(*point) for point in local_coordinates([valid[i] for i in best], (lat, lon)))
    fraction = len(best)/len(valid)
    result.update(dominant_fraction=fraction, spread_m=spread)
    if len(best) >= minimum_samples and fraction >= minimum_fraction and spread <= radius_m:
        result["location"] = [lat, lon]
    return result
