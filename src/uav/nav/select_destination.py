from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .apt_dat import iter_airports_from_apt_dat, Airport
from .geo import haversine_m


@dataclass
class Destination:
    icao: str
    name: str
    lat: float
    lon: float


def pick_nearest_airport(
    apt_dat_path: str | Path,
    lat: float,
    lon: float,
    min_distance_nm: float = 5.0,
    max_distance_nm: float = 80.0,
) -> Optional[Destination]:
    min_m = min_distance_nm * 1852.0
    max_m = max_distance_nm * 1852.0

    best: tuple[float, Airport] | None = None
    for ap in iter_airports_from_apt_dat(apt_dat_path):
        if not ap.best_runway:
            continue
        # airport lat/lon computed from runway midpoint in parser
        d = haversine_m(lat, lon, ap.lat, ap.lon)
        if d < min_m or d > max_m:
            continue
        if best is None or d < best[0]:
            best = (d, ap)

    if not best:
        return None

    _, ap = best
    return Destination(icao=ap.icao, name=ap.name, lat=ap.lat, lon=ap.lon)
