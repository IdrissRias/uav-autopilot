from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class Runway:
    airport_icao: str
    name1: str
    name2: str
    lat1: float
    lon1: float
    lat2: float
    lon2: float
    length_m: float
    width_m: float
    surface: int


@dataclass
class Airport:
    icao: str
    name: str
    lat: float
    lon: float
    best_runway: Optional[Runway] = None


def iter_airports_from_apt_dat(path: str | Path) -> Iterator[Airport]:
    """Very small apt.dat parser.

    Supports:
    - Airport header lines: 1/16/17 (airport types)
      Format: <code> <elev_ft> <has_tower> <icao> <name...>
    - Runway lines: 100
      We only parse endpoints and size.

    We compute airport (lat,lon) as midpoint of best runway found.
    """
    path = Path(path)
    cur: Optional[Airport] = None

    def finish() -> Optional[Airport]:
        nonlocal cur
        if not cur:
            return None
        if cur.best_runway:
            rw = cur.best_runway
            cur.lat = (rw.lat1 + rw.lat2) / 2.0
            cur.lon = (rw.lon1 + rw.lon2) / 2.0
        out = cur
        cur = None
        return out

    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            code = parts[0]

            if code in {"1", "16", "17"}:
                prev = finish()
                if prev:
                    yield prev
                # apt.dat header format (common):
                # <code> <elev_ft> <has_tower> <misc> <icao> <name...>
                # Example: "1 1625 0 0 5TE Tetlin"
                if len(parts) >= 6:
                    icao = parts[4]
                    name = " ".join(parts[5:])
                elif len(parts) >= 5:
                    icao = parts[3]
                    name = " ".join(parts[4:])
                else:
                    continue
                cur = Airport(icao=icao, name=name, lat=0.0, lon=0.0, best_runway=None)
                continue

            if code == "99":
                prev = finish()
                if prev:
                    yield prev
                continue

            if code == "100" and cur is not None:
                # Runway record (apt.dat v1050+)
                # We use:
                # 100 <width_m> <surface> ... <lat1> <lon1> ... <lat2> <lon2> ... <rwy1> <rwy2>
                # Indices are stable for endpoints in modern apt.dat.
                try:
                    width_m = float(parts[1])
                    surface = int(parts[2])
                    lat1 = float(parts[9])
                    lon1 = float(parts[10])
                    lat2 = float(parts[18])
                    lon2 = float(parts[19])
                    name1 = parts[22]
                    name2 = parts[23]
                except Exception:
                    continue

                # Approx length (rough) computed later by haversine; placeholder here.
                # compute approximate length via haversine
                try:
                    from .geo import haversine_m

                    length_m = haversine_m(lat1, lon1, lat2, lon2)
                except Exception:
                    length_m = 0.0

                rw = Runway(
                    airport_icao=cur.icao,
                    name1=name1,
                    name2=name2,
                    lat1=lat1,
                    lon1=lon1,
                    lat2=lat2,
                    lon2=lon2,
                    length_m=length_m,
                    width_m=width_m,
                    surface=surface,
                )
                # Keep the longest runway.
                if cur.best_runway is None or rw.length_m > (cur.best_runway.length_m or 0.0):
                    cur.best_runway = rw

    last = finish()
    if last:
        yield last
