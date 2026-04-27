from __future__ import annotations

import math


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def destination_point(
    lat: float, lon: float, distance_m: float, bearing: float,
) -> tuple[float, float]:
    """Compute destination lat/lon from a start point, bearing (deg), and distance (m)."""
    R = 6371000.0
    brng = math.radians(bearing)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    d = distance_m / R

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) +
        math.cos(lat1) * math.sin(d) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lat2), math.degrees(lon2))


def project_onto_segment(
    lat: float, lon: float,
    lat_start: float, lon_start: float,
    lat_end: float, lon_end: float,
) -> tuple[float, float]:
    """Project a point onto a runway centerline segment.

    Returns (along_m, cross_m):
      • along_m:  signed distance from start along the segment axis.
                  0 at start, length_m at end, negative if aircraft is behind start.
      • cross_m:  perpendicular distance from the centerline.
                  Positive = left of centerline looking start→end, negative = right.

    Uses flat-earth approximation centered at lat_start. For runway-scale
    geometry (≤ 5 km) the error is sub-millimeter, which is why we don't
    bother with full great-circle projection.
    """
    # Meters per degree at this latitude. 110540 for north, 111320·cos(φ) for east.
    # (Standard WGS-84 local tangent-plane approximation.)
    lat0 = math.radians(lat_start)
    m_per_deg_lat = 110540.0
    m_per_deg_lon = 111320.0 * math.cos(lat0)

    # Runway axis vector, start → end, in meters.
    ax = (lon_end - lon_start) * m_per_deg_lon
    ay = (lat_end - lat_start) * m_per_deg_lat
    length = math.hypot(ax, ay)
    if length < 1e-6:
        # Degenerate zero-length segment — treat as a point.
        return (0.0, 0.0)
    # Unit vector along the runway.
    ux = ax / length
    uy = ay / length

    # Aircraft offset from start, in meters.
    px = (lon - lon_start) * m_per_deg_lon
    py = (lat - lat_start) * m_per_deg_lat

    # Along = dot product with unit axis.
    along_m = px * ux + py * uy
    # Cross = 2D cross product (z-component). Positive = left of axis.
    cross_m = py * ux - px * uy
    return (along_m, cross_m)
