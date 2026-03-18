"""Peregrine — Runway boundary geometry.

Given a runway's threshold/end coordinates and width, computes:
- Distance remaining to end of runway (ft)
- Lateral offset from centerline (ft, signed: +right, -left)
- Whether the aircraft is inside the runway box

Used during GROUND and LAND phases to:
- Trigger rejected takeoff if running out of runway
- Detect runway excursion (off the sides)
- Score landing touchdown zone accuracy
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


# Earth radius in feet
_R_FT = 20_902_231.0


def _to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _haversine_ft(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in feet."""
    dlat = _to_rad(lat2 - lat1)
    dlon = _to_rad(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(_to_rad(lat1)) * math.cos(_to_rad(lat2))
         * math.sin(dlon / 2) ** 2)
    return _R_FT * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing in radians from point 1 to point 2."""
    lat1r, lat2r = _to_rad(lat1), _to_rad(lat2)
    dlon = _to_rad(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2r)
    y = (math.cos(lat1r) * math.sin(lat2r)
         - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return math.atan2(x, y)


@dataclass
class RunwayPosition:
    """Aircraft position relative to a runway."""
    along_ft: float           # distance along runway from threshold (0 = at threshold)
    lateral_ft: float         # offset from centerline (+right, -left when facing rwy heading)
    remaining_ft: float       # distance to end of runway
    runway_length_ft: float   # total runway length
    runway_width_ft: float    # total runway width
    inside: bool              # True if within the runway rectangle
    pct_used: float           # 0.0 = at threshold, 1.0 = at end


class RunwayBox:
    """Defines a runway rectangle from threshold/end coordinates + width.

    The runway is modeled as a rectangle:
    - Long axis: threshold → end (the centerline)
    - Short axis: width_ft / 2 on each side

    Call `locate(lat, lon)` to get the aircraft's position relative to the runway.
    """

    def __init__(
        self,
        threshold_lat: float,
        threshold_lon: float,
        end_lat: float,
        end_lon: float,
        width_ft: float,
        length_ft: float,
        designator: str = "",
    ):
        self.threshold_lat = threshold_lat
        self.threshold_lon = threshold_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        self.width_ft = width_ft
        self.length_ft = length_ft
        self.designator = designator

        # Precompute runway bearing (threshold → end)
        self._rwy_bearing = _bearing_rad(
            threshold_lat, threshold_lon, end_lat, end_lon
        )
        # Computed length from coordinates (may differ slightly from stated length)
        self._computed_length_ft = _haversine_ft(
            threshold_lat, threshold_lon, end_lat, end_lon
        )

    def locate(self, lat: float, lon: float) -> RunwayPosition:
        """Compute aircraft position relative to this runway.

        Projects the aircraft position onto the runway centerline to get:
        - along_ft: how far down the runway (0 = threshold)
        - lateral_ft: how far off centerline (+ = right, - = left)
        - remaining_ft: runway left ahead
        """
        # Vector from threshold to aircraft
        dist = _haversine_ft(self.threshold_lat, self.threshold_lon, lat, lon)
        bearing = _bearing_rad(self.threshold_lat, self.threshold_lon, lat, lon)

        # Angle between runway heading and bearing to aircraft
        angle_diff = bearing - self._rwy_bearing

        # Project onto runway axes
        along_ft = dist * math.cos(angle_diff)
        lateral_ft = dist * math.sin(angle_diff)

        # Use stated length for remaining (more accurate than computed)
        remaining_ft = self.length_ft - along_ft

        # Inside check
        inside = (
            0 <= along_ft <= self.length_ft
            and abs(lateral_ft) <= self.width_ft / 2
        )

        pct_used = max(0.0, min(1.0, along_ft / self.length_ft)) if self.length_ft > 0 else 0.0

        return RunwayPosition(
            along_ft=along_ft,
            lateral_ft=lateral_ft,
            remaining_ft=remaining_ft,
            runway_length_ft=self.length_ft,
            runway_width_ft=self.width_ft,
            inside=inside,
            pct_used=pct_used,
        )

    def __repr__(self) -> str:
        return (f"RunwayBox('{self.designator}' {self.length_ft:.0f}ft x {self.width_ft:.0f}ft "
                f"hdg={math.degrees(self._rwy_bearing):.0f}°)")


def runway_box_from_db(runway_row: dict) -> RunwayBox:
    """Create a RunwayBox from a Supabase runway row."""
    return RunwayBox(
        threshold_lat=runway_row["threshold_lat"],
        threshold_lon=runway_row["threshold_lon"],
        end_lat=runway_row["end_lat"],
        end_lon=runway_row["end_lon"],
        width_ft=runway_row["width_ft"],
        length_ft=runway_row["length_ft"],
        designator=runway_row.get("designator", ""),
    )
