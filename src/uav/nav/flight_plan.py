"""Flight path planner — computes a full waypoint sequence from takeoff to landing.

Given departure position, destination airport/runway, and aircraft performance data,
generates a list of Waypoints that the reactive director follows as its "GPS".

Each waypoint has: lat, lon, alt_ft, speed_kts, phase, heading (optional).
The reactive director still handles stick-and-throttle — this just tells it WHERE to go.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from uav.nav.geo import bearing_deg, destination_point, haversine_m


@dataclass
class Waypoint:
    """A single point on the flight path."""
    name: str
    lat: float
    lon: float
    alt_ft: float                # target MSL altitude at this point
    speed_kts: float             # target airspeed at this point
    phase: str                   # GROUND, CLIMB, CRUISE, APPROACH, LAND
    heading: Optional[float] = None  # if set, fly this exact heading (else fly toward next WP)
    fly_over: bool = False       # if True, must pass over point (not just get close)


@dataclass
class FlightPlan:
    """A complete flight path from takeoff to landing."""
    waypoints: List[Waypoint] = field(default_factory=list)
    active_idx: int = 0          # index of the waypoint we're currently flying toward

    @property
    def active(self) -> Optional[Waypoint]:
        if 0 <= self.active_idx < len(self.waypoints):
            return self.waypoints[self.active_idx]
        return None

    @property
    def completed(self) -> bool:
        return self.active_idx >= len(self.waypoints)

    def advance(self) -> Optional[Waypoint]:
        """Move to the next waypoint. Returns it, or None if done."""
        self.active_idx += 1
        return self.active

    def remaining(self) -> List[Waypoint]:
        return self.waypoints[self.active_idx:]


def _wrap_deg(a: float) -> float:
    """Normalize angle to ±180°."""
    return (a + 180.0) % 360.0 - 180.0


NM_TO_M = 1852.0
FT_TO_M = 0.3048
GLIDE_FT_PER_NM = 636.0  # 6° glideslope — steep, short approach with idle+drag


def build_flight_plan(
    # Departure
    dep_lat: float,
    dep_lon: float,
    dep_alt_ft: float,           # field elevation MSL
    dep_heading: float,          # runway heading
    # Destination
    dest_lat: float,
    dest_lon: float,
    dest_alt_ft: float,          # field elevation MSL
    dest_rwy_heading: Optional[float] = None,  # runway heading at destination
    dest_threshold_lat: Optional[float] = None,
    dest_threshold_lon: Optional[float] = None,
    # Aircraft performance
    v_rotate: float = 90.0,
    v_climb: float = 130.0,
    v_cruise: float = 200.0,
    v_approach: float = 83.0,
    v_land: float = 77.0,
    cruise_alt_ft: float = 3000.0,  # MSL
    takeoff_roll_ft: float = 2000.0,
) -> FlightPlan:
    """Build a complete flight plan from departure to destination.

    Returns a FlightPlan with waypoints the reactive director can follow.
    """
    wps: List[Waypoint] = []

    # Use threshold if available, else airport center
    arr_lat = dest_threshold_lat if dest_threshold_lat is not None else dest_lat
    arr_lon = dest_threshold_lon if dest_threshold_lon is not None else dest_lon
    rwy_hdg = dest_rwy_heading

    # Total route distance
    route_dist_m = haversine_m(dep_lat, dep_lon, arr_lat, arr_lon)
    route_dist_nm = route_dist_m / NM_TO_M

    # ── TAKEOFF ───────────────────────────────────────────────────────
    wps.append(Waypoint(
        name="TAKEOFF_START",
        lat=dep_lat, lon=dep_lon,
        alt_ft=dep_alt_ft,
        speed_kts=0.0,
        phase="GROUND",
        heading=dep_heading,
    ))

    # ── APPROACH ENTRY — the ONE point the plane flies straight to ────
    # After takeoff the plane climbs and cruises in a STRAIGHT LINE to
    # this single point. It sits on the extended runway centerline, far
    # enough back from the threshold for the plane to descend on a 4.5°
    # glidepath from cruise altitude, plus a 2nm margin to stabilise.
    #
    # Geometry:
    #   descent_nm  = (cruise_alt - runway_elev - 50ft) / 477
    #   entry_nm    = descent_nm + 2nm margin
    #   position    = threshold + entry_nm along the reciprocal runway heading
    #
    # The plane arrives here at cruise altitude, already aligned with the
    # runway. It immediately begins descending — no dogleg, no heading snap.

    cruise_agl = cruise_alt_ft - dest_alt_ft
    descent_nm = max(0.0, (cruise_agl - 50.0)) / GLIDE_FT_PER_NM
    # Margin: small buffer before the descent begins.
    # The reactive director handles alignment and deceleration during
    # the approach phase itself, so we just need ~1nm stabilisation room.
    entry_margin_nm = 1.0
    entry_nm = descent_nm + entry_margin_nm

    if rwy_hdg is not None:
        entry_lat, entry_lon = destination_point(
            arr_lat, arr_lon, entry_nm * NM_TO_M,
            (rwy_hdg + 180.0) % 360.0,  # back along the runway centerline
        )
    else:
        # No runway heading — place entry on the direct line from departure
        dep_bearing = bearing_deg(dep_lat, dep_lon, arr_lat, arr_lon)
        entry_lat, entry_lon = destination_point(
            arr_lat, arr_lon, entry_nm * NM_TO_M,
            (dep_bearing + 180.0) % 360.0,
        )

    wps.append(Waypoint(
        name="APPROACH_ENTRY",
        lat=entry_lat, lon=entry_lon,
        alt_ft=cruise_alt_ft,
        speed_kts=v_approach + 30,
        phase="APPROACH",  # triggers approach phase on capture
        heading=rwy_hdg,  # already aligned with runway
    ))

    # ── LANDING ───────────────────────────────────────────────────────
    wps.append(Waypoint(
        name="THRESHOLD",
        lat=arr_lat, lon=arr_lon,
        alt_ft=dest_alt_ft + 50.0,
        speed_kts=v_land,
        phase="LAND",
        heading=rwy_hdg,
        fly_over=True,
    ))

    # Touchdown zone — ~1000ft past threshold
    if rwy_hdg is not None:
        tdz_lat, tdz_lon = destination_point(
            arr_lat, arr_lon, 300.0, rwy_hdg,
        )
    else:
        tdz_lat, tdz_lon = arr_lat, arr_lon
    wps.append(Waypoint(
        name="TOUCHDOWN",
        lat=tdz_lat, lon=tdz_lon,
        alt_ft=dest_alt_ft,
        speed_kts=v_land,
        phase="LAND",
        heading=rwy_hdg,
    ))

    return FlightPlan(waypoints=wps)


def format_plan(plan: FlightPlan) -> str:
    """Pretty-print a flight plan for logging."""
    lines = ["─── FLIGHT PLAN ───"]
    for i, wp in enumerate(plan.waypoints):
        marker = "→" if i == plan.active_idx else " "
        hdg_str = f"hdg={wp.heading:.0f}°" if wp.heading is not None else ""
        lines.append(
            f"  {marker} WP{i:02d} {wp.name:<22s} "
            f"{wp.lat:9.5f}, {wp.lon:10.5f}  "
            f"{wp.alt_ft:6.0f}ft  {wp.speed_kts:5.1f}kts  "
            f"{wp.phase:<10s} {hdg_str}"
        )
    lines.append("────────────────────")
    return "\n".join(lines)
