"""
V2 Flight Plan — Dense ribbon of PathPoints.

The path planner runs ONCE before takeoff.  It computes the entire flight
as a dense ribbon of hundreds of PathPoints, spaced ~0.1 nm apart.
Each point carries lat, lon, altitude, speed, heading, phase, and
gear/flap config.

The FlightEngine follows this ribbon in real time — it never decides
*what* to do, only *how closely* to track the ribbon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from uav.nav.geo import haversine_m, bearing_deg


# ─── Data ─────────────────────────────────────────────────────────────

@dataclass
class PathPoint:
    lat: float
    lon: float
    alt_ft: float          # MSL
    speed_kts: float
    heading_deg: float     # track over ground
    phase: str             # GROUND / CLIMB / CRUISE / DESCENT / APPROACH / FLARE / ROLLOUT
    gear_down: bool = True
    flap_ratio: float = 0.0
    dist_from_start_nm: float = 0.0   # cumulative distance along ribbon


class RibbonPath:
    """Immutable dense ribbon.  Provides spatial queries for the engine."""

    def __init__(self, points: List[PathPoint]) -> None:
        if not points:
            raise ValueError("RibbonPath needs at least one point")
        self.points = points

    def __len__(self) -> int:
        return len(self.points)

    # -- Spatial queries --------------------------------------------------

    def nearest_ahead(self, lat: float, lon: float, heading: float,
                      last_idx: int = 0) -> int:
        """Return the index of the nearest ribbon point that is ahead of
        (lat, lon) travelling on *heading*.

        Searches forward from *last_idx* (monotonic — never goes backward)
        within a window of 200 points.  This keeps the search O(1) per tick.
        """
        best_idx = last_idx
        best_dist = math.inf
        end = min(last_idx + 200, len(self.points))
        cos_h = math.cos(math.radians(heading))
        sin_h = math.sin(math.radians(heading))

        for i in range(last_idx, end):
            p = self.points[i]
            dlat = p.lat - lat
            dlon = (p.lon - lon) * math.cos(math.radians(lat))
            # Project onto heading vector — positive = ahead
            along = dlat * cos_h + dlon * sin_h
            if along < -0.001:  # behind us
                continue
            d = dlat * dlat + dlon * dlon
            if d < best_dist:
                best_dist = d
                best_idx = i

        return best_idx

    def lookahead(self, idx: int, dist_nm: float) -> int:
        """Return the index of the point ~dist_nm ahead of *idx*."""
        target = self.points[idx].dist_from_start_nm + dist_nm
        for i in range(idx, len(self.points)):
            if self.points[i].dist_from_start_nm >= target:
                return i
        return len(self.points) - 1


# ─── Helpers ──────────────────────────────────────────────────────────

def _destination_point(lat: float, lon: float, bearing_deg_: float,
                       dist_m: float) -> tuple[float, float]:
    """Return (lat, lon) that is *dist_m* metres from (lat, lon) on bearing."""
    R = 6_371_000.0
    d = dist_m / R
    brng = math.radians(bearing_deg_)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) +
        math.cos(lat1) * math.sin(d) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _wrap180(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


# ─── Path Builder ─────────────────────────────────────────────────────

_STEP_NM = 0.1          # ribbon density: one point every ~185 m
_STEP_M = _STEP_NM * 1852.0
_GLIDESLOPE_DEG = 3.0   # 318 ft per nm descent
_GLIDE_FT_PER_NM = 318.0


def plan_path(
    *,
    dep_lat: float,
    dep_lon: float,
    dep_alt_ft: float,
    dep_heading: float,
    dest_lat: float,
    dest_lon: float,
    dest_alt_ft: float,
    dest_rwy_heading: float | None = None,
    dest_threshold_lat: float | None = None,
    dest_threshold_lon: float | None = None,
    cruise_alt_ft: float,
    v_rotate: float = 90.0,
    v_climb: float = 160.0,
    v_cruise: float = 200.0,
    v_approach: float = 83.0,
    v_land: float = 77.0,
    climb_fpm: float = 1600.0,
    takeoff_roll_ft: float = 2000.0,
) -> RibbonPath:
    """Build the full ribbon from departure to arrival.

    Segments:
      GROUND   — takeoff roll along dep_heading
      CLIMB    — climb to cruise_alt_ft, initially on dep_heading
      CRUISE   — level flight toward destination
      DESCENT  — top-of-descent (TOD) to approach altitude
      APPROACH — 3° glideslope to flare height
      FLARE    — last 50 ft, decelerate
      ROLLOUT  — on the ground, braking
    """
    points: list[PathPoint] = []
    cumul_nm = 0.0

    # Destination runway threshold (use dest coords if no runway data)
    thr_lat = dest_threshold_lat if dest_threshold_lat is not None else dest_lat
    thr_lon = dest_threshold_lon if dest_threshold_lon is not None else dest_lon
    rwy_hdg = dest_rwy_heading if dest_rwy_heading is not None else bearing_deg(
        dep_lat, dep_lon, dest_lat, dest_lon,
    )

    # ── GROUND (takeoff roll) ────────────────────────────────────────
    roll_nm = takeoff_roll_ft / 6076.0
    n_roll = max(1, int(roll_nm / _STEP_NM))
    lat, lon = dep_lat, dep_lon
    for i in range(n_roll):
        t = i / max(n_roll - 1, 1)
        spd = _lerp(0.0, v_rotate, t)
        points.append(PathPoint(
            lat=lat, lon=lon, alt_ft=dep_alt_ft, speed_kts=spd,
            heading_deg=dep_heading, phase="GROUND",
            gear_down=True, flap_ratio=0.5, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _destination_point(lat, lon, dep_heading, _STEP_M)
        cumul_nm += _STEP_NM

    # ── CLIMB ────────────────────────────────────────────────────────
    # Straight ahead on dep_heading until 600ft AGL, then turn toward dest.
    alt = dep_alt_ft
    alt_per_step = (climb_fpm / 60.0) * (_STEP_M / (v_climb * 0.5144)) * (1.0 / 60.0)
    # More accurate: time per step = dist / speed, then alt_gain = fpm/60 * time
    time_per_step_s = _STEP_M / (v_climb * 0.5144)  # speed in m/s
    alt_per_step = (climb_fpm / 60.0) * time_per_step_s

    hdg = dep_heading
    target_hdg = bearing_deg(lat, lon, thr_lat, thr_lon)
    turn_rate = 3.0  # degrees per step — matches ReactiveFlightDirector

    while alt < cruise_alt_ft:
        agl = alt - dep_alt_ft
        # Hold runway heading below 600ft AGL
        if agl > 600.0:
            err = _wrap180(target_hdg - hdg)
            advance = max(-turn_rate, min(turn_rate, err))
            hdg = (hdg + advance) % 360.0
            target_hdg = bearing_deg(lat, lon, thr_lat, thr_lon)

        gear = agl < 50.0  # retract gear quickly after liftoff
        flap = 0.5 if agl < 300.0 else 0.0  # retract flaps above 300ft AGL

        points.append(PathPoint(
            lat=lat, lon=lon, alt_ft=alt, speed_kts=v_climb,
            heading_deg=hdg, phase="CLIMB",
            gear_down=gear, flap_ratio=flap, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _destination_point(lat, lon, hdg, _STEP_M)
        cumul_nm += _STEP_NM
        alt = min(alt + alt_per_step, cruise_alt_ft)

        # Safety: don't generate more than 5000 points in climb
        if len(points) > 5000:
            break

    # ── CRUISE ───────────────────────────────────────────────────────
    # Level flight toward destination.  Compute TOD first.
    alt_to_lose = cruise_alt_ft - dest_alt_ft
    tod_nm = alt_to_lose / _GLIDE_FT_PER_NM if alt_to_lose > 0 else 0.0
    # Add 3nm for approach segment below glideslope start
    approach_nm = 3.0

    total_dist_nm = haversine_m(lat, lon, thr_lat, thr_lon) / 1852.0
    cruise_nm = max(0.0, total_dist_nm - tod_nm - approach_nm)

    # Accelerate from v_climb to v_cruise over first 2nm of cruise
    accel_nm = min(2.0, cruise_nm / 2.0)
    hdg = bearing_deg(lat, lon, thr_lat, thr_lon)

    n_cruise = max(1, int(cruise_nm / _STEP_NM))
    for i in range(n_cruise):
        t_accel = min(1.0, (i * _STEP_NM) / accel_nm) if accel_nm > 0 else 1.0
        spd = _lerp(v_climb, v_cruise, t_accel)

        # Speed bleed: decelerate from v_cruise to v_approach over last 8nm of cruise
        remaining_cruise_nm = cruise_nm - i * _STEP_NM
        if remaining_cruise_nm < 8.0:
            t_bleed = 1.0 - remaining_cruise_nm / 8.0
            spd = _lerp(v_cruise, v_approach + 10.0, t_bleed)

        hdg = bearing_deg(lat, lon, thr_lat, thr_lon)
        points.append(PathPoint(
            lat=lat, lon=lon, alt_ft=cruise_alt_ft, speed_kts=spd,
            heading_deg=hdg, phase="CRUISE",
            gear_down=False, flap_ratio=0.0, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _destination_point(lat, lon, hdg, _STEP_M)
        cumul_nm += _STEP_NM

        if len(points) > 20000:
            break

    # ── DESCENT (TOD to approach altitude) ───────────────────────────
    descent_target_alt = dest_alt_ft + 500.0  # start approach at 500ft AGL
    n_descent = max(1, int(tod_nm / _STEP_NM))
    alt = cruise_alt_ft
    for i in range(n_descent):
        t = (i + 1) / n_descent
        alt = _lerp(cruise_alt_ft, descent_target_alt, t)
        spd = _lerp(v_approach + 10.0, v_approach, t)
        hdg = bearing_deg(lat, lon, thr_lat, thr_lon)

        points.append(PathPoint(
            lat=lat, lon=lon, alt_ft=alt, speed_kts=spd,
            heading_deg=hdg, phase="DESCENT",
            gear_down=False, flap_ratio=0.0, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _destination_point(lat, lon, hdg, _STEP_M)
        cumul_nm += _STEP_NM

        if len(points) > 25000:
            break

    # ── APPROACH (glideslope to flare height) ────────────────────────
    # Lock onto runway heading, follow 3° glideslope
    approach_hdg = rwy_hdg
    approach_start_alt = alt  # wherever descent ended
    flare_alt = dest_alt_ft + 50.0
    approach_dist = max(0.5, (approach_start_alt - flare_alt) / _GLIDE_FT_PER_NM)
    n_approach = max(1, int(approach_dist / _STEP_NM))

    for i in range(n_approach):
        t = (i + 1) / n_approach
        alt_here = _lerp(approach_start_alt, flare_alt, t)
        # Flap schedule: half flaps when slow enough, full inside last 1nm
        dist_remaining = approach_dist - i * _STEP_NM
        flap = 1.0 if dist_remaining < 1.0 else 0.5

        points.append(PathPoint(
            lat=lat, lon=lon, alt_ft=alt_here, speed_kts=v_approach,
            heading_deg=approach_hdg, phase="APPROACH",
            gear_down=True, flap_ratio=flap, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _destination_point(lat, lon, approach_hdg, _STEP_M)
        cumul_nm += _STEP_NM

    # ── FLARE (last 50ft) ────────────────────────────────────────────
    n_flare = max(3, int(0.3 / _STEP_NM))  # ~0.3nm flare
    for i in range(n_flare):
        t = (i + 1) / n_flare
        alt_here = _lerp(flare_alt, dest_alt_ft + 5.0, t)
        spd = _lerp(v_approach, v_land, t)
        points.append(PathPoint(
            lat=lat, lon=lon, alt_ft=alt_here, speed_kts=spd,
            heading_deg=approach_hdg, phase="FLARE",
            gear_down=True, flap_ratio=1.0, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _destination_point(lat, lon, approach_hdg, _STEP_M)
        cumul_nm += _STEP_NM

    # ── ROLLOUT ──────────────────────────────────────────────────────
    n_rollout = max(3, int(0.3 / _STEP_NM))
    for i in range(n_rollout):
        t = (i + 1) / n_rollout
        spd = _lerp(v_land, 0.0, t)
        points.append(PathPoint(
            lat=lat, lon=lon, alt_ft=dest_alt_ft, speed_kts=spd,
            heading_deg=approach_hdg, phase="ROLLOUT",
            gear_down=True, flap_ratio=1.0, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _destination_point(lat, lon, approach_hdg, _STEP_M)
        cumul_nm += _STEP_NM

    return RibbonPath(points)


# ─── Formatting ───────────────────────────────────────────────────────

def format_ribbon(ribbon: RibbonPath) -> str:
    """Pretty-print a ribbon summary for the log."""
    if not ribbon.points:
        return "[RIBBON] (empty)"

    lines = [f"[RIBBON] {len(ribbon.points)} points, "
             f"{ribbon.points[-1].dist_from_start_nm:.1f} nm total"]

    # Show phase transitions
    prev_phase = ""
    for i, p in enumerate(ribbon.points):
        if p.phase != prev_phase:
            lines.append(
                f"  {p.phase:>10s}  @{p.dist_from_start_nm:6.1f}nm  "
                f"alt={p.alt_ft:7.0f}ft  spd={p.speed_kts:5.0f}kts  "
                f"hdg={p.heading_deg:5.1f}°  "
                f"gear={'DN' if p.gear_down else 'UP'}  flap={p.flap_ratio:.1f}"
            )
            prev_phase = p.phase

    p = ribbon.points[-1]
    lines.append(
        f"       END  @{p.dist_from_start_nm:6.1f}nm  "
        f"alt={p.alt_ft:7.0f}ft  spd={p.speed_kts:5.0f}kts"
    )
    return "\n".join(lines)
