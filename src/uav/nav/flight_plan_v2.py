"""
V2 Flight Plan — Dense ribbon of PathPoints.

The path planner runs ONCE before takeoff.  It computes the entire flight
as a dense ribbon of hundreds of PathPoints, spaced ~0.1 nm apart.
Each point carries lat, lon, altitude, speed, heading, phase, and
gear/flap config.

Architecture: BUILD FORWARD ONLY.
  1. GROUND — takeoff roll on departure heading
  2. CLIMB  — climb to cruise alt, turning toward destination
  3. CRUISE — straight line toward destination (optional, skipped if short)
  4. DESCENT — constant-rate descent, decelerating to approach speed
  5. APPROACH — final segment on runway heading, 3° glideslope
  6. FLARE  — last 50ft, arresting descent
  7. ROLLOUT — on the ground, braking

The landing segments (APPROACH/FLARE/ROLLOUT) are pinned to the destination
runway threshold. The planner computes where to START descending by working
backward from the threshold mathematically, then flies forward to that point.
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
        best_idx = last_idx
        best_dist = math.inf
        end = min(last_idx + 200, len(self.points))
        cos_h = math.cos(math.radians(heading))
        sin_h = math.sin(math.radians(heading))

        for i in range(last_idx, end):
            p = self.points[i]
            dlat = p.lat - lat
            dlon = (p.lon - lon) * math.cos(math.radians(lat))
            along = dlat * cos_h + dlon * sin_h
            if along < -0.001:
                continue
            d = dlat * dlat + dlon * dlon
            if d < best_dist:
                best_dist = d
                best_idx = i

        return best_idx

    def lookahead(self, idx: int, dist_nm: float) -> int:
        target = self.points[idx].dist_from_start_nm + dist_nm
        for i in range(idx, len(self.points)):
            if self.points[i].dist_from_start_nm >= target:
                return i
        return len(self.points) - 1


# ─── Helpers ──────────────────────────────────────────────────────────

def _dest_pt(lat: float, lon: float, bearing_deg_: float,
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

_STEP_NM = 0.1
_STEP_M = _STEP_NM * 1852.0
_GLIDE_FT_PER_NM = 318.0   # 3° glideslope


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
    dest_rwy_length_ft: float = 6000.0,
    cruise_alt_ft: float,
    v_rotate: float = 90.0,
    v_climb: float = 160.0,
    v_cruise: float = 200.0,
    v_approach: float = 83.0,
    v_land: float = 77.0,
    climb_fpm: float = 1600.0,
    takeoff_roll_ft: float = 2000.0,
) -> RibbonPath:
    """Build the full ribbon FORWARD from departure to arrival.

    Key principles:
      - Cruise is straight — no turns. Turns happen only in CLIMB.
      - Short flights skip cruise entirely.
      - Descent uses constant V/S — the plane decelerates and descends simultaneously.
      - Landing segments are pinned to the actual runway threshold.
    """
    pts: list[PathPoint] = []
    cumul_nm = 0.0

    # Resolve destination runway
    thr_lat = dest_threshold_lat if dest_threshold_lat is not None else dest_lat
    thr_lon = dest_threshold_lon if dest_threshold_lon is not None else dest_lon
    approach_hdg = dest_rwy_heading if dest_rwy_heading is not None else bearing_deg(
        dep_lat, dep_lon, dest_lat, dest_lon,
    )

    # ── Pre-compute key distances ───────────────────────────────────
    total_dist_nm = haversine_m(dep_lat, dep_lon, thr_lat, thr_lon) / 1852.0

    # How far does the approach segment extend back from the threshold?
    approach_alt = dest_alt_ft + 500.0  # approach starts 500ft above runway
    approach_nm = max(0.5, (approach_alt - dest_alt_ft - 50.0) / _GLIDE_FT_PER_NM)

    # How far does descent extend back from approach start?
    alt_to_descend = cruise_alt_ft - approach_alt
    descent_nm = alt_to_descend / _GLIDE_FT_PER_NM if alt_to_descend > 0 else 0.0
    # Add ~3nm for deceleration at cruise altitude
    decel_nm = max(1.0, (v_cruise - v_approach) / 40.0)
    total_descent_nm = descent_nm + decel_nm

    # How far does climb take?
    time_per_step_s = _STEP_M / (v_climb * 0.5144)
    alt_per_step = (climb_fpm / 60.0) * time_per_step_s
    climb_steps = max(1, int((cruise_alt_ft - dep_alt_ft) / max(alt_per_step, 0.1)))
    climb_nm = climb_steps * _STEP_NM

    # Is there room for cruise?
    cruise_available_nm = total_dist_nm - climb_nm - total_descent_nm - approach_nm
    has_cruise = cruise_available_nm > 0.5

    # ── 1. GROUND (takeoff roll) ────────────────────────────────────
    lat, lon = dep_lat, dep_lon
    roll_nm = takeoff_roll_ft / 6076.0
    n_roll = max(1, int(roll_nm / _STEP_NM))
    for i in range(n_roll):
        t = i / max(n_roll - 1, 1)
        spd = _lerp(0.0, v_rotate, t)
        pts.append(PathPoint(
            lat=lat, lon=lon, alt_ft=dep_alt_ft, speed_kts=spd,
            heading_deg=dep_heading, phase="GROUND",
            gear_down=True, flap_ratio=0.5, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _dest_pt(lat, lon, dep_heading, _STEP_M)
        cumul_nm += _STEP_NM

    # ── 2. CLIMB — turn toward destination, climb to cruise alt ─────
    alt = dep_alt_ft
    hdg = dep_heading
    turn_rate = 3.0  # degrees per step

    while alt < cruise_alt_ft:
        agl = alt - dep_alt_ft
        # Turn toward destination above 600ft AGL
        if agl > 600.0:
            target_hdg = bearing_deg(lat, lon, thr_lat, thr_lon)
            err = _wrap180(target_hdg - hdg)
            advance = max(-turn_rate, min(turn_rate, err))
            hdg = (hdg + advance) % 360.0

        gear = agl < 50.0
        flap = 0.5 if agl < 300.0 else 0.0

        pts.append(PathPoint(
            lat=lat, lon=lon, alt_ft=alt, speed_kts=v_climb,
            heading_deg=hdg, phase="CLIMB",
            gear_down=gear, flap_ratio=flap, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _dest_pt(lat, lon, hdg, _STEP_M)
        cumul_nm += _STEP_NM
        alt = min(alt + alt_per_step, cruise_alt_ft)
        if len(pts) > 5000:
            break

    # Complete the turn at cruise altitude before entering cruise
    for _ in range(500):
        target_hdg = bearing_deg(lat, lon, thr_lat, thr_lon)
        if abs(_wrap180(target_hdg - hdg)) < 2.0:
            break
        err = _wrap180(target_hdg - hdg)
        advance = max(-2.0, min(2.0, err))
        hdg = (hdg + advance) % 360.0
        pts.append(PathPoint(
            lat=lat, lon=lon, alt_ft=cruise_alt_ft, speed_kts=v_climb,
            heading_deg=hdg, phase="CLIMB",
            gear_down=False, flap_ratio=0.0, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _dest_pt(lat, lon, hdg, _STEP_M)
        cumul_nm += _STEP_NM
        if len(pts) > 5000:
            break

    # ── 3. CRUISE — straight line, no turns ─────────────────────────
    if has_cruise:
        n_cruise = max(1, int(cruise_available_nm / _STEP_NM))
        for i in range(n_cruise):
            # Heading: straight toward destination (recalc for great circle)
            hdg = bearing_deg(lat, lon, thr_lat, thr_lon)
            # Speed: ramp up from v_climb to v_cruise in first 2nm
            t_accel = min(1.0, (i * _STEP_NM) / 2.0)
            spd = _lerp(v_climb, v_cruise, t_accel)
            pts.append(PathPoint(
                lat=lat, lon=lon, alt_ft=cruise_alt_ft, speed_kts=spd,
                heading_deg=hdg, phase="CRUISE",
                gear_down=False, flap_ratio=0.0, dist_from_start_nm=cumul_nm,
            ))
            lat, lon = _dest_pt(lat, lon, hdg, _STEP_M)
            cumul_nm += _STEP_NM
            if len(pts) > 20000:
                break

    # ── 4. DESCENT — decelerate + descend simultaneously ────────────
    # Compute where approach starts (back from threshold along approach heading)
    approach_start_lat, approach_start_lon = _dest_pt(
        thr_lat, thr_lon, (approach_hdg + 180.0) % 360.0,
        approach_nm * 1852.0,
    )

    # Fly from current position toward approach start point,
    # descending and decelerating along the way.
    dist_to_approach = haversine_m(lat, lon, approach_start_lat, approach_start_lon) / 1852.0
    n_descent = max(1, int(dist_to_approach / _STEP_NM))
    descent_start_alt = cruise_alt_ft if has_cruise else alt
    descent_start_spd = v_cruise if has_cruise else v_climb

    for i in range(n_descent):
        t = (i + 1) / n_descent
        # Altitude: linear descent from cruise to approach start altitude
        alt_here = _lerp(descent_start_alt, approach_alt, t)
        # Speed: linear deceleration from cruise/climb speed to approach speed
        spd = _lerp(descent_start_spd, v_approach, t)
        # Heading: toward the approach start point (smooth great circle)
        hdg = bearing_deg(lat, lon, approach_start_lat, approach_start_lon)
        # Flaps: deploy half flaps to help decelerate
        flap = 0.5
        # Gear: deploy when slow enough
        gear = spd < (v_approach + 30.0)

        pts.append(PathPoint(
            lat=lat, lon=lon, alt_ft=alt_here, speed_kts=spd,
            heading_deg=hdg, phase="DESCENT",
            gear_down=gear, flap_ratio=flap, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _dest_pt(lat, lon, hdg, _STEP_M)
        cumul_nm += _STEP_NM
        if len(pts) > 25000:
            break

    # ── 5. APPROACH — on runway heading, 3° glideslope ──────────────
    n_approach = max(1, int(approach_nm / _STEP_NM))
    for i in range(n_approach):
        t = (i + 1) / n_approach
        alt_here = _lerp(approach_alt, dest_alt_ft + 50.0, t)
        flap = 1.0 if t > 0.5 else 0.5

        pts.append(PathPoint(
            lat=lat, lon=lon, alt_ft=alt_here, speed_kts=v_approach,
            heading_deg=approach_hdg, phase="APPROACH",
            gear_down=True, flap_ratio=flap, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _dest_pt(lat, lon, approach_hdg, _STEP_M)
        cumul_nm += _STEP_NM

    # ── 6. FLARE — last 50ft ────────────────────────────────────────
    flare_alt = dest_alt_ft + 50.0
    n_flare = max(3, int(0.3 / _STEP_NM))
    for i in range(n_flare):
        t = (i + 1) / n_flare
        alt_here = _lerp(flare_alt, dest_alt_ft + 5.0, t)
        spd = _lerp(v_approach, v_land, t)
        pts.append(PathPoint(
            lat=lat, lon=lon, alt_ft=alt_here, speed_kts=spd,
            heading_deg=approach_hdg, phase="FLARE",
            gear_down=True, flap_ratio=1.0, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _dest_pt(lat, lon, approach_hdg, _STEP_M)
        cumul_nm += _STEP_NM

    # ── 7. ROLLOUT — on the ground, braking ─────────────────────────
    n_rollout = max(3, int(0.3 / _STEP_NM))
    for i in range(n_rollout):
        t = (i + 1) / n_rollout
        spd = _lerp(v_land, 0.0, t)
        pts.append(PathPoint(
            lat=lat, lon=lon, alt_ft=dest_alt_ft, speed_kts=spd,
            heading_deg=approach_hdg, phase="ROLLOUT",
            gear_down=True, flap_ratio=1.0, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _dest_pt(lat, lon, approach_hdg, _STEP_M)
        cumul_nm += _STEP_NM

    # Recompute cumulative distances from actual point-to-point positions
    _recompute_distances(pts)

    return RibbonPath(pts)


def _recompute_distances(points: list[PathPoint]) -> None:
    """Set dist_from_start_nm based on actual point-to-point distance."""
    if not points:
        return
    points[0].dist_from_start_nm = 0.0
    for i in range(1, len(points)):
        d_m = haversine_m(points[i - 1].lat, points[i - 1].lon,
                          points[i].lat, points[i].lon)
        points[i].dist_from_start_nm = points[i - 1].dist_from_start_nm + d_m / 1852.0


# ─── Formatting ───────────────────────────────────────────────────────

def format_ribbon(ribbon: RibbonPath) -> str:
    """Pretty-print a ribbon summary for the log."""
    if not ribbon.points:
        return "[RIBBON] (empty)"

    lines = [f"[RIBBON] {len(ribbon.points)} points, "
             f"{ribbon.points[-1].dist_from_start_nm:.1f} nm total"]

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
        f"({p.lat:.6f}, {p.lon:.6f})  "
        f"alt={p.alt_ft:7.0f}ft  spd={p.speed_kts:5.0f}kts"
    )
    return "\n".join(lines)
