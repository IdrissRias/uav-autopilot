"""
V2 Flight Plan — Dense ribbon of PathPoints.

The path planner runs ONCE before takeoff.  It computes the entire flight
as a dense ribbon of hundreds of PathPoints, spaced ~0.1 nm apart.
Each point carries lat, lon, altitude, speed, heading, phase, and
gear/flap config.

Build-from-both-ends strategy:
  FORWARD  from departure runway:  GROUND → CLIMB  →  top-of-climb
  BACKWARD from destination runway: ROLLOUT ← FLARE ← APPROACH ← DESCENT ← TOD
  CRUISE fills the gap between top-of-climb and TOD.

Both runway endpoints are pinned to exact database coordinates.
Any accumulated error lives in CRUISE where it doesn't matter.
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
    """Build the full ribbon from departure to arrival.

    Strategy: build from BOTH ENDS and meet in the middle.

      FORWARD  (from departure):  GROUND → CLIMB  →  top-of-climb
      BACKWARD (from destination): ROLLOUT ← FLARE ← APPROACH ← DESCENT ← TOD
      CRUISE connects top-of-climb to TOD.

    Both runway endpoints are exact coordinates — no drift.
    """

    # ── Resolve destination runway ────────────────────────────────────
    thr_lat = dest_threshold_lat if dest_threshold_lat is not None else dest_lat
    thr_lon = dest_threshold_lon if dest_threshold_lon is not None else dest_lon
    rwy_hdg = dest_rwy_heading if dest_rwy_heading is not None else bearing_deg(
        dep_lat, dep_lon, dest_lat, dest_lon,
    )
    approach_hdg = rwy_hdg

    # ══════════════════════════════════════════════════════════════════
    # PART 1: BUILD FORWARD — departure end
    # ══════════════════════════════════════════════════════════════════
    fwd: list[PathPoint] = []
    cumul_nm = 0.0

    # ── GROUND (takeoff roll) ────────────────────────────────────────
    roll_nm = takeoff_roll_ft / 6076.0
    n_roll = max(1, int(roll_nm / _STEP_NM))
    lat, lon = dep_lat, dep_lon
    for i in range(n_roll):
        t = i / max(n_roll - 1, 1)
        spd = _lerp(0.0, v_rotate, t)
        fwd.append(PathPoint(
            lat=lat, lon=lon, alt_ft=dep_alt_ft, speed_kts=spd,
            heading_deg=dep_heading, phase="GROUND",
            gear_down=True, flap_ratio=0.5, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _destination_point(lat, lon, dep_heading, _STEP_M)
        cumul_nm += _STEP_NM

    # ── CLIMB ────────────────────────────────────────────────────────
    alt = dep_alt_ft
    time_per_step_s = _STEP_M / (v_climb * 0.5144)
    alt_per_step = (climb_fpm / 60.0) * time_per_step_s

    hdg = dep_heading
    turn_rate = 3.0  # degrees per step — gentle turn during climb

    while alt < cruise_alt_ft:
        agl = alt - dep_alt_ft
        if agl > 600.0:
            target_hdg = bearing_deg(lat, lon, thr_lat, thr_lon)
            err = _wrap180(target_hdg - hdg)
            advance = max(-turn_rate, min(turn_rate, err))
            hdg = (hdg + advance) % 360.0

        gear = agl < 50.0
        flap = 0.5 if agl < 300.0 else 0.0

        fwd.append(PathPoint(
            lat=lat, lon=lon, alt_ft=alt, speed_kts=v_climb,
            heading_deg=hdg, phase="CLIMB",
            gear_down=gear, flap_ratio=flap, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _destination_point(lat, lon, hdg, _STEP_M)
        cumul_nm += _STEP_NM
        alt = min(alt + alt_per_step, cruise_alt_ft)

        if len(fwd) > 5000:
            break

    # ── TURN TO COURSE — level off at cruise altitude and complete the
    # turn toward destination BEFORE entering cruise. The plane should be
    # wings-level and on heading when cruise begins — no aggressive turns.
    turn_rate_cruise = 2.0  # gentle 2° per step at cruise alt

    for _ in range(500):  # safety cap
        # Recalculate bearing each step since position changes during turn
        target_cruise_hdg = bearing_deg(lat, lon, thr_lat, thr_lon)
        hdg_remaining = abs(_wrap180(target_cruise_hdg - hdg))
        if hdg_remaining < 2.0:
            break

        err = _wrap180(target_cruise_hdg - hdg)
        advance = max(-turn_rate_cruise, min(turn_rate_cruise, err))
        hdg = (hdg + advance) % 360.0

        fwd.append(PathPoint(
            lat=lat, lon=lon, alt_ft=cruise_alt_ft, speed_kts=v_climb,
            heading_deg=hdg, phase="CLIMB",  # still CLIMB phase during turn
            gear_down=False, flap_ratio=0.0, dist_from_start_nm=cumul_nm,
        ))
        lat, lon = _destination_point(lat, lon, hdg, _STEP_M)
        cumul_nm += _STEP_NM

        if len(fwd) > 5000:
            break

    toc_lat, toc_lon = lat, lon

    # ══════════════════════════════════════════════════════════════════
    # PART 2: BUILD BACKWARD — destination end (reversed at the end)
    # ══════════════════════════════════════════════════════════════════
    back_hdg = (approach_hdg + 180.0) % 360.0
    bwd: list[PathPoint] = []

    # ── ROLLOUT (on the ground, braking) ─────────────────────────────
    touchdown_dist_ft = 1000.0
    td_lat, td_lon = _destination_point(thr_lat, thr_lon, approach_hdg,
                                         touchdown_dist_ft * 0.3048)
    rollout_ft = 2000.0
    n_rollout = max(3, int((rollout_ft / 6076.0) / _STEP_NM))
    stop_lat, stop_lon = _destination_point(td_lat, td_lon, approach_hdg,
                                             rollout_ft * 0.3048)
    rlat, rlon = stop_lat, stop_lon
    for i in range(n_rollout):
        t = i / max(n_rollout - 1, 1)
        spd = _lerp(0.0, v_land, t)
        bwd.append(PathPoint(
            lat=rlat, lon=rlon, alt_ft=dest_alt_ft, speed_kts=spd,
            heading_deg=approach_hdg, phase="ROLLOUT",
            gear_down=True, flap_ratio=1.0, dist_from_start_nm=0.0,
        ))
        rlat, rlon = _destination_point(rlat, rlon, back_hdg, _STEP_M)

    # ── FLARE (last 50ft above ground) ──────────────────────────────
    flare_alt = dest_alt_ft + 50.0
    n_flare = max(3, int(0.3 / _STEP_NM))
    flat, flon = td_lat, td_lon
    for i in range(n_flare):
        t = i / max(n_flare - 1, 1)
        alt_here = _lerp(dest_alt_ft + 5.0, flare_alt, t)
        spd = _lerp(v_land, v_approach, t)
        bwd.append(PathPoint(
            lat=flat, lon=flon, alt_ft=alt_here, speed_kts=spd,
            heading_deg=approach_hdg, phase="FLARE",
            gear_down=True, flap_ratio=1.0, dist_from_start_nm=0.0,
        ))
        flat, flon = _destination_point(flat, flon, back_hdg, _STEP_M)

    # ── APPROACH (3° glideslope down to flare height) ────────────────
    approach_start_alt = dest_alt_ft + 500.0
    approach_dist_nm = max(0.5, (approach_start_alt - flare_alt) / _GLIDE_FT_PER_NM)
    n_approach = max(1, int(approach_dist_nm / _STEP_NM))
    alat, alon = flat, flon
    for i in range(n_approach):
        t = i / max(n_approach - 1, 1)
        alt_here = _lerp(flare_alt, approach_start_alt, t)
        dist_from_thr = (n_flare + i) * _STEP_NM
        flap = 1.0 if dist_from_thr < 1.0 else 0.5

        bwd.append(PathPoint(
            lat=alat, lon=alon, alt_ft=alt_here, speed_kts=v_approach,
            heading_deg=approach_hdg, phase="APPROACH",
            gear_down=True, flap_ratio=flap, dist_from_start_nm=0.0,
        ))
        alat, alon = _destination_point(alat, alon, back_hdg, _STEP_M)

    # ── DESCENT — two phases: decelerate level, then descend clean ──
    #
    # Phase A: DECELERATE at cruise altitude with flaps (speed: cruise → approach)
    #   Flaps create drag, plane slows down while staying level.
    #   No altitude change — just speed bleed.
    #
    # Phase B: DESCEND clean at approach speed (alt: cruise → approach start)
    #   Already slow, gentle 3° glideslope, flaps half for drag control.

    # Phase B first (built backward): clean descent at approach speed
    alt_to_lose = cruise_alt_ft - approach_start_alt
    descent_nm = alt_to_lose / _GLIDE_FT_PER_NM if alt_to_lose > 0 else 0.0
    n_descent = max(1, int(descent_nm / _STEP_NM))
    dlat, dlon = alat, alon
    for i in range(n_descent):
        t = i / max(n_descent - 1, 1)
        alt_here = _lerp(approach_start_alt, cruise_alt_ft, t)
        # Already at approach speed — gentle descent, half flaps for drag
        bwd.append(PathPoint(
            lat=dlat, lon=dlon, alt_ft=alt_here, speed_kts=v_approach,
            heading_deg=approach_hdg, phase="DESCENT",
            gear_down=True, flap_ratio=0.5, dist_from_start_nm=0.0,
        ))
        dlat, dlon = _destination_point(dlat, dlon, back_hdg, _STEP_M)

    # Phase A (built backward): decelerate at cruise altitude with flaps
    # Estimate ~2nm to decelerate from cruise to approach speed
    decel_nm = max(1.0, (v_cruise - v_approach) / 40.0)  # ~40 kts per nm
    n_decel = max(1, int(decel_nm / _STEP_NM))
    for i in range(n_decel):
        t = i / max(n_decel - 1, 1)
        # Speed ramps from approach (bottom/start of decel) to cruise (top/end)
        spd = _lerp(v_approach, v_cruise, t)
        bwd.append(PathPoint(
            lat=dlat, lon=dlon, alt_ft=cruise_alt_ft, speed_kts=spd,
            heading_deg=approach_hdg, phase="DESCENT",
            gear_down=False, flap_ratio=0.5, dist_from_start_nm=0.0,
        ))
        dlat, dlon = _destination_point(dlat, dlon, back_hdg, _STEP_M)

    tod_lat, tod_lon = dlat, dlon

    # Reverse so it flows TOWARD the runway
    bwd.reverse()

    # ══════════════════════════════════════════════════════════════════
    # PART 3: CRUISE — connect top-of-climb to TOD
    # ══════════════════════════════════════════════════════════════════
    cruise_dist_nm = haversine_m(toc_lat, toc_lon, tod_lat, tod_lon) / 1852.0
    n_cruise = max(1, int(cruise_dist_nm / _STEP_NM))

    accel_nm = min(2.0, cruise_dist_nm / 2.0)

    cruise_pts: list[PathPoint] = []
    clat, clon = toc_lat, toc_lon
    for i in range(n_cruise):
        t_accel = min(1.0, (i * _STEP_NM) / accel_nm) if accel_nm > 0 else 1.0
        spd = _lerp(v_climb, v_cruise, t_accel)

        remaining_nm = cruise_dist_nm - i * _STEP_NM
        if remaining_nm < 8.0:
            t_bleed = 1.0 - remaining_nm / 8.0
            spd = _lerp(v_cruise, v_approach + 10.0, t_bleed)

        hdg = bearing_deg(clat, clon, tod_lat, tod_lon)
        # Deploy partial flaps in last 4nm of cruise to begin speed bleed
        flap = 0.5 if remaining_nm < 4.0 else 0.0
        cruise_pts.append(PathPoint(
            lat=clat, lon=clon, alt_ft=cruise_alt_ft, speed_kts=spd,
            heading_deg=hdg, phase="CRUISE",
            gear_down=False, flap_ratio=flap, dist_from_start_nm=0.0,
        ))
        clat, clon = _destination_point(clat, clon, hdg, _STEP_M)

    # ══════════════════════════════════════════════════════════════════
    # PART 4: STITCH — fwd + cruise + bwd, recompute cumulative distance
    # ══════════════════════════════════════════════════════════════════
    all_pts = fwd + cruise_pts + bwd
    _recompute_distances(all_pts)

    return RibbonPath(all_pts)


def _recompute_distances(points: list[PathPoint]) -> None:
    """Set dist_from_start_nm on every point based on actual point-to-point distance."""
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
        f"({p.lat:.6f}, {p.lon:.6f})  "
        f"alt={p.alt_ft:7.0f}ft  spd={p.speed_kts:5.0f}kts"
    )
    return "\n".join(lines)
