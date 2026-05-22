"""
V2 Flight Plan — Keyframe ribbon (commander).

The commander describes the flight as ~12 keyframes.  Each keyframe is:
  * a PHASE (GROUND / CLIMB / CRUISE / DESCENT / APPROACH / FLARE / ROLLOUT)
  * a COMMAND BLOCK (throttle, altitude, speed, heading, gear, flap, brake…)
  * a TRIGGER that advances to the next keyframe

The soldier (FlightEngine) walks keyframes forward: every tick, if the
current keyframe's trigger has fired, advance; then emit its targets.

One configuration knob: `v_stall` (default 77).  All other speeds derive
from it via fixed ratios:
    V_rotate   = 1.17 × Vs   (takeoff rotation)
    V_approach = 1.08 × Vs
    V_land     = 1.00 × Vs
    gear_safe  = 1.30 × Vs
    flap_safe  = 1.50 × Vs

Cruise throttle = alt-scaled:  clip(0.72 + 0.005 × alt_kft, 0.72, 0.92).
Speed emerges from throttle+drag; no cruise-speed target.

Landing geometry is backward-solved from the runway threshold, exactly
like v1:  threshold → touchdown → flare_start → approach_start →
descent_start, all collinear with the runway axis.  Cruise/climb aim
at descent_start so the plane arrives on the extended centerline.

A sparse preview (`.points`) is emitted for app map display — the engine
doesn't use it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from uav.nav.geo import bearing_deg, haversine_m


# ═══ Ratios & constants ═════════════════════════════════════════════════

_VS_DEFAULT = 77.0
_RATIO_V_ROTATE = 1.17
_RATIO_V_APPROACH = 1.08
_RATIO_V_LAND = 1.00
_RATIO_GEAR_SAFE = 1.30
_RATIO_FLAP_SAFE = 1.50
_RATIO_V_CRUISE = 1.5        # cruise target speed — was 2.0 which produced
# v_cruise = 196 kts (when v_stall = 98), miles above the SF50's actual
# observed cruise of ~133 kts. The mismatch drove the throttle PID to
# 100% saturation during cruise/transition, which the alt PID couldn't
# counteract via its capped pitch-down, so the plane climbed thousands
# of feet over target. 1.5 × v_stall ≈ 147 kts — comfortably within
# the airframe's clean-config range and close to the envelope's
# observed value.

_GLIDE_FT_PER_NM = 1000.0   # ~9.4° glideslope — doubled from 500 ft/nm to
# halve the descent footprint per user request. The SF50 can dive at this
# rate cleanly with gear+full flaps deployed (~1700 fpm at 100 kts approach
# speed). Earlier test of 1000 ft/nm was rejected when the controllers
# couldn't track it; current build (yesterday's stable PIDs + wide pitch
# authority) handles the steeper angle fine. Halving the descent length
# matters most on short flights (9 nm KUBE↔KRPD): old 500 ft/nm spent
# ~5 nm on descent+approach+decel out of 9 nm total. New 1000 ft/nm cuts
# the descent portion in half, leaving more of the flight as actual cruise.
_APPROACH_ALT_AGL_FT = 600.0  # where approach begins (above threshold alt)
_FLARE_ALT_AGL_FT = 30.0      # where flare begins
_TOUCHDOWN_FT_PAST_THR = 1000.0
_FLARE_LEN_NM = 0.2

# ═══ Measured physics model (SF50, from flight logs) ═══════════════════
# Numbers extracted from tools/measure_physics.py over 12 flights on
# 2026-04-19. Planning numbers are conservative relative to measured
# grand-medians, so the ribbon never assumes better performance than
# the jet can deliver.
#
#   metric               measured   planning  units
#   decel_rate_clean      0.84       0.70     kts/sec (idle, clean, level)
#   sink_rate_drag        50         50       ft/sec  (flaps+gear at idle)
#   climb_rate            34         30       ft/sec  (full throttle, climb)
#
# Source notes:
#  * Decel is the aircraft's drag-limited bleed rate at idle. Used to
#    size the DECELERATE leg so the plane reliably reaches flap_safe
#    before the ribbon demands flap deployment. Flight 20260419-145406
#    crashed because the decel leg was 3 nm but the measured bleed over
#    that distance was only ~36 kts (needed 55) — hence the sizing is
#    now physics-based, not heuristic.
#  * Sink rate in drag config is far steeper than any glideslope we
#    command (14.7° natural vs our 4.7° ask), so descent-slope sizing
#    is bounded by glide geometry, not sink authority.
_DECEL_RATE_CLEAN_KPS = 0.70        # conservative; measured 0.84
_SINK_RATE_DRAG_FPS = 50.0           # reference only; slope is geometry-bounded
_CLIMB_RATE_FPS = 30.0               # conservative; measured 34
_DECEL_SAFETY_MARGIN_NM = 1.0        # extra runway beyond calc'd bleed dist
_DECEL_LEN_MAX_NM = 15.0             # cap for extreme-alt descents
_INTERCEPT_ANGLE_DEG = 30.0  # shallow intercept onto extended centerline
_JOIN_MIN_OFFSET_NM = 3.0    # join point always ≥ 3 nm behind decel_start
_JOIN_MAX_OFFSET_NM = 80.0   # cap so we don't fly wildly off course

# When the single-leg geometric join would force a turn ≥ this many
# degrees at the JOIN point, the planner switches from a single-leg
# (CRUISE → INBOUND) pattern to a two-leg pattern (CRUISE → BASE_LEG
# → INBOUND), inserting a BASE_TURN waypoint perpendicular to the
# centerline on the dep's side. This produces two ~90° turns (still
# well within the SF50's measured roll-limit envelope at cruise
# speed) instead of one 120°+ elbow that the controllers can't
# track. 60° is the empirical edge — above it, prior flights have
# saturated R to ±1.0 and pulled 2g+. See flights 20260419_135940
# and 20260419_145024.
_MAX_SINGLE_LEG_TURN_DEG = 60.0
# Minimum perpendicular offset of the BASE_TURN point from the
# centerline. The actual offset matches the dep's perpendicular
# distance when that's larger, so the cruise leg from dep to
# BASE_TURN is roughly along-axis (small turn at BASE_TURN).
_PATTERN_OFFSET_MIN_NM = 2.0


# ═══ Data model ═════════════════════════════════════════════════════════

@dataclass
class Trigger:
    """Condition that advances the ribbon to the next keyframe.

    Supported kinds:
        speed_gte      airspeed_kts >= value
        speed_lte      airspeed_kts <= value
        agl_gte        agl_ft       >= value
        agl_lte        agl_ft       <= value
        alt_reached    |altitude_ft - value| < 50
        near_point     horizontal distance to (lat, lon) <= value  (nm)
        any            ANY of `subs` fires  (OR over a tuple of Triggers)
        all            ALL of `subs` fire   (AND over a tuple of Triggers)
        always         fires immediately
        never          terminal; never fires
    """
    kind: str = "never"
    value: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    # For kind="any": tuple of sub-triggers that are OR'd together.
    subs: tuple = ()

    def fired(self, telemetry, agl_ft: float) -> bool:
        k = self.kind
        if k == "never":
            return False
        if k == "always":
            return True
        if k == "any":
            return any(s.fired(telemetry, agl_ft) for s in self.subs)
        if k == "all":
            return all(s.fired(telemetry, agl_ft) for s in self.subs)
        spd = telemetry.airspeed_kts
        alt = telemetry.altitude_ft
        if k == "speed_gte":
            return spd >= self.value
        if k == "speed_lte":
            return spd <= self.value
        if k == "agl_gte":
            return agl_ft >= self.value
        if k == "agl_lte":
            return agl_ft <= self.value
        if k == "alt_reached":
            return abs(alt - self.value) < 50.0
        if k == "near_point":
            if not telemetry.has_position():
                return False
            d_nm = haversine_m(
                telemetry.lat_deg, telemetry.lon_deg, self.lat, self.lon,
            ) / 1852.0
            return d_nm <= self.value
        return False

    def describe(self) -> str:
        k = self.kind
        if k in ("never", "always"):
            return k
        if k == "any":
            return " OR ".join(s.describe() for s in self.subs)
        if k == "all":
            return " AND ".join(s.describe() for s in self.subs)
        if k in ("speed_gte", "speed_lte"):
            op = "≥" if k.endswith("gte") else "≤"
            return f"spd {op} {self.value:.0f}kts"
        if k in ("agl_gte", "agl_lte"):
            op = "≥" if k.endswith("gte") else "≤"
            return f"agl {op} {self.value:.0f}ft"
        if k == "alt_reached":
            return f"alt→{self.value:.0f}ft"
        if k == "near_point":
            return f"≤{self.value:.1f}nm to ({self.lat:.4f},{self.lon:.4f})"
        return k


@dataclass
class Keyframe:
    """One stable commander-state.  Fields default to None = 'inherit prior'."""
    name: str
    phase: str                    # GROUND | CLIMB | CRUISE | DESCENT | APPROACH | FLARE | ROLLOUT

    # Throttle
    throttle_mode: str = "explicit"   # "explicit" | "alt_scaled" | "idle" | "speed_pid"
    throttle: Optional[float] = None  # used only when mode == "explicit"

    # Speed target (None = throttle-driven, no speed regulation)
    target_speed_kts: Optional[float] = None

    # Altitude
    alt_mode: str = "target"          # "target" | "hold" | "glideslope"
    target_alt_ft: Optional[float] = None

    # Heading
    heading_mode: str = "hold"        # "hold" | "fixed" | "aim_at" | "dep_runway" | "dest_runway"
    target_heading_deg: Optional[float] = None
    aim_lat: float = 0.0
    aim_lon: float = 0.0

    # Levers
    gear_down: Optional[bool] = None
    flap_ratio: Optional[float] = None
    brake_ratio: Optional[float] = None

    # Commander-issued limits (None = full authority)
    roll_limit: Optional[float] = None
    pitch_limit: Optional[float] = None
    pitch_down_limit: Optional[float] = None  # opt-in nose-down cap (descent only)

    # Yaw coordination (for takeoff roll)
    yaw_hold: Optional[bool] = None
    yaw_kp: Optional[float] = None
    yaw_limit: Optional[float] = None

    # Trigger that advances to the next keyframe
    trigger: Trigger = field(default_factory=lambda: Trigger("never"))


@dataclass
class Geometry:
    """Backward-solved landing geometry.  Computed once per flight plan."""
    dep_lat: float
    dep_lon: float
    dep_alt_ft: float
    dep_heading: float
    # Runway
    thr_lat: float
    thr_lon: float
    thr_alt_ft: float
    rwy_heading: float
    # Derived landing chain (all collinear with runway axis)
    touchdown_lat: float
    touchdown_lon: float
    flare_start_lat: float
    flare_start_lon: float
    approach_start_lat: float
    approach_start_lon: float
    approach_start_alt_ft: float       # = thr_alt + 600
    descent_start_lat: float
    descent_start_lon: float
    decel_start_lat: float                 # 5 nm back from descent_start
    decel_start_lon: float
    join_lat: float                        # intercept point on extended centerline
    join_lon: float
    cruise_alt_ft: float
    # Speeds (derived from Vs)
    v_stall: float
    v_rotate: float
    v_cruise: float
    v_approach: float
    v_land: float
    gear_safe_kts: float
    flap_safe_kts: float
    # Optional BASE_TURN point — populated only when the single-leg
    # cruise→join geometry would produce a turn > _MAX_SINGLE_LEG_TURN_DEG
    # at JOIN. When present, an extra BASE_LEG keyframe is inserted
    # between CRUISE and INBOUND: CRUISE aims at BASE_TURN, BASE_LEG
    # aims at JOIN, INBOUND aims at decel_start (unchanged). The two
    # ~90° turns this produces stay inside the SF50's roll-limit
    # envelope; the alternative 120°+ single-leg turn does not.
    base_turn_lat: Optional[float] = None
    base_turn_lon: Optional[float] = None


# Legacy PathPoint retained for the map-preview list (`.points`).
@dataclass
class PathPoint:
    lat: float
    lon: float
    alt_ft: float
    speed_kts: float
    heading_deg: float
    phase: str
    gear_down: bool = True
    flap_ratio: float = 0.0
    throttle: Optional[float] = None
    dist_from_start_nm: float = 0.0


@dataclass
class Ribbon:
    """Ribbon = keyframes (commander) + geometry (reference frame) + sparse
    preview points (for app map display)."""
    keyframes: List[Keyframe]
    geometry: Geometry
    points: List[PathPoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.keyframes)


# ═══ Helpers ════════════════════════════════════════════════════════════

def _dest_pt(lat: float, lon: float, bearing: float, dist_m: float) -> tuple[float, float]:
    R = 6_371_000.0
    d = dist_m / R
    brng = math.radians(bearing)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d)
        + math.cos(lat1) * math.sin(d) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def _bezier_corner(
    in_lat: float, in_lon: float,
    vertex_lat: float, vertex_lon: float,
    out_lat: float, out_lon: float,
    n_steps: int = 20,
) -> list:
    """Quadratic Bézier arc through `n_steps` interior points from
    (in_lat, in_lon) to (out_lat, out_lon), bending toward the corner
    vertex (vertex_lat, vertex_lon).

    The curve never actually reaches the vertex — that's the whole
    point: the polyline visits a smooth arc instead of a sharp elbow,
    which the L1 follower can chase without the plane banking
    impossibly hard. Used to round corners at BASE_TURN, JOIN, and
    the lift-off corner.

    Returns interior points only (t = 1/(n+1) … n/(n+1)); the caller
    already has the in and out endpoints.
    """
    pts = []
    for i in range(1, n_steps + 1):
        t = i / (n_steps + 1)
        u = 1.0 - t
        lat = u * u * in_lat + 2.0 * u * t * vertex_lat + t * t * out_lat
        lon = u * u * in_lon + 2.0 * u * t * vertex_lon + t * t * out_lon
        pts.append((lat, lon))
    return pts


def _course_reversal_arc(
    start_lat: float, start_lon: float,
    initial_brg_deg: float, target_brg_deg: float,
    turn_radius_nm: float,
    n_steps: int = 24,
) -> list:
    """Dubins-style true circular arc from `start` heading
    `initial_brg_deg`, turning to end up heading `target_brg_deg`,
    with the plane's actual turn radius. Picks left or right turn
    automatically — whichever requires less rotation. Returns interior
    arc points (start point not included).

    Used for the lift-off corner when the departure heading and the
    cruise direction differ by enough that a quadratic Bézier would
    degenerate (>~120°). A real arc has constant curvature the plane
    can fly at a fixed bank angle.
    """
    delta = ((target_brg_deg - initial_brg_deg + 540.0) % 360.0) - 180.0
    if abs(delta) < 1e-3:
        return []
    turn_sign = 1.0 if delta > 0 else -1.0
    total_turn_rad = math.radians(abs(delta))
    perp_brg = (initial_brg_deg + 90.0 * turn_sign) % 360.0
    centre_lat, centre_lon = _dest_pt(
        start_lat, start_lon, perp_brg, turn_radius_nm * 1852.0,
    )
    # bearing from centre back to the start point
    bearing_to_start = (perp_brg + 180.0) % 360.0
    pts = []
    for i in range(1, n_steps + 1):
        t = i / n_steps
        ang = (bearing_to_start
               + turn_sign * math.degrees(total_turn_rad) * t) % 360.0
        pt = _dest_pt(centre_lat, centre_lon, ang,
                      turn_radius_nm * 1852.0)
        pts.append(pt)
    return pts


def _interp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _decel_len_nm(v_from_kts: float, v_to_kts: float) -> float:
    """Distance needed to bleed from v_from→v_to at idle, clean config,
    level flight. Returns nautical miles including a safety margin.

    Physics: at a constant decel rate R (kts/sec), bleeding delta
    ΔV (kts) takes t = ΔV/R seconds. Forward distance covered is
    V_avg × t, where V_avg = (v_from + v_to) / 2 because velocity
    decreases linearly under constant decel. Converting kts→nm/s
    (÷3600) and adding a safety margin gives the leg length.
    """
    delta_kts = max(0.0, v_from_kts - v_to_kts)
    time_s = delta_kts / _DECEL_RATE_CLEAN_KPS
    avg_kts = (v_from_kts + v_to_kts) / 2.0
    dist_nm = (avg_kts / 3600.0) * time_s
    return min(_DECEL_LEN_MAX_NM, dist_nm + _DECEL_SAFETY_MARGIN_NM)


# ═══ Join-point (intercept) ═══════════════════════════════════════════

def _compute_join_geometry(
    dep_lat: float, dep_lon: float,
    thr_lat: float, thr_lon: float,
    back_hdg: float,
    decel_lat: float, decel_lon: float,
) -> tuple[float, float, Optional[float], Optional[float]]:
    """Returns (join_lat, join_lon, base_turn_lat | None, base_turn_lon | None).

    Two regimes:
      1. **Single-leg join** (preferred): when geometry allows a 30°
         intercept onto the extended centerline at a JOIN point that
         sits behind the decel buffer, return only (join_lat, join_lon,
         None, None).  Cruise → INBOUND, one turn at the join.
      2. **Two-leg pattern** (fallback): when the single-leg turn at
         the join would exceed _MAX_SINGLE_LEG_TURN_DEG (60°), insert
         a BASE_TURN waypoint perpendicular to the centerline on the
         dep's side.  The plane flies CRUISE → BASE_TURN (≈ along
         centerline direction) → BASE_LEG (perpendicular to runway,
         crossing onto centerline) → INBOUND.  Two ~90° turns instead
         of one >60° elbow.

    Flat-earth nm projection around the threshold; good to ~200 nm.

    Geometry:
        • t = along-track position from threshold, positive in back_hdg.
        • F = foot of perpendicular from dep onto the centerline.
        • d_perp = |dep − F|.
        • Ideal close 30° intercept: t_J = t_F − d_perp / tan(30°).
        • Clamp: t_decel + 3 nm ≤ t_J ≤ t_decel + 80 nm.
        • If clamped t_J produces turn-at-join > 60°: switch to pattern.
    """
    mean_lat_rad = math.radians((thr_lat + dep_lat) / 2.0)
    dep_e = (dep_lon - thr_lon) * 60.0 * math.cos(mean_lat_rad)
    dep_n = (dep_lat - thr_lat) * 60.0
    decel_e = (decel_lon - thr_lon) * 60.0 * math.cos(mean_lat_rad)
    decel_n = (decel_lat - thr_lat) * 60.0

    bh_rad = math.radians(back_hdg)
    ue, un = math.sin(bh_rad), math.cos(bh_rad)  # along-track unit vector (east, north)

    t_dep = dep_e * ue + dep_n * un
    t_decel = decel_e * ue + decel_n * un

    perp_e = dep_e - t_dep * ue
    perp_n = dep_n - t_dep * un
    d_perp = math.hypot(perp_e, perp_n)

    intercept_off = d_perp / math.tan(math.radians(_INTERCEPT_ANGLE_DEG))
    t_min = t_decel + _JOIN_MIN_OFFSET_NM
    t_max = t_decel + _JOIN_MAX_OFFSET_NM

    t_ideal = t_dep - intercept_off
    t_join = max(t_ideal, t_min)
    t_join = min(t_join, t_max)

    # Compute join lat/lon.
    join_lat, join_lon = _dest_pt(thr_lat, thr_lon, back_hdg, t_join * 1852.0)

    # Estimate the actual turn at the JOIN for a single-leg path.
    # cruise heading = bearing(dep, join); inbound heading = bearing(
    # join, decel) ≈ rwy_hdg (since join is behind decel).
    cruise_hdg = bearing_deg(dep_lat, dep_lon, join_lat, join_lon)
    inbound_hdg = bearing_deg(join_lat, join_lon, decel_lat, decel_lon)
    turn_at_join = abs(((cruise_hdg - inbound_hdg + 180.0) % 360.0) - 180.0)

    if turn_at_join <= _MAX_SINGLE_LEG_TURN_DEG:
        return (join_lat, join_lon, None, None)

    # Two-leg pattern. Place BASE_TURN perpendicular to the centerline
    # on the dep's side, at along-track t_join (so BASE_LEG is purely
    # perpendicular to the runway). Pattern offset matches dep's
    # perpendicular distance when reasonable, so CRUISE from dep to
    # BASE_TURN is roughly along-axis (small turn at BASE_TURN).
    pattern_offset = max(_PATTERN_OFFSET_MIN_NM, min(d_perp, 5.0))

    # Unit perpendicular vector from foot-of-perp toward dep (in e/n).
    if d_perp > 1e-6:
        perp_ue = perp_e / d_perp
        perp_un = perp_n / d_perp
    else:
        # Dep is exactly on centerline — pick an arbitrary side
        # (right-hand normal of back_hdg).
        perp_ue, perp_un = -un, ue

    # BASE_TURN point on dep's side of centerline at along-track t_join.
    base_e = t_join * ue + pattern_offset * perp_ue
    base_n = t_join * un + pattern_offset * perp_un
    base_lat = thr_lat + base_n / 60.0
    base_lon = thr_lon + base_e / (60.0 * math.cos(mean_lat_rad))

    # Diagnostic log so the pattern path is visible in flight logs.
    print(
        f"[RIBBON] Single-leg turn at JOIN would be {turn_at_join:.0f}° "
        f"(> {_MAX_SINGLE_LEG_TURN_DEG:.0f}°). Inserting BASE_TURN at "
        f"perp offset {pattern_offset:.1f}nm; pattern is now "
        f"CRUISE → BASE → INBOUND with two ~90° turns."
    )

    return (join_lat, join_lon, base_lat, base_lon)


# ═══ Geometry solver ═══════════════════════════════════════════════════

def _solve_geometry(
    *,
    dep_lat: float,
    dep_lon: float,
    dep_alt_ft: float,
    dep_heading: float,
    dest_lat: float,
    dest_lon: float,
    dest_alt_ft: float,
    dest_rwy_heading: Optional[float],
    dest_threshold_lat: Optional[float],
    dest_threshold_lon: Optional[float],
    cruise_alt_ft: float,
    v_stall: float,
) -> Geometry:
    # Resolve runway
    thr_lat = dest_threshold_lat if dest_threshold_lat is not None else dest_lat
    thr_lon = dest_threshold_lon if dest_threshold_lon is not None else dest_lon
    rwy_hdg = (dest_rwy_heading
               if dest_rwy_heading is not None
               else bearing_deg(dep_lat, dep_lon, dest_lat, dest_lon))
    back_hdg = (rwy_hdg + 180.0) % 360.0

    # Touchdown: 1000 ft past threshold along runway
    td_lat, td_lon = _dest_pt(thr_lat, thr_lon, rwy_hdg, _TOUCHDOWN_FT_PAST_THR * 0.3048)

    # Flare start: touchdown - flare_len back
    flare_lat, flare_lon = _dest_pt(td_lat, td_lon, back_hdg, _FLARE_LEN_NM * 1852.0)

    # Approach start: 3° glideslope from 600ft AGL down to 30ft AGL
    approach_alt = dest_alt_ft + _APPROACH_ALT_AGL_FT
    approach_descent_ft = _APPROACH_ALT_AGL_FT - _FLARE_ALT_AGL_FT
    approach_len_nm = max(0.5, approach_descent_ft / _GLIDE_FT_PER_NM)
    app_lat, app_lon = _dest_pt(flare_lat, flare_lon, back_hdg, approach_len_nm * 1852.0)

    # Descent start: glideslope from cruise_alt down to approach_alt
    alt_to_descend = max(0.0, cruise_alt_ft - approach_alt)
    descent_len_nm = alt_to_descend / _GLIDE_FT_PER_NM if alt_to_descend > 0 else 0.0
    desc_lat, desc_lon = _dest_pt(app_lat, app_lon, back_hdg, descent_len_nm * 1852.0)

    # Speeds (computed early so decel-leg sizing can use them)
    v_rot = _RATIO_V_ROTATE * v_stall
    v_cru = _RATIO_V_CRUISE * v_stall
    v_app = _RATIO_V_APPROACH * v_stall
    v_land = _RATIO_V_LAND * v_stall
    gear_safe = _RATIO_GEAR_SAFE * v_stall
    flap_safe = _RATIO_FLAP_SAFE * v_stall

    # Decel start: physics-sized from measured decel rate (0.70 kts/sec
    # at idle, clean, level), not an altitude heuristic. Distance = what
    # the jet actually needs to bleed Vcruise→Vfe at drag-limited idle.
    # See _decel_len_nm() for math and _DECEL_RATE_CLEAN_KPS for source.
    decel_len_nm = _decel_len_nm(v_cru, flap_safe)
    decel_lat, decel_lon = _dest_pt(desc_lat, desc_lon, back_hdg, decel_len_nm * 1852.0)

    # Join point: where cruise meets the extended centerline at a 30° angle.
    # Cruise aims here, then a short INBOUND leg runs along the centerline to
    # decel_start.  Prevents the 90° elbow that used to happen when dep was
    # off the runway axis.  When a single-leg geometry would force a turn
    # > 60° at JOIN, base_lat/lon are populated and an extra BASE_LEG
    # keyframe is inserted by _build_keyframes.
    join_lat, join_lon, base_lat, base_lon = _compute_join_geometry(
        dep_lat, dep_lon, thr_lat, thr_lon,
        back_hdg, decel_lat, decel_lon,
    )

    return Geometry(
        dep_lat=dep_lat, dep_lon=dep_lon,
        dep_alt_ft=dep_alt_ft, dep_heading=dep_heading,
        thr_lat=thr_lat, thr_lon=thr_lon,
        thr_alt_ft=dest_alt_ft,
        rwy_heading=rwy_hdg,
        touchdown_lat=td_lat, touchdown_lon=td_lon,
        flare_start_lat=flare_lat, flare_start_lon=flare_lon,
        approach_start_lat=app_lat, approach_start_lon=app_lon,
        approach_start_alt_ft=approach_alt,
        descent_start_lat=desc_lat, descent_start_lon=desc_lon,
        decel_start_lat=decel_lat, decel_start_lon=decel_lon,
        join_lat=join_lat, join_lon=join_lon,
        cruise_alt_ft=cruise_alt_ft,
        v_stall=v_stall, v_rotate=v_rot, v_cruise=v_cru,
        v_approach=v_app, v_land=v_land,
        gear_safe_kts=gear_safe, flap_safe_kts=flap_safe,
        base_turn_lat=base_lat, base_turn_lon=base_lon,
    )


# ═══ Keyframe builder ══════════════════════════════════════════════════

def _build_keyframes(g: Geometry) -> List[Keyframe]:
    """Emit the ~12-row flight plan.

    When `g.base_turn_lat` is populated (close-in departure that can't
    do a clean single-leg join), CLIMB_CLEAN / TRANSITION / CRUISE all
    aim at BASE_TURN instead of JOIN, and an extra BASE_LEG keyframe
    is inserted between CRUISE and INBOUND. This produces a proper
    two-turn pattern (≈90° at BASE_TURN, ≈90° at JOIN) instead of
    the single 120°+ elbow that wrecks the controllers.
    """
    # First cruise-and-climb target: BASE_TURN if pattern is active,
    # otherwise JOIN.
    has_base = (
        g.base_turn_lat is not None and g.base_turn_lon is not None
    )
    cruise_aim_lat = g.base_turn_lat if has_base else g.join_lat
    cruise_aim_lon = g.base_turn_lon if has_base else g.join_lon

    keyframes: List[Keyframe] = [
        # ── 1. TAKEOFF_ROLL ──────────────────────────────────────────
        # Full throttle, flaps 0.5 for lift, hold runway heading with yaw
        # coordination.  Advances when airspeed crosses V_rotate.
        Keyframe(
            name="TAKEOFF_ROLL", phase="GROUND",
            throttle_mode="explicit", throttle=1.0,
            alt_mode="hold",
            heading_mode="dep_runway",
            gear_down=True, flap_ratio=0.5, brake_ratio=0.0,
            yaw_hold=True, yaw_kp=0.02, yaw_limit=0.35,
            trigger=Trigger("speed_gte", value=g.v_rotate),
        ),

        # ── 2. CLIMB_ROTATE ──────────────────────────────────────────
        # Rotation + initial climb.  Hold departure runway heading, climb
        # to cruise alt.  Gear still down, flaps still 0.5.  Pitch limit
        # 0.20 (≈12°) so alt-PID can't command 90° nose-up and stall us.
        # Advance when clear of obstacles — agl ≥ 50 ft.
        Keyframe(
            name="CLIMB_ROTATE", phase="CLIMB",
            throttle_mode="explicit", throttle=0.95,
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="dep_runway",
            gear_down=True, flap_ratio=0.5,
            pitch_limit=0.20,
            trigger=Trigger("agl_gte", value=50.0),
        ),

        # ── 3. CLIMB_GEAR_UP ─────────────────────────────────────────
        # Gear retracted, still straight-out on dep-runway heading.
        # Pitch still capped.  Advance when safely above pattern
        # altitude — agl ≥ 300 ft.
        Keyframe(
            name="CLIMB_GEAR_UP", phase="CLIMB",
            throttle_mode="explicit", throttle=0.95,
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="dep_runway",
            gear_down=False, flap_ratio=0.5,
            pitch_limit=0.20,
            trigger=Trigger("agl_gte", value=300.0),
        ),

        # ── 4. CLIMB_CLEAN ───────────────────────────────────────────
        # Flaps retracted, turn toward the join point (where we meet the
        # extended runway centerline at a 30° intercept).  Pitch still
        # capped.  roll_limit kept shallow (0.25 ≈ ~15° bank) because
        # CLIMB_CLEAN begins at 300 AGL — a tight aim_at turn at
        # saturation (flight 20260419_145024 drove R to -1.0 at ~500 AGL
        # and hit terrain). Advance when we reach cruise altitude — next
        # keyframe (TRANSITION) handles speed accel at cruise alt.
        Keyframe(
            name="CLIMB_CLEAN", phase="CLIMB",
            throttle_mode="explicit", throttle=0.95,
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="aim_at",
            aim_lat=cruise_aim_lat, aim_lon=cruise_aim_lon,
            gear_down=False, flap_ratio=0.0,
            roll_limit=0.25,
            pitch_limit=0.20,
            trigger=Trigger("alt_reached", value=g.cruise_alt_ft),
        ),

        # ── 4b. TRANSITION ───────────────────────────────────────────
        # Bridge between CLIMB_CLEAN and CRUISE. Previously CRUISE entered
        # directly from CLIMB_CLEAN — plane arrived at cruise alt at ~115
        # kts while CRUISE instantly demanded v_cruise (154 kts). The
        # 39-kt step error saturated the speed PID (throttle 1.0) AND the
        # alt PID was still pitching up (alt -48 below target): both PIDs
        # pumped energy in-phase → +1300 ft alt overshoot, then dive to
        # 190 kts. Flight 20260419_184932 made this far worse: an earlier
        # attempt put a compound "alt AND speed" trigger on CLIMB_CLEAN
        # itself — CLIMB_CLEAN has no pitch_down_limit, so when plane
        # overshot cruise alt the alt PID dove pitch to -0.44, throttle
        # 0.95 held, speed ran away to 272 kts before the compound fired.
        # Plane entered CRUISE massively over-energy and blew past the
        # destination.
        #
        # Proper fix: an explicit TRANSITION keyframe that levels off at
        # cruise alt with TIGHT pitch bounds (±0.08 ≈ ±4.6°) and speed-PID
        # throttle targeting v_cruise. Plane can't dive (bounded) and
        # can't climb (bounded); throttle modulates to build speed while
        # level. Exits when speed within 5 kts of v_cruise, handing off to
        # CRUISE with both PIDs near zero-error → no saturation, no phugoid.
        # "Fix the ribbon, keep the follower dumb."
        Keyframe(
            name="TRANSITION", phase="CRUISE",
            throttle_mode="speed_pid",
            target_speed_kts=g.v_cruise,
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="aim_at",
            aim_lat=cruise_aim_lat, aim_lon=cruise_aim_lon,
            gear_down=False, flap_ratio=0.0,
            roll_limit=0.25,
            # Pitch_down widened from 0.08 → 0.40. The previous 0.08 cap
            # meant when the plane was over-altitude (e.g. CLIMB
            # overshoot, classic decoupled-PID failure where speed PID
            # at 100% throttle out-pumps the alt PID's capped nose-down),
            # the alt PID could only command 5° nose-down — not enough
            # to bleed energy faster than the speed PID was adding it.
            # Plane climbed forever. Observed: 8000 ft over target at
            # 100% throttle. New 0.40 (~24° nose-down) lets the plane
            # actually descend when it needs to.
            pitch_limit=0.08, pitch_down_limit=0.40,
            trigger=Trigger("speed_gte", value=g.v_cruise - 5.0),
        ),

        # ── 5. CRUISE ────────────────────────────────────────────────
        # Speed-PID throttle holds Vcruise; alt-hold holds cruise altitude.
        # Both metrics tracked — no more free-drifting speed or altitude.
        # Aim at the join point (30° intercept with runway axis).
        Keyframe(
            name="CRUISE", phase="CRUISE",
            throttle_mode="speed_pid",
            target_speed_kts=g.v_cruise,
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="aim_at",
            aim_lat=cruise_aim_lat, aim_lon=cruise_aim_lon,
            gear_down=False, flap_ratio=0.0,
            # Bank ≤ ~31° during intercept turn (observed 60° in flight
            # 135940 caused 2g at cruise speed). Pitch ≤ ~14° nose-up cap.
            roll_limit=0.35, pitch_limit=0.15,
            trigger=Trigger("near_point", value=1.0,
                           lat=cruise_aim_lat, lon=cruise_aim_lon),
        ),

        # ── 6. INBOUND ───────────────────────────────────────────────
        # On the extended centerline.  Still Vcruise + cruise alt; this
        # is the gentle roll-out of the intercept turn toward decel_start.
        Keyframe(
            name="INBOUND", phase="CRUISE",
            throttle_mode="speed_pid",
            target_speed_kts=g.v_cruise,
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="aim_at",
            aim_lat=g.decel_start_lat, aim_lon=g.decel_start_lon,
            gear_down=False, flap_ratio=0.0,
            roll_limit=0.25, pitch_limit=0.15,  # gentle roll-out, level flight
            trigger=Trigger("near_point", value=0.3,
                           lat=g.decel_start_lat, lon=g.decel_start_lon),
        ),

        # ── 7. DECELERATE ────────────────────────────────────────────
        # Level flight at cruise altitude, clean config, IDLE power.
        # Trigger is speed-only — we MUST hit flap_safe before
        # descending. Flight 20260419_145404 released DECELERATE at 132
        # kts via a near_point fallback; DESCENT then converted PE→KE
        # faster than clean-config drag could bleed, speed climbed to
        # 168 kts and the jet hit terrain. Decel zone was bumped to
        # min 5 nm (was 3) to give ~110 s of idle-coast from Vcruise
        # (170) to flap_safe (115) at ~0.5 kt/sec.
        #
        # Flight 20260419 showed speed_pid getting stuck: as error
        # closed toward 0, the PID eased throttle back toward 0.17–0.20,
        # finding equilibrium with drag at ~133 kts. The plane stopped
        # decelerating 18 kts short of target. `idle` forces thrust
        # fully off for the entire phase — target_speed_kts is kept
        # for the trigger and telemetry display but throttle is flat
        # zero until we bleed through 115.
        Keyframe(
            name="DECELERATE", phase="CRUISE",
            throttle_mode="idle",
            target_speed_kts=g.flap_safe_kts,
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="aim_at",
            aim_lat=g.descent_start_lat, aim_lon=g.descent_start_lon,
            gear_down=False, flap_ratio=0.0,
            roll_limit=0.25, pitch_limit=0.15,
            trigger=Trigger("speed_lte", value=g.flap_safe_kts),
        ),

        # ── 8. DESCENT ───────────────────────────────────────────────
        # Idle-coast descent: we're too fast and too high entering this
        # phase — a fixed-thrust setting (previously 0.30) plus a
        # glideslope pitch loop feeds gravitational PE into airspeed
        # whenever the jet drops below the slope. Idle throttle + a
        # pitch-down cap prevents the pitch loop from staging a
        # catch-up dive that shreds the flap envelope (flight
        # 20260419_142509: pitch -0.53, 167 kts with flaps deployed).
        # No flaps yet — advance when speed bleeds below flap_safe AND
        # we're below 2000 ft AGL.
        Keyframe(
            name="DESCENT", phase="DESCENT",
            throttle_mode="speed_pid",
            target_speed_kts=g.flap_safe_kts,  # hold ~115 kts during clean descent
            alt_mode="glideslope",
            heading_mode="aim_at",
            aim_lat=g.approach_start_lat, aim_lon=g.approach_start_lon,
            pitch_limit=0.15,
            # pitch_down_limit 0.40 (~23° stick) so plane can actually
            # descend when behind the glideslope. Previous 0.15 (~8.6°)
            # was too restrictive: flight 20260419_173924 arrived at
            # threshold 1700ft HIGH because pitch was pegged at -0.15
            # but only produced +3° nose-up (thrust + trim overpowering
            # the gentle stick). Speed PID now (kp=0.05) will slam
            # throttle to idle on any overspeed, so PE→KE conversion
            # during a steeper dive won't shred the flap envelope.
            pitch_down_limit=0.40,
            gear_down=False, flap_ratio=0.0,
            # Flaps only deploy when BOTH: speed below Vfe AND we're
            # low enough (2000ft AGL). Using `any` let flaps out at 170
            # kts in flight 135940; `all` prevents that structural
            # violation.
            trigger=Trigger(
                "all",
                subs=(
                    Trigger("speed_lte", value=g.flap_safe_kts),
                    Trigger("agl_lte", value=2000.0),
                ),
            ),
        ),

        # ── 9. DESCENT_FLAP ──────────────────────────────────────────
        # Half flaps deployed (more drag, less thrust needed).  Idle
        # throttle + pitch cap here for the same reason as DESCENT:
        # holding 0.25 thrust while glideslope pitch pointed the nose
        # down let the jet accelerate from 115 → 167 kts with flaps
        # extended (flight 20260419_142509). Advance when speed bleeds
        # below gear_safe AND we're at 1200 ft AGL, OR 800 AGL reached.
        Keyframe(
            name="DESCENT_FLAP", phase="DESCENT",
            throttle_mode="speed_pid",
            target_speed_kts=g.gear_safe_kts,  # bleed from ~115 to ~100 kts
            alt_mode="glideslope",
            heading_mode="aim_at",
            aim_lat=g.approach_start_lat, aim_lon=g.approach_start_lon,
            pitch_limit=0.15,
            pitch_down_limit=0.40,
            gear_down=False, flap_ratio=0.5,
            # Gear drop logic:
            #   (A) Normal: speed ≤ Vlo AND ≤ 1200 AGL  — clean config
            #       change while still high enough to stabilise.
            #   (B) Safety floor: ≤ 800 AGL regardless of speed.
            # Flight 20260419_180202 spent the entire descent in this
            # keyframe: glideslope PE→KE held speed at 111 kts (above
            # gear_safe 100) with throttle already at idle (0.06). Gear
            # never came down, plane touched down wheels-up. The AGL
            # floor guarantees gear by short final even if speed won't
            # bleed. At 111 kts we're far below Vlo (~170 kts for SF50)
            # so structural concern that motivated the `all` gate
            # doesn't apply here. Flight 135940's 166-kt gear drop is
            # still prevented because we don't reach 800 AGL with 166
            # kts on the glideslope.
            trigger=Trigger(
                "any",
                subs=(
                    Trigger(
                        "all",
                        subs=(
                            Trigger("speed_lte", value=g.gear_safe_kts),
                            Trigger("agl_lte", value=1200.0),
                        ),
                    ),
                    Trigger("agl_lte", value=800.0),
                ),
            ),
        ),

        # ── 10. DESCENT_GEAR ─────────────────────────────────────────
        # Gear down (even more drag).  Idle throttle — gear+flap drag is
        # plenty to stabilise speed on a 3° slope; fixed thrust here
        # had the same catch-up-dive failure mode as DESCENT_FLAP.
        # Advance at 500 ft AGL → APPROACH.
        Keyframe(
            name="DESCENT_GEAR", phase="DESCENT",
            throttle_mode="speed_pid",
            target_speed_kts=g.v_approach,  # stabilise at ~83 kts before flare
            alt_mode="glideslope",
            heading_mode="aim_at",
            aim_lat=g.flare_start_lat, aim_lon=g.flare_start_lon,
            pitch_limit=0.15,
            pitch_down_limit=0.40,
            gear_down=True, flap_ratio=0.5,
            trigger=Trigger("agl_lte", value=500.0),
        ),

        # ── 11. APPROACH ─────────────────────────────────────────────
        # Full flaps + gear down on the glideslope.  Previously idle-only
        # throttle here: when the prior descent phase delivered us below
        # V_approach (flight 20260419_143638 arrived at APPROACH at 79 kts
        # and bled to 60 kts with pitch saturated +1.0 — imminent stall),
        # idle meant no thrust to defend minimum speed.  Now speed_pid
        # with V_approach target so throttle adds thrust below ~83 kts.
        # Aim at the touchdown point so the ribbon corridor stays tight.
        # Advance at 30 ft AGL → FLARE.
        Keyframe(
            name="APPROACH", phase="APPROACH",
            throttle_mode="speed_pid",
            target_speed_kts=g.v_approach,
            alt_mode="glideslope",
            heading_mode="aim_at",
            aim_lat=g.touchdown_lat, aim_lon=g.touchdown_lon,
            pitch_limit=0.15,
            # APPROACH stays tighter than outer DESCENT keyframes (0.20)
            # because we're short final: a big dive here would smash the
            # gear. Still relaxed from 0.15 so we can catch a
            # behind-schedule glideslope without the previous failure mode.
            pitch_down_limit=0.20,
            gear_down=True, flap_ratio=1.0,
            trigger=Trigger("agl_lte", value=_FLARE_ALT_AGL_FT),
        ),

        # ── 12. FLARE ────────────────────────────────────────────────
        # Throttle idle, bleed to V_land.  Advance when wheels on ground
        # (agl < 3 ft) → ROLLOUT.
        Keyframe(
            name="FLARE", phase="FLARE",
            throttle_mode="idle",
            target_speed_kts=g.v_land,
            alt_mode="glideslope",
            heading_mode="dest_runway",
            gear_down=True, flap_ratio=1.0,
            trigger=Trigger("agl_lte", value=3.0),
        ),

        # ── 13. ROLLOUT ──────────────────────────────────────────────
        # Full brakes, wings level (tight roll limit), heading hold on
        # dest runway.  Advance when decelerated to taxi speed.
        Keyframe(
            name="ROLLOUT", phase="ROLLOUT",
            throttle_mode="idle",
            alt_mode="hold",
            heading_mode="dest_runway",
            gear_down=True, flap_ratio=1.0, brake_ratio=1.0,
            roll_limit=0.02,
            trigger=Trigger("speed_lte", value=5.0),
        ),

        # ── 14. STOP (terminal) ──────────────────────────────────────
        Keyframe(
            name="STOP", phase="ROLLOUT",
            throttle_mode="idle",
            alt_mode="hold",
            heading_mode="dest_runway",
            gear_down=True, flap_ratio=1.0, brake_ratio=1.0,
            roll_limit=0.02,
            trigger=Trigger("never"),
        ),
    ]

    # When pattern flying is active (close-in dep), insert BASE_LEG
    # between CRUISE (which now ends at BASE_TURN) and INBOUND. The
    # plane crosses perpendicular to the runway centerline at cruise
    # speed and altitude; INBOUND immediately picks up the descent
    # countdown.
    if has_base:
        # Find the CRUISE keyframe index so we can splice in after it.
        cruise_idx = next(
            (i for i, k in enumerate(keyframes) if k.name == "CRUISE"),
            -1,
        )
        if cruise_idx >= 0:
            base_leg = Keyframe(
                name="BASE_LEG", phase="CRUISE",
                throttle_mode="speed_pid",
                target_speed_kts=g.v_cruise,
                alt_mode="target", target_alt_ft=g.cruise_alt_ft,
                heading_mode="aim_at",
                aim_lat=g.join_lat, aim_lon=g.join_lon,
                gear_down=False, flap_ratio=0.0,
                # Same bank cap as CRUISE — this is the second of the
                # two ~90° pattern turns; controllers handle it the
                # same way they handle any other aim_at leg.
                roll_limit=0.35, pitch_limit=0.15,
                trigger=Trigger("near_point", value=0.5,
                               lat=g.join_lat, lon=g.join_lon),
            )
            keyframes.insert(cruise_idx + 1, base_leg)

    return keyframes


# ═══ Map preview (sparse points for app display) ═══════════════════════

def _build_preview(g: Geometry, keyframes: List[Keyframe]) -> List[PathPoint]:
    """Emit a sparse (~25 point) path for app map display."""
    pts: List[PathPoint] = []
    back_hdg = (g.rwy_heading + 180.0) % 360.0

    # Takeoff roll: 2 points along the DEPARTURE runway heading (g.dep_heading),
    # not the destination runway heading (g.rwy_heading). Previously both points
    # used g.rwy_heading, which meant a KRPD→KUBE flight (dep rwy 010°, dest
    # rwy 270°) laid the takeoff roll 1500ft WEST of the KRPD threshold instead
    # of NNE up the runway — the ribbon appeared "beside" the plane rather than
    # underneath it. The rest of the ribbon (descent/approach/flare) still uses
    # g.rwy_heading because those points are back-solved from the destination
    # runway threshold along back_hdg — that's correct.
    pts.append(PathPoint(g.dep_lat, g.dep_lon, g.dep_alt_ft, 0.0,
                         g.dep_heading, "GROUND", True, 0.5, 1.0))
    roll_lat, roll_lon = _dest_pt(g.dep_lat, g.dep_lon, g.dep_heading,
                                   1500.0 * 0.3048)
    pts.append(PathPoint(roll_lat, roll_lon, g.dep_alt_ft, g.v_rotate,
                         g.dep_heading, "GROUND", True, 0.5, 1.0))

    # Climb+cruise: targets BASE_TURN if pattern flying is active,
    # otherwise JOIN. We use either a Dubins-style true circular arc
    # (when the lift-off corner exceeds 90°, e.g. departure heading
    # is ~opposite the cruise heading) or a Bézier-smoothed corner
    # (gentler turns) before settling into a straight climb to
    # cruise_target. This is the trio: Bellman (the whole geometry
    # is backward-solved from the threshold), Bézier (small-angle
    # corners), Dubins (large-angle turns at the plane's actual
    # turn radius).
    climb_start_lat, climb_start_lon = roll_lat, roll_lon
    has_base = (
        g.base_turn_lat is not None and g.base_turn_lon is not None
    )
    cruise_target_lat = g.base_turn_lat if has_base else g.join_lat
    cruise_target_lon = g.base_turn_lon if has_base else g.join_lon

    # Heading from end-of-roll toward cruise_target. If this differs
    # from dep_heading by > 120°, a quadratic Bézier degenerates
    # (the curve loops back through the vertex). For those cases we
    # fly a true circular arc at the SF50's turn radius at climb
    # speed (~0.5 nm at 20° bank, 100 kts). Otherwise we use a Bézier
    # corner with the vertex at the climb-start point.
    target_brg = bearing_deg(climb_start_lat, climb_start_lon,
                             cruise_target_lat, cruise_target_lon)
    delta_brg = abs(((target_brg - g.dep_heading + 540.0) % 360.0) - 180.0)

    if delta_brg > 120.0:
        # ── Dubins lift-off arc ────────────────────────────────────
        arc_pts = _course_reversal_arc(
            climb_start_lat, climb_start_lon,
            g.dep_heading, target_brg,
            turn_radius_nm=0.5,
            n_steps=16,
        )
        # Interpolate altitude linearly across the arc + the straight
        # climb that follows. Total length ≈ arc + straight; we use
        # the arc length as a fraction of total.
        arc_pts_count = len(arc_pts)
        for i, (alat, alon) in enumerate(arc_pts):
            t = (i + 1) / (arc_pts_count + 8)  # 8 straight points after
            alt = _interp(g.dep_alt_ft, g.cruise_alt_ft, t)
            pts.append(PathPoint(alat, alon, alt, g.v_rotate * 1.3,
                                 target_brg, "CLIMB", False, 0.5, 0.95))
        # After the arc, plane is aligned with target_brg at some
        # point near the original takeoff. Walk a straight climb from
        # the last arc point to the cruise_target.
        last_lat, last_lon = arc_pts[-1] if arc_pts else (climb_start_lat, climb_start_lon)
        for i in range(1, 9):
            t = i / 8.0
            lat = _interp(last_lat, cruise_target_lat, t)
            lon = _interp(last_lon, cruise_target_lon, t)
            # Map back into the global progress for altitude
            global_t = (arc_pts_count + i) / (arc_pts_count + 8)
            alt = _interp(g.dep_alt_ft, g.cruise_alt_ft, global_t)
            hdg = bearing_deg(lat, lon, cruise_target_lat, cruise_target_lon)
            gear = False
            flap = 0.0
            pts.append(PathPoint(lat, lon, alt, g.v_rotate * 1.3, hdg,
                                 "CLIMB" if global_t < 0.95 else "CRUISE",
                                 gear, flap, 0.95))
    else:
        # ── Bézier lift-off corner (small angle) ───────────────────
        # Pull the curve away from the straight line slightly so the
        # plane starts its turn after climbing a bit. Vertex = a point
        # a third of the way along the straight line, biased toward
        # the takeoff direction. For small deltas this is essentially
        # the straight line; for moderate deltas it rounds the corner.
        vertex_lat = _interp(climb_start_lat, cruise_target_lat, 0.33)
        vertex_lon = _interp(climb_start_lon, cruise_target_lon, 0.33)
        bezier_pts = _bezier_corner(
            climb_start_lat, climb_start_lon,
            vertex_lat, vertex_lon,
            cruise_target_lat, cruise_target_lon,
            n_steps=12,
        )
        for i, (blat, blon) in enumerate(bezier_pts):
            t = (i + 1) / (len(bezier_pts) + 1)
            alt = _interp(g.dep_alt_ft, g.cruise_alt_ft, t)
            hdg = bearing_deg(blat, blon, cruise_target_lat, cruise_target_lon)
            gear = t < 0.05
            flap = 0.5 if t < 0.2 else 0.0
            pts.append(PathPoint(blat, blon, alt, g.v_rotate * 1.3, hdg,
                                 "CLIMB" if t < 0.95 else "CRUISE",
                                 gear, flap, 0.95))

    # When pattern flying: BASE_LEG from BASE_TURN perpendicular across
    # to JOIN, then INBOUND straight along centerline to decel_start.
    # We smooth the BASE_TURN corner (climb arrives heading toward
    # BASE_TURN, then turns to base_to_join_hdg) with a Bézier whose
    # vertex sits AT the BASE_TURN point — the path approaches it,
    # bends around it, and leaves toward the join.
    if has_base:
        base_to_join_hdg = bearing_deg(
            g.base_turn_lat, g.base_turn_lon, g.join_lat, g.join_lon
        )
        # In-point: end of the climb (cruise_target = BASE_TURN here).
        # Out-point: a fraction of the way toward JOIN.
        out_lat = _interp(g.base_turn_lat, g.join_lat, 0.5)
        out_lon = _interp(g.base_turn_lon, g.join_lon, 0.5)
        # Approach point: a fraction back from BASE_TURN along the
        # climb direction (we already arrived there, so use the
        # previous point's bearing).
        approach_t = 0.8
        in_lat = _interp(climb_start_lat, g.base_turn_lat, approach_t)
        in_lon = _interp(climb_start_lon, g.base_turn_lon, approach_t)
        base_corner = _bezier_corner(
            in_lat, in_lon,
            g.base_turn_lat, g.base_turn_lon,  # vertex = the corner point
            out_lat, out_lon,
            n_steps=10,
        )
        for blat, blon in base_corner:
            pts.append(PathPoint(blat, blon, g.cruise_alt_ft,
                                 g.v_approach * 1.6, base_to_join_hdg,
                                 "CRUISE", False, 0.0, 0.0))
        # Then straight from out-point to JOIN.
        pts.append(PathPoint(g.join_lat, g.join_lon, g.cruise_alt_ft,
                             g.v_approach * 1.6, base_to_join_hdg,
                             "CRUISE", False, 0.0, 0.0))

    # ── JOIN corner (Bézier) ──────────────────────────────────────
    # Plane is heading along base_to_join_hdg arriving at JOIN, needs
    # to turn to rwy_heading for INBOUND. Smooth this corner too.
    if has_base:
        join_in_brg = base_to_join_hdg
    else:
        # Without base-leg pattern, climb arrives directly at JOIN.
        join_in_brg = bearing_deg(climb_start_lat, climb_start_lon,
                                  g.join_lat, g.join_lon)
    delta_join = abs(((g.rwy_heading - join_in_brg + 540.0) % 360.0) - 180.0)
    if delta_join > 5.0:
        # Bézier around JOIN: approach point sits a fraction back
        # along the incoming heading; out point a fraction along the
        # outgoing centerline.
        approach_pt = _dest_pt(g.join_lat, g.join_lon,
                               (join_in_brg + 180.0) % 360.0,
                               0.4 * 1852.0)  # 0.4 nm back
        out_pt = _dest_pt(g.join_lat, g.join_lon,
                          g.rwy_heading, 0.4 * 1852.0)  # 0.4 nm forward
        join_corner = _bezier_corner(
            approach_pt[0], approach_pt[1],
            g.join_lat, g.join_lon,
            out_pt[0], out_pt[1],
            n_steps=10,
        )
        for blat, blon in join_corner:
            pts.append(PathPoint(blat, blon, g.cruise_alt_ft,
                                 g.v_approach * 1.6, g.rwy_heading,
                                 "CRUISE", False, 0.0, 0.0))

    # Inbound: straight leg from join_point → decel_start along centerline
    pts.append(PathPoint(g.decel_start_lat, g.decel_start_lon, g.cruise_alt_ft,
                         g.v_approach * 1.6, g.rwy_heading, "CRUISE",
                         False, 0.0, 0.0))

    # Descent leg: 16 points from descent_start → approach_start.
    # Dense sampling (was 4) for future obstacle-avoidance work — the
    # finer the polyline, the smaller the segment a re-router can
    # tweak around a terrain bump. Altitude stays linearly interpolated
    # (constant flight-path angle); only the sample count changed.
    for i in range(1, 17):
        t = i / 16.0
        lat = _interp(g.descent_start_lat, g.approach_start_lat, t)
        lon = _interp(g.descent_start_lon, g.approach_start_lon, t)
        alt = _interp(g.cruise_alt_ft, g.approach_start_alt_ft, t)
        flap = 0.5 if t > 0.4 else 0.0
        gear = t > 0.8
        pts.append(PathPoint(lat, lon, alt, g.v_approach * 1.2, g.rwy_heading,
                             "DESCENT", gear, flap, 0.40))

    # Approach: 10 points from approach_start → flare_start (was 3).
    for i in range(1, 11):
        t = i / 10.0
        lat = _interp(g.approach_start_lat, g.flare_start_lat, t)
        lon = _interp(g.approach_start_lon, g.flare_start_lon, t)
        alt = _interp(g.approach_start_alt_ft, g.thr_alt_ft + _FLARE_ALT_AGL_FT, t)
        pts.append(PathPoint(lat, lon, alt, g.v_approach, g.rwy_heading,
                             "APPROACH", True, 1.0, None))

    # Flare: 1 point at touchdown
    pts.append(PathPoint(g.touchdown_lat, g.touchdown_lon,
                         g.thr_alt_ft + 2.0, g.v_land, g.rwy_heading,
                         "FLARE", True, 1.0, 0.0))

    # Rollout: 1 point 2000 ft past touchdown
    ro_lat, ro_lon = _dest_pt(g.touchdown_lat, g.touchdown_lon,
                               g.rwy_heading, 2000.0 * 0.3048)
    pts.append(PathPoint(ro_lat, ro_lon, g.thr_alt_ft, 0.0,
                         g.rwy_heading, "ROLLOUT", True, 1.0, 0.0))

    # Cumulative distances
    pts[0].dist_from_start_nm = 0.0
    for i in range(1, len(pts)):
        d = haversine_m(pts[i-1].lat, pts[i-1].lon, pts[i].lat, pts[i].lon) / 1852.0
        pts[i].dist_from_start_nm = pts[i-1].dist_from_start_nm + d

    return pts


# ═══ Public entry point ════════════════════════════════════════════════

def plan_path(
    *,
    dep_lat: float,
    dep_lon: float,
    dep_alt_ft: float,
    dep_heading: float,
    dest_lat: float,
    dest_lon: float,
    dest_alt_ft: float,
    dest_rwy_heading: Optional[float] = None,
    dest_threshold_lat: Optional[float] = None,
    dest_threshold_lon: Optional[float] = None,
    dest_rwy_length_ft: float = 6000.0,  # unused, kept for API compat
    cruise_alt_ft: float,
    v_stall: float = _VS_DEFAULT,
    # Legacy kwargs — accepted and ignored for API compat
    **_legacy,
) -> Ribbon:
    """Build a keyframe ribbon from `dep` to `dest`.

    Only one speed knob: `v_stall`.  All other speeds (rotate/approach/
    land, gear/flap-safe) are fixed ratios of Vs.  Cruise throttle is
    altitude-scaled.  No v_rotate/v_climb/v_cruise/v_approach/v_land/
    climb_fpm/takeoff_roll_ft configuration — ribbon decides everything.
    """
    _ = dest_rwy_length_ft  # unused
    geometry = _solve_geometry(
        dep_lat=dep_lat, dep_lon=dep_lon,
        dep_alt_ft=dep_alt_ft, dep_heading=dep_heading,
        dest_lat=dest_lat, dest_lon=dest_lon,
        dest_alt_ft=dest_alt_ft,
        dest_rwy_heading=dest_rwy_heading,
        dest_threshold_lat=dest_threshold_lat,
        dest_threshold_lon=dest_threshold_lon,
        cruise_alt_ft=cruise_alt_ft,
        v_stall=v_stall,
    )
    keyframes = _build_keyframes(geometry)
    preview = _build_preview(geometry, keyframes)
    return Ribbon(keyframes=keyframes, geometry=geometry, points=preview)


# ═══ Pretty-print ══════════════════════════════════════════════════════

def format_ribbon(ribbon: Ribbon) -> str:
    g = ribbon.geometry
    lines = [
        f"[RIBBON] {len(ribbon.keyframes)} keyframes · Vs={g.v_stall:.0f}kts",
        f"  Speeds:  Vr={g.v_rotate:.0f}  Vcru={g.v_cruise:.0f}  "
        f"Vapp={g.v_approach:.0f}  Vland={g.v_land:.0f}  "
        f"gear≤{g.gear_safe_kts:.0f}  flap≤{g.flap_safe_kts:.0f}",
        f"  Cruise:  {g.cruise_alt_ft:.0f}ft MSL  "
        f"Rwy hdg {g.rwy_heading:.0f}°  Thr alt {g.thr_alt_ft:.0f}ft",
    ]
    for i, kf in enumerate(ribbon.keyframes, 1):
        # Describe command
        cmd_parts = []
        if kf.throttle_mode == "explicit" and kf.throttle is not None:
            cmd_parts.append(f"thr={kf.throttle:.2f}")
        elif kf.throttle_mode == "alt_scaled":
            cmd_parts.append("thr=alt_scaled")
        elif kf.throttle_mode == "idle":
            cmd_parts.append("thr=idle")
        elif kf.throttle_mode == "speed_pid":
            cmd_parts.append("thr=PID")
        if kf.target_speed_kts is not None:
            cmd_parts.append(f"spd={kf.target_speed_kts:.0f}")
        if kf.alt_mode == "target" and kf.target_alt_ft is not None:
            cmd_parts.append(f"alt={kf.target_alt_ft:.0f}")
        elif kf.alt_mode == "glideslope":
            cmd_parts.append("alt=glide")
        elif kf.alt_mode == "hold":
            cmd_parts.append("alt=hold")
        if kf.heading_mode == "dep_runway":
            cmd_parts.append("hdg=dep-rwy")
        elif kf.heading_mode == "dest_runway":
            cmd_parts.append("hdg=dest-rwy")
        elif kf.heading_mode == "aim_at":
            cmd_parts.append(f"hdg→({kf.aim_lat:.3f},{kf.aim_lon:.3f})")
        elif kf.heading_mode == "fixed" and kf.target_heading_deg is not None:
            cmd_parts.append(f"hdg={kf.target_heading_deg:.0f}")
        if kf.gear_down is not None:
            cmd_parts.append("gear↓" if kf.gear_down else "gear↑")
        if kf.flap_ratio is not None:
            cmd_parts.append(f"flap={kf.flap_ratio:.1f}")
        if kf.brake_ratio is not None and kf.brake_ratio > 0:
            cmd_parts.append(f"brake={kf.brake_ratio:.1f}")
        cmd = " ".join(cmd_parts)

        lines.append(
            f"  {i:2d}. {kf.name:<14} [{kf.phase:>8}]  {cmd}"
        )
        lines.append(
            f"        advance: {kf.trigger.describe()}"
        )
    return "\n".join(lines)
