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
_RATIO_FLAP_SAFE = 1.35   # was 1.50, which collided exactly with
# _RATIO_V_CRUISE (also 1.5): flap_safe == v_cruise meant DECELERATE's
# speed_lte trigger fired instantly (no decel leg at all) and flaps were
# scheduled at cruise speed. 1.35 × 98.4 ≈ 133 kts restores a ~15 kt
# bleed segment and keeps flap deployment below the real Vfe.
_RATIO_V_CRUISE = 1.5        # cruise target speed — was 2.0 which produced
# v_cruise = 196 kts (when v_stall = 98), miles above the SF50's actual
# observed cruise of ~133 kts. The mismatch drove the throttle PID to
# 100% saturation during cruise/transition, which the alt PID couldn't
# counteract via its capped pitch-down, so the plane climbed thousands
# of feet over target. 1.5 × v_stall ≈ 147 kts — comfortably within
# the airframe's clean-config range and close to the envelope's
# observed value.

_GLIDE_FT_PER_NM = 500.0    # ~4.7° glideslope. The 1000 ft/nm this replaced
# was tuned for the SF50 jet ("can dive cleanly with gear+full flaps"), not
# the King Air we actually fly now — physically unachievable for it at idle.
# Measured in the calibrated offline sim (tests/sim_longitudinal.py's real
# C90B mass/wing/thrust model): holding ANY sane nose attitude at idle power,
# the King Air settles into 450-700 ft/nm depending on speed (453 ft/nm level
# at 133 kt, up to 701 ft/nm at -6° nose / 195 kt). Commanding 1000 meant the
# plane could never sink fast enough, floated 300-500 ft above the line for
# the whole steep segment, then had to dive hard to catch up near the ground
# and overshot through the line into a short landing (flight 2dc2cef9 —
# Idriss: "we need to be right on the money in terms of alt"). 500 sits
# comfortably inside the measured achievable range with margin.
_FINAL_FT_PER_NM = 450.0    # final-approach slope (~4.2°), APPROACH keyframe
# only. The 1000 ft/nm descent slope is fine up high but produces ~1700 fpm
# of sink at flare height — no flare law can arrest that in the ~1 s
# available from 30 ft. The last 600 ft AGL fly a conventional stabilized
# final instead so the flare starts from ~600 fpm.
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
_JOIN_MIN_OFFSET_NM = 3.0    # cap: join ≤ 3 nm behind decel_start (long trips)
_JOIN_MIN_OFFSET_FLOOR_NM = 1.0  # floor for tiny loop flights
_JOIN_MAX_OFFSET_NM = 80.0   # cap so we don't fly wildly off course

# When the single-leg geometric join would force a turn ≥ this many
# degrees at the JOIN point, the planner switches from a single-leg
# (CRUISE → INBOUND) pattern to a two-leg pattern (CRUISE → BASE_LEG
# → INBOUND), inserting a BASE_TURN waypoint perpendicular to the
# centerline on the dep's side.
#
# History: this was 60° when JOIN was a sharp polyline elbow the
# controllers had to absorb mid-flight — above 60° they saturated
# roll and pulled 2g+ (flights 20260419_135940, 20260419_145024).
# Since the Dubins rework the corner is a fillet arc at the plane's
# real turn radius, flown at the same fixed bank as any other turn,
# so a 90° join is as flyable as a 30° one. 100° keeps the pattern
# only for arrivals from beyond-perpendicular (approaching from
# behind the runway axis), where a base leg genuinely reads better
# than a near-U-turn fillet. Kills the unnecessary dogleg on
# moderately off-axis departures (seen on Amery→KUBE, ~61° join).
_MAX_SINGLE_LEG_TURN_DEG = 100.0
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
    # Arc-length of this trigger's fix on the ribbon polyline (nm).
    # Set by plan_path after the preview is built. The PASS-BY branch
    # of near_point may only fire once the follower's along-track
    # progress has reached the fix: on a loop's outbound leg every
    # arrival fix sits BEHIND the plane, and "behind" is geometrically
    # identical to "passed" — the whole pattern cascaded in one tick
    # (flight at KUBE, 2026-07-08). Direction can lie; distance along
    # the ribbon can't. None = no gate (pre-gate ribbons).
    along_gate_nm: Optional[float] = None

    def fired(self, telemetry, agl_ft: float,
              along_nm: Optional[float] = None) -> bool:
        k = self.kind
        if k == "never":
            return False
        if k == "always":
            return True
        if k == "any":
            return any(s.fired(telemetry, agl_ft, along_nm) for s in self.subs)
        if k == "all":
            return all(s.fired(telemetry, agl_ft, along_nm) for s in self.subs)
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
            # Along-track gate — applies to PROXIMITY too, not just the
            # pass-by: a loop's outbound leg physically overlaps the
            # arrival corridor, so the plane OVERFLIES arrival fixes on
            # its way out. near_point means "arrived at this fix along
            # the route," never "happened to fly over it."
            if (self.along_gate_nm is not None and along_nm is not None
                    and along_nm < self.along_gate_nm):
                return False
            d_nm = haversine_m(
                telemetry.lat_deg, telemetry.lon_deg, self.lat, self.lon,
            ) / 1852.0
            if d_nm <= self.value:
                return True
            # Also fire if we've crossed PAST the point. The L1 ribbon
            # follower steers the polyline, not the raw aim point — it can
            # sail wide of cruise_aim by more than the radius, leaving the
            # trigger forever unfulfilled (flight 20260522_085103 flew 16 nm
            # past cruise_aim in CRUISE before being killed). "Past" =
            # bearing-to-point is >90° off plane heading. Capped at 5×radius
            # so a stray heading swing during cross-track correction at long
            # range can't false-fire. When progress is UNKNOWN the pass-by
            # is denied outright for gated triggers (conservative).
            if self.along_gate_nm is not None and along_nm is None:
                return False
            if (not math.isnan(telemetry.heading_deg)
                    and d_nm < max(5.0, self.value * 5.0)):
                brg = bearing_deg(
                    telemetry.lat_deg, telemetry.lon_deg, self.lat, self.lon,
                )
                ang = abs(((brg - telemetry.heading_deg + 180.0) % 360.0) - 180.0)
                if ang > 90.0:
                    return True
            return False
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

    # Throttle-for-altitude coupling. When True, the controller swaps:
    # throttle ← alt PID (power for altitude), pitch ← speed PID
    # (attitude for airspeed). Use on phases where alt is the priority
    # and we don't want the speed PID pushing throttle past target alt.
    throttle_for_alt: bool = False

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


def _turn_radius_nm(v_kts: float, bank_deg: float = 31.0) -> float:
    """Minimum flyable turn radius at `v_kts` and the given bank angle.

    R = V² / (g·tan(φ)). At 133 kts / 31° bank ≈ 0.43 nm. Every curve
    the ribbon draws must be at least this gentle, otherwise the plane
    physically cannot stay on the line and the follower reports phantom
    cross-track error. 31° matches the CRUISE keyframe's roll_limit
    (0.35 × 90°).
    """
    v_ms = max(30.0, v_kts) * 0.514444
    r_m = (v_ms * v_ms) / (9.81 * math.tan(math.radians(bank_deg)))
    return r_m / 1852.0


def _tangent_arc(
    start_lat: float, start_lon: float, start_hdg: float,
    tgt_lat: float, tgt_lon: float,
    radius_nm: float,
    step_deg: float = 10.0,
    max_sweep_deg: float = 300.0,
) -> tuple[list, float]:
    """Departure arc: turn at fixed radius from `start_hdg` until the
    arc's tangent points at the target, then stop. This is the user's
    "half circle until it faces the destination" — a Dubins departure
    arc whose sweep is whatever the geometry needs (0° for a straight-
    out departure, ~180° for a course reversal).

    Returns (arc_points, exit_heading). Points exclude the start point.
    Sweep is capped so a target sitting inside the turn circle (rare:
    destination < 2R away and behind) can't produce an endless loop.
    """
    brg0 = bearing_deg(start_lat, start_lon, tgt_lat, tgt_lon)
    delta0 = ((brg0 - start_hdg + 540.0) % 360.0) - 180.0
    if abs(delta0) < 8.0:
        return [], start_hdg  # already facing the target
    sign = 1.0 if delta0 > 0 else -1.0
    centre_lat, centre_lon = _dest_pt(
        start_lat, start_lon, (start_hdg + 90.0 * sign) % 360.0,
        radius_nm * 1852.0,
    )
    ang_to_start = (start_hdg + 90.0 * sign + 180.0) % 360.0
    pts: list = []
    sweep = 0.0
    hdg_here = start_hdg
    while sweep < max_sweep_deg:
        sweep += step_deg
        ang = (ang_to_start + sign * sweep) % 360.0
        p = _dest_pt(centre_lat, centre_lon, ang, radius_nm * 1852.0)
        hdg_here = (ang + 90.0 * sign) % 360.0
        pts.append(p)
        brg_tgt = bearing_deg(p[0], p[1], tgt_lat, tgt_lon)
        if abs(((brg_tgt - hdg_here + 540.0) % 360.0) - 180.0) <= step_deg * 0.75:
            break
    return pts, hdg_here


def _fillet_arc(
    prev_lat: float, prev_lon: float,
    corner_lat: float, corner_lon: float,
    next_lat: float, next_lon: float,
    radius_nm: float,
    step_deg: float = 10.0,
) -> list:
    """Replace a polyline corner with a constant-radius arc tangent to
    both legs. Returns the arc points (entry → exit); empty list when
    the corner is nearly straight. The corner point itself is NOT on the
    returned path — the arc cuts inside it, which is exactly what a
    plane flying through the corner at fixed bank does.

    If either leg is too short for the ideal radius, the radius shrinks
    to fit (the follower's roll limit then rounds the residual — better
    than the arc overrunning into the next leg).
    """
    hdg_in = bearing_deg(prev_lat, prev_lon, corner_lat, corner_lon)
    hdg_out = bearing_deg(corner_lat, corner_lon, next_lat, next_lon)
    delta = ((hdg_out - hdg_in + 540.0) % 360.0) - 180.0
    if abs(delta) < 5.0:
        return []
    sign = 1.0 if delta > 0 else -1.0

    len_in = haversine_m(prev_lat, prev_lon, corner_lat, corner_lon) / 1852.0
    len_out = haversine_m(corner_lat, corner_lon, next_lat, next_lon) / 1852.0
    # Tangent offset from the corner along each leg: t = R·tan(Δ/2).
    half = math.radians(abs(delta)) / 2.0
    t_ideal = radius_nm * math.tan(half)
    t_max = 0.45 * min(len_in, len_out)
    r_eff = radius_nm if t_ideal <= t_max else t_max / math.tan(half)
    t_off = r_eff * math.tan(half)

    entry_lat, entry_lon = _dest_pt(
        corner_lat, corner_lon, (hdg_in + 180.0) % 360.0, t_off * 1852.0,
    )
    centre_lat, centre_lon = _dest_pt(
        entry_lat, entry_lon, (hdg_in + 90.0 * sign) % 360.0, r_eff * 1852.0,
    )
    ang_start = (hdg_in + 90.0 * sign + 180.0) % 360.0
    n_steps = max(2, int(abs(delta) / step_deg))
    pts = []
    for i in range(n_steps + 1):
        ang = (ang_start + sign * abs(delta) * (i / n_steps)) % 360.0
        pts.append(_dest_pt(centre_lat, centre_lon, ang, r_eff * 1852.0))
    return pts


def pick_cruise_alt_agl(dist_nm: float) -> float:
    """Trip-length-optimal cruise altitude (AGL).

    Altitude buys speed (TAS grows ~1.5% per 1000 ft at fixed IAS), but
    the climb costs time (~0.005 s per ft at the SF50's measured 30 ft/s
    climb with ~15% forward-speed deficit). Marginal break-even sits
    near 12 nm: below it the climb never pays for itself, above it every
    extra foot is profit until a cap binds. Caps: 5000 AGL (config
    ceiling for these short hops) and the descent+climb footprint must
    fit inside the route.

    clamp(350 × (D − 8), 1500, 5000): ≤ ~12 nm stays at 1500 AGL,
    ~22 nm reaches 5000, smooth ramp between.
    """
    return max(1500.0, min(5000.0, 350.0 * (dist_nm - 8.0)))


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
    # Join run-up scales with the trip: a 30 nm flight earns the full
    # 3 nm alignment leg; a same-field loop only pays ~1 nm, which cuts
    # the loop's total footprint (and test-cycle time) roughly in half.
    dep_dist_nm = math.hypot(dep_e, dep_n)
    join_offset = max(_JOIN_MIN_OFFSET_FLOOR_NM,
                      min(_JOIN_MIN_OFFSET_NM, 0.15 * dep_dist_nm))
    t_min = t_decel + join_offset
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
    v_cruise_kts: Optional[float] = None,
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
    approach_len_nm = max(0.5, approach_descent_ft / _FINAL_FT_PER_NM)
    app_lat, app_lon = _dest_pt(flare_lat, flare_lon, back_hdg, approach_len_nm * 1852.0)

    # Descent start: glideslope from cruise_alt down to approach_alt
    alt_to_descend = max(0.0, cruise_alt_ft - approach_alt)
    descent_len_nm = alt_to_descend / _GLIDE_FT_PER_NM if alt_to_descend > 0 else 0.0
    desc_lat, desc_lon = _dest_pt(app_lat, app_lon, back_hdg, descent_len_nm * 1852.0)

    # Speeds (computed early so decel-leg sizing can use them)
    v_rot = _RATIO_V_ROTATE * v_stall
    # Prefer the learned envelope's cruise speed over the ratio guess —
    # the SF50's measured cruise is ~133 kts while 1.5 × v_land gives
    # 147.6. Planning to a speed the jet doesn't actually fly means the
    # speed PID saturates chasing it. Ratio remains the fallback when no
    # envelope is available.
    v_cru = (float(v_cruise_kts) if v_cruise_kts and v_cruise_kts > 0
             else _RATIO_V_CRUISE * v_stall)
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
            # yaw_kp 0.06 / limit 0.6 (was 0.02/0.35): the commanded
            # heading correction is only as strong as the rudder that
            # executes it — 2% pedal was losing to P-factor at full
            # power. The -0.008·hdg_rate damping in the yaw law keeps
            # the stronger gain from oscillating.
            yaw_hold=True, yaw_kp=0.06, yaw_limit=0.60,
            trigger=Trigger("speed_gte", value=g.v_rotate),
        ),

        # ── 2. CLIMB_ROTATE ──────────────────────────────────────────
        # Rotation + initial climb.  Hold departure runway heading, climb
        # to cruise alt.  Gear still down, flaps still 0.5.  Pitch limit
        # 0.20 (≈12°) so alt-PID can't command 90° nose-up and stall us.
        # Advance when clear of obstacles — agl ≥ 50 ft.
        Keyframe(
            name="CLIMB_ROTATE", phase="CLIMB",
            # Closed-loop climb power (user doctrine: nothing pinned). The
            # walk seeds from takeoff's full power and eases on its own
            # as speed passes climb speed; pitch holds the climb RATE.
            throttle_mode="speed_pid",
            target_speed_kts=g.v_rotate * 1.1,
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
            # Closed-loop climb power (user doctrine: nothing pinned). The
            # walk seeds from takeoff's full power and eases on its own
            # as speed passes climb speed; pitch holds the climb RATE.
            throttle_mode="speed_pid",
            target_speed_kts=g.v_rotate * 1.1,
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="dep_runway",
            gear_down=False, flap_ratio=0.5,
            pitch_limit=0.20,
            trigger=Trigger("agl_gte", value=300.0),
        ),

        # ── 4. CLIMB_CLEAN ───────────────────────────────────────────
        # Flaps retracted, turn toward the join point (where we meet the
        # extended runway centerline at a 30° intercept).  Pitch still
        # capped.  roll_limit WIDENED 0.25->0.667 (~15°->60° bank), Idriss,
        # 2026-07-11: explicit, informed override — the 0.25 cap traces to
        # flight 20260419_145024 (tight aim_at turn at saturation drove R to
        # -1.0 at ~500 AGL, hit terrain). Owner's call: the plane needs
        # authority to roll itself OUT of a big unwanted bank, and a low
        # roll_limit was judged too restrictive for that. Flagged, heard,
        # deliberate; owner will revert if this next flight shows it's wrong.
        # Advance when we reach cruise altitude — next keyframe (TRANSITION)
        # handles speed accel at cruise alt.
        Keyframe(
            name="CLIMB_CLEAN", phase="CLIMB",
            # Closed-loop climb power (user doctrine: nothing pinned). The
            # walk seeds from takeoff's full power and eases on its own
            # as speed passes climb speed; pitch holds the climb RATE.
            throttle_mode="speed_pid",
            target_speed_kts=g.v_rotate * 1.1,
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="aim_at",
            aim_lat=cruise_aim_lat, aim_lon=cruise_aim_lon,
            gear_down=False, flap_ratio=0.0,
            roll_limit=0.667,
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
            throttle_for_alt=True,  # alt priority — engine defends target alt
            target_speed_kts=g.v_cruise,
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="aim_at",
            aim_lat=cruise_aim_lat, aim_lon=cruise_aim_lon,
            gear_down=False, flap_ratio=0.0,
            # roll_limit widened 0.25->0.667 (~15°->60° bank), Idriss,
            # 2026-07-11: same explicit override as CLIMB_CLEAN — recovery
            # authority from a big unwanted bank matters more here than the
            # shallow-turn caution. No documented crash tied to this
            # specific value (unlike CLIMB_CLEAN/CRUISE below).
            roll_limit=0.667,
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
            # Advance on ALTITUDE CAPTURED, not speed. Cruise is now
            # emergent-speed (throttle defends altitude, speed floats),
            # so the plane never reaches v_cruise in level flight and a
            # speed_gte trigger would trap it in TRANSITION forever,
            # circling, never descending. alt_reached fires within 50 ft
            # of cruise altitude — the real definition of "levelled off."
            # Hands to CRUISE, which flies the same hold to the join.
            trigger=Trigger("alt_reached", value=g.cruise_alt_ft),
        ),

        # ── 5. CRUISE ────────────────────────────────────────────────
        # Speed-PID throttle holds Vcruise; alt-hold holds cruise altitude.
        # Both metrics tracked — no more free-drifting speed or altitude.
        # Aim at the join point (30° intercept with runway axis).
        Keyframe(
            name="CRUISE", phase="CRUISE",
            throttle_mode="speed_pid",
            throttle_for_alt=True,  # alt priority
            target_speed_kts=g.v_cruise,
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="aim_at",
            aim_lat=cruise_aim_lat, aim_lon=cruise_aim_lon,
            gear_down=False, flap_ratio=0.0,
            # roll_limit widened 0.35->0.667 (~31°->60° bank), Idriss,
            # 2026-07-11: EXPLICIT override of the 135940 incident this cap
            # was built to prevent (60° bank at cruise speed caused 2g) —
            # owner has heard this history and wants recovery authority
            # from a big unwanted bank prioritized over that margin. Flagged
            # directly before making this change; owner will revert if the
            # next flight shows it's wrong. Pitch ≤ ~14° nose-up cap.
            roll_limit=0.667, pitch_limit=0.15,
            trigger=Trigger("near_point", value=1.0,
                           lat=cruise_aim_lat, lon=cruise_aim_lon),
        ),

        # ── 6. INBOUND ───────────────────────────────────────────────
        # On the extended centerline.  Still Vcruise + cruise alt; this
        # is the gentle roll-out of the intercept turn toward decel_start.
        Keyframe(
            name="INBOUND", phase="CRUISE",
            throttle_mode="speed_pid",
            throttle_for_alt=True,
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
            # HOLD ALTITUDE, do not idle-sink. `idle` was for the fast
            # SF50 that had to dump a lot of speed before the descent;
            # the King Air (crash 374c35b5) can't hold altitude at idle
            # and sank 1500 ft into terrain 10 nm out, before ever
            # reaching the descent point. Emergent-speed cruise arrives
            # here already near flap_safe, so no aggressive bleed is
            # needed: throttle_mode "speed_pid" trips cruise_hold (pitch
            # holds altitude, throttle reactive), the plane flies level
            # to descent_start, and the DESCENT phase owns the descent.
            throttle_mode="speed_pid",
            target_speed_kts=min(g.flap_safe_kts, g.v_cruise),
            alt_mode="target", target_alt_ft=g.cruise_alt_ft,
            heading_mode="aim_at",
            aim_lat=g.descent_start_lat, aim_lon=g.descent_start_lon,
            gear_down=False, flap_ratio=0.0,
            roll_limit=0.25, pitch_limit=0.15,
            # POSITION-ONLY release: the descent begins where the
            # geometry says it begins. The old speed_lte condition
            # became degenerate once learned v_cruise (132.7) dropped
            # below flap_safe (132.8) — it fired the instant DECELERATE
            # began, starting DESCENT a mile early on the clamped-level
            # segment, where the plane seesawed 230 ft down / 140 up
            # burning 86% throttle (flight 914edfc6). DECELERATE's job
            # is bleeding speed BEFORE descent_start; if speed is
            # already fine, this leg is simply level cruise until the
            # descent point arrives. If speed is still high there, the
            # bleed/flap/stall protections own it on the way down.
            trigger=Trigger("near_point", value=0.3,
                            lat=g.descent_start_lat,
                            lon=g.descent_start_lon),
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
            throttle_for_alt=True,  # throttle defends the glideslope alt
            # SOFT reference only — on the glideslope the controller flies
            # altitude-priority (throttle→idle, speed emergent). Capped to a
            # sane descent speed: flap_safe is a stall RATIO off v_land (108)
            # and ballooned to ~146-162 when the real stall speeds went in,
            # which the old speed-holding law chased at 0.9-1.0 power — the
            # plane sat at 2600 ft and never descended (flight c269d01f).
            target_speed_kts=min(g.flap_safe_kts, 130.0),
            alt_mode="glideslope",
            heading_mode="aim_at",
            aim_lat=g.approach_start_lat, aim_lon=g.approach_start_lon,
            pitch_limit=0.25,  # VS-tracking pitch needs nose-up room
            # pitch_down_limit 0.40 (~23° stick) so plane can actually
            # descend when behind the glideslope. Previous 0.15 (~8.6°)
            # was too restrictive: flight 20260419_173924 arrived at
            # threshold 1700ft HIGH because pitch was pegged at -0.15
            # but only produced +3° nose-up (thrust + trim overpowering
            # the gentle stick). Speed PID now (kp=0.05) will slam
            # throttle to idle on any overspeed, so PE→KE conversion
            # during a steeper dive won't shred the flap envelope.
            # Full flaps SCHEDULED from the top of the descent —
            # configure early, take the disturbance high. The engine's
            # speed staging decides how much actually deploys (half
            # above v_app+10, full below), so the schedule is a cap,
            # not a command to slam them at cruise speed.
            gear_down=True, flap_ratio=1.0,
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
            throttle_for_alt=True,
            # Soft reference (altitude-priority descent; see DESCENT above).
            target_speed_kts=min(g.gear_safe_kts, 120.0),
            alt_mode="glideslope",
            heading_mode="aim_at",
            aim_lat=g.approach_start_lat, aim_lon=g.approach_start_lon,
            pitch_limit=0.25,
            gear_down=True, flap_ratio=1.0,
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
            throttle_for_alt=True,
            target_speed_kts=g.v_approach,  # soft ref (~120); alt-priority descent
            alt_mode="glideslope",
            heading_mode="aim_at",
            aim_lat=g.flare_start_lat, aim_lon=g.flare_start_lon,
            pitch_limit=0.25,
            gear_down=True, flap_ratio=1.0,
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
            throttle_for_alt=True,  # power for altitude on final approach
            # Fly SHORT FINAL at the landing speed, not v_approach. Crossing
            # the threshold at v_approach (120, held ~116) carried too much
            # energy into the flare — the roundout ballooned it long and it
            # bounced (Idriss report 2026-07-11, flight 143847). v_land (108,
            # still 1.2× the 90-kt flap stall) crosses slow so the flare
            # settles on the aim point. Deceleration from the descent speed
            # happens over the final, before the flare.
            target_speed_kts=g.v_land,
            alt_mode="glideslope",
            # Track the RUNWAY CENTERLINE directly on final, not the L1
            # ribbon follower. The follower left a lateral offset the weak
            # flare-only correction couldn't fix in time (landed in the
            # grass, then the ground steering hauled it onto the pavement).
            # dest_runway drives the meter-scale cross-track correction
            # through the whole approach, so the plane is ON the centerline
            # before the wheels come down.
            heading_mode="dest_runway",
            aim_lat=g.touchdown_lat, aim_lon=g.touchdown_lon,
            pitch_limit=0.25,
            # APPROACH stays tighter than outer DESCENT keyframes (0.20)
            # because we're short final: a big dive here would smash the
            # gear. Still relaxed from 0.15 so we can catch a
            # behind-schedule glideslope without the previous failure mode.
            gear_down=True, flap_ratio=1.0,
            trigger=Trigger("agl_lte", value=_FLARE_ALT_AGL_FT),
        ),

        # ── 12. FLARE ────────────────────────────────────────────────
        # Throttle idle, bleed to V_land.  Pitch is driven by a SINK-RATE
        # target (engine emits vs_target_fpm scaled by AGL, controller
        # tracks it) — not by the glideslope alt PID, which used to
        # command nose-DOWN at 30 ft because the clamped target alt sat
        # below the plane. pitch_down_limit 0.05 means the nose can
        # barely drop below neutral this close to the ground; nose-up
        # cap 0.25 keeps the flare from ballooning. Advance when wheels
        # on ground (agl < 3 ft) → ROLLOUT.
        Keyframe(
            name="FLARE", phase="FLARE",
            throttle_mode="idle",
            target_speed_kts=g.v_land,
            alt_mode="glideslope",
            heading_mode="dest_runway",
            pitch_limit=0.25, pitch_down_limit=0.05,
            gear_down=True, flap_ratio=1.0,
            trigger=Trigger("agl_lte", value=3.0),
        ),

        # ── 13. ROLLOUT ──────────────────────────────────────────────
        # Full brakes, wings level (tight roll limit), heading hold on
        # dest runway. yaw_hold gives RUDDER ground steering — without
        # it the only heading authority on the ground was roll, which
        # does nothing on wheels: flight 914edfc6 veered 84°→101° during
        # rollout and departed the runway. Advance when decelerated to
        # taxi speed.
        Keyframe(
            name="ROLLOUT", phase="ROLLOUT",
            throttle_mode="idle",
            alt_mode="hold",
            heading_mode="dest_runway",
            gear_down=True, flap_ratio=1.0, brake_ratio=1.0,
            roll_limit=0.02,
            # Nose-up capped at ~zero on the ground (tail-strike
            # protection); the engine's derotation VS demand supplies
            # the gentle forward stick.
            pitch_limit=0.02,
            yaw_hold=True, yaw_kp=0.06, yaw_limit=0.60,
            trigger=Trigger("speed_lte", value=5.0),
        ),

        # ── 14. STOP (terminal) ──────────────────────────────────────
        Keyframe(
            name="STOP", phase="ROLLOUT",
            throttle_mode="idle",
            alt_mode="hold",
            heading_mode="dest_runway",
            gear_down=True, flap_ratio=1.0, brake_ratio=1.0,
            roll_limit=0.02, pitch_limit=0.02,
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
                # The ONE cruise keyframe that never got the coupling:
                # flight 96786124 climbed +1000 ft at full throttle on
                # base (classic speed-PID slamming power for an 11-kt
                # deficit while the wound alt PID pitched up), then
                # dove to 179 kts at the join.
                throttle_for_alt=True,
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

    # ── Climb + cruise: Dubins-style geometry ────────────────────────
    # The horizontal route is assembled from pieces the plane can
    # physically fly at its bank limit:
    #   1. straight climb-out on the runway heading (1 nm)
    #   2. DEPARTURE ARC at the min turn radius, sweeping until the
    #      tangent points at the first fix (BASE_TURN or JOIN)
    #   3. straight legs between fixes
    #   4. FILLET ARCS replacing the corners at BASE_TURN and JOIN
    # No corner in the resulting polyline exceeds what R = V²/(g·tanφ)
    # allows, so the follower can hold the line instead of cutting it.
    climb_start_lat, climb_start_lon = roll_lat, roll_lon
    has_base = (
        g.base_turn_lat is not None and g.base_turn_lon is not None
    )
    cruise_target_lat = g.base_turn_lat if has_base else g.join_lat
    cruise_target_lon = g.base_turn_lon if has_base else g.join_lon

    turn_r_nm = _turn_radius_nm(g.v_cruise)

    # Straight initial climb-out: 1 nm dead ahead (liftoff, gear, and
    # no banking below ~500 AGL).
    arc_anchor_lat, arc_anchor_lon = _dest_pt(
        climb_start_lat, climb_start_lon, g.dep_heading, 1.0 * 1852.0,
    )

    # Departure arc → tangent at cruise_target.
    dep_arc, _exit_hdg = _tangent_arc(
        arc_anchor_lat, arc_anchor_lon, g.dep_heading,
        cruise_target_lat, cruise_target_lon, turn_r_nm,
    )

    # Horizontal skeleton: (lat, lon) list from climb start to
    # decel_start, arcs included, corners filleted.
    route: list[tuple[float, float]] = [(climb_start_lat, climb_start_lon),
                                        (arc_anchor_lat, arc_anchor_lon)]
    route.extend(dep_arc)

    if has_base:
        # Fillet at BASE_TURN (between arc exit and JOIN) and at JOIN
        # (between BASE_TURN and decel_start). The fillets replace the
        # corner points — the plane never visits the sharp vertex.
        prev_lat, prev_lon = route[-1]
        base_fillet = _fillet_arc(prev_lat, prev_lon,
                                  g.base_turn_lat, g.base_turn_lon,
                                  g.join_lat, g.join_lon, turn_r_nm)
        route.extend(base_fillet if base_fillet
                     else [(g.base_turn_lat, g.base_turn_lon)])
        join_fillet = _fillet_arc(g.base_turn_lat, g.base_turn_lon,
                                  g.join_lat, g.join_lon,
                                  g.decel_start_lat, g.decel_start_lon,
                                  turn_r_nm)
        route.extend(join_fillet if join_fillet
                     else [(g.join_lat, g.join_lon)])
    else:
        prev_lat, prev_lon = route[-1]
        join_fillet = _fillet_arc(prev_lat, prev_lon,
                                  g.join_lat, g.join_lon,
                                  g.decel_start_lat, g.decel_start_lon,
                                  turn_r_nm)
        route.extend(join_fillet if join_fillet
                     else [(g.join_lat, g.join_lon)])

    route.append((g.decel_start_lat, g.decel_start_lon))

    # Densify long straight gaps so the follower always has a segment
    # nearby (arcs are already dense).
    dense: list[tuple[float, float]] = [route[0]]
    for a, b in zip(route, route[1:]):
        gap_nm = haversine_m(a[0], a[1], b[0], b[1]) / 1852.0
        n_mid = int(gap_nm / 1.5)
        for i in range(1, n_mid + 1):
            t = i / (n_mid + 1)
            dense.append((_interp(a[0], b[0], t), _interp(a[1], b[1], t)))
        dense.append(b)

    # ── Vertical profile along the route ─────────────────────────────
    # Climb at the measured gradient (climb rate over forward speed)
    # until cruise alt, then level. The plane climbs THROUGH the
    # departure arc — no "climb first, then turn" fiction.
    v_climb_kts = g.v_rotate * 1.3
    climb_grad_ft_per_nm = (_CLIMB_RATE_FPS / (v_climb_kts * 1.68781)) * 6076.12
    cum_nm = 0.0
    prev_pt = dense[0]
    for idx, (lat, lon) in enumerate(dense[1:], start=1):
        cum_nm += haversine_m(prev_pt[0], prev_pt[1], lat, lon) / 1852.0
        prev_pt = (lat, lon)
        alt = min(g.cruise_alt_ft,
                  g.dep_alt_ft + cum_nm * climb_grad_ft_per_nm)
        climbing = alt < g.cruise_alt_ft - 50.0
        hdg = bearing_deg(dense[idx - 1][0], dense[idx - 1][1], lat, lon)
        gear = cum_nm < 0.3
        flap = 0.5 if cum_nm < 0.8 else 0.0
        pts.append(PathPoint(
            lat, lon, alt,
            v_climb_kts if climbing else g.v_cruise,
            hdg, "CLIMB" if climbing else "CRUISE",
            gear, flap, 0.95 if climbing else 0.0,
        ))

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
    v_cruise_kts: Optional[float] = None,  # learned envelope cruise; ratio fallback
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
        v_cruise_kts=v_cruise_kts,
    )
    keyframes = _build_keyframes(geometry)
    preview = _build_preview(geometry, keyframes)
    # Stamp each near_point trigger with its fix's arc-length on the
    # polyline: the pass-by branch may only believe "the fix is behind
    # me" once the follower's progress has actually reached it (on a
    # loop's outbound leg every arrival fix sits behind the plane and
    # the whole pattern cascaded in one tick).
    def _stamp(trig):
        if trig.kind == "near_point":
            best = min(preview,
                       key=lambda pp: (pp.lat - trig.lat) ** 2
                                      + (pp.lon - trig.lon) ** 2)
            trig.along_gate_nm = max(
                0.0, best.dist_from_start_nm - max(1.0, trig.value * 2.0))
        for sub in trig.subs:
            _stamp(sub)
    for kf in keyframes:
        _stamp(kf.trigger)
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
