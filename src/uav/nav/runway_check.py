"""
Pre-flight runway check.

Given the aircraft's current position + heading and a user-drawn runway,
decide three things:

  1. on_runway        — is the plane physically on the runway band?
  2. heading_aligned  — is the plane pointing roughly along the runway axis?
  3. room_to_roll     — is there enough runway ahead for the aircraft's
                        minimum takeoff roll?

All three must pass for a green Fly button. The check is pure: no I/O,
no side effects. Callers hand in the aircraft's takeoff_roll_ft (from the
envelope) and this module returns a RunwayCheck dataclass that maps
directly to the preflight broadcast payload.

Why separate from runway_detect.py:
  - runway_detect scans airports-DB runways by proximity to bootstrap a
    new flight. It answers "where am I?".
  - runway_check validates a *specific* runway the user already chose.
    It answers "can I take off from this one?".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from uav.nav.geo import project_onto_segment


# Heading tolerance — plane must be within this many degrees of the runway
# axis to count as "aligned". 20° is lenient enough for low-speed taxi
# misalignment but tight enough that someone pointing across the runway
# gets caught. Tweakable per-user later if needed.
HEADING_TOLERANCE_DEG = 20.0

# Safety factor applied to the aircraft's published minimum takeoff roll.
# 1.2× covers density-altitude, slight tailwind, and a soft surface.
TAKEOFF_ROLL_SAFETY_FACTOR = 1.2


@dataclass
class RunwayCheck:
    """Result of preflight_runway_check(). All numeric fields are meters/deg."""
    # Three boolean gates
    on_runway: bool
    heading_aligned: bool
    room_to_roll: bool
    # Overall — True only when all three above are True
    ok: bool

    # Diagnostics — always populated so the app can show nuance even when ok=False
    cross_track_m: float        # signed perpendicular offset from centerline (left = +)
    along_track_m: float        # distance from takeoff-end threshold along runway axis
    heading_diff_deg: float     # signed diff from runway heading (left = -, right = +)
    remaining_m: float          # distance from plane's position to far end of runway
    needed_m: float             # aircraft's min takeoff roll × safety factor
    runway_length_m: float
    runway_width_m: float
    runway_heading_deg: float

    # Human-readable reasons for each failing gate. Empty when ok=True.
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def preflight_runway_check(
    lat: float,
    lon: float,
    heading_deg: float,
    runway: Dict[str, Any],
    takeoff_roll_ft: Optional[float],
    *,
    heading_tolerance_deg: float = HEADING_TOLERANCE_DEG,
    safety_factor: float = TAKEOFF_ROLL_SAFETY_FACTOR,
) -> RunwayCheck:
    """Validate the plane-runway relationship for takeoff.

    runway must be a dict with keys lat_start, lon_start, lat_end, lon_end,
    width_m (and optionally heading_deg / length_m; derived if absent).
    takeoff_roll_ft may be None — in which case room_to_roll defaults to True
    (we can't judge what we don't know; better to warn than block).

    The returned RunwayCheck is safe to JSON-serialize for the app.
    """
    reasons: List[str] = []

    # ── Project aircraft onto the runway centerline ─────────────
    along_m, cross_m = project_onto_segment(
        lat, lon,
        runway["lat_start"], runway["lon_start"],
        runway["lat_end"], runway["lon_end"],
    )

    # Length / heading of the runway — prefer derived values if already
    # attached (list_user_runways populates them); otherwise compute.
    length_m = float(runway.get("length_m") or 0.0)
    rwy_heading = runway.get("heading_deg")
    if length_m <= 0.0 or rwy_heading is None:
        from uav.nav.geo import haversine_m, bearing_deg
        length_m = haversine_m(
            runway["lat_start"], runway["lon_start"],
            runway["lat_end"], runway["lon_end"],
        )
        rwy_heading = bearing_deg(
            runway["lat_start"], runway["lon_start"],
            runway["lat_end"], runway["lon_end"],
        )

    width_m = float(runway.get("width_m") or 20.0)

    # ── Gate 1: on_runway ──────────────────────────────────────
    # Aircraft is on the runway band when:
    #   • |cross_m| <= width/2  (within the width band)
    #   • 0 <= along_m <= length_m  (between start and end along the axis)
    # A plane sitting 5 m past the far end is NOT on the runway even if
    # perfectly centered — there's no runway ahead to roll on.
    within_width = abs(cross_m) <= (width_m / 2.0)
    # Use a 0.5m slack on length so sub-meter float noise at the threshold
    # doesn't trip the "behind threshold" reason when the plane is effectively
    # at the threshold.
    EPS_M = 0.5
    within_length = (along_m >= -EPS_M) and (along_m <= length_m + EPS_M)
    on_runway = within_width and within_length

    if not on_runway:
        if not within_width:
            reasons.append(
                f"Off centerline by {abs(cross_m):.1f} m "
                f"(runway is only {width_m:.0f} m wide)"
            )
        if not within_length:
            if along_m < -EPS_M:
                reasons.append(f"Behind the takeoff-end threshold by {-along_m:.0f} m")
            elif along_m > length_m + EPS_M:
                reasons.append(f"Past the far end by {along_m - length_m:.0f} m")

    # ── Gate 2: heading_aligned ────────────────────────────────
    # Signed heading diff in (-180, +180]. Negative = aircraft pointed left
    # of runway axis, positive = right.
    diff = ((heading_deg - rwy_heading + 540.0) % 360.0) - 180.0
    heading_aligned = abs(diff) <= heading_tolerance_deg
    if not heading_aligned:
        side = "right" if diff > 0 else "left"
        reasons.append(
            f"Heading is {abs(diff):.0f}° off axis to the {side} "
            f"(tolerance is ±{heading_tolerance_deg:.0f}°)"
        )

    # ── Gate 3: room_to_roll ───────────────────────────────────
    # Distance from aircraft's projected position to the far end. If the
    # plane is behind start (along_m < 0), remaining is length + |along_m|
    # because the plane has to cover that extra ground before reaching start.
    # Realistically along_m < 0 also fails on_runway, but we compute cleanly.
    if along_m < 0.0:
        remaining_m = length_m + abs(along_m)  # the plane still has to enter + roll the whole runway
    else:
        remaining_m = max(0.0, length_m - along_m)

    if takeoff_roll_ft and takeoff_roll_ft > 0.0:
        needed_m = (takeoff_roll_ft / 3.28084) * safety_factor
        room_to_roll = remaining_m >= needed_m
        if not room_to_roll:
            short_m = needed_m - remaining_m
            reasons.append(
                f"Only {remaining_m:.0f} m of runway ahead — "
                f"need {needed_m:.0f} m (short by {short_m:.0f} m). "
                f"Back up or pick a longer runway."
            )
    else:
        # Unknown aircraft takeoff roll — warn rather than block.
        needed_m = 0.0
        room_to_roll = True
        reasons.append(
            "Aircraft minimum takeoff roll unknown — room-to-roll check skipped."
        )

    ok = on_runway and heading_aligned and room_to_roll

    return RunwayCheck(
        on_runway=on_runway,
        heading_aligned=heading_aligned,
        room_to_roll=room_to_roll,
        ok=ok,
        cross_track_m=round(cross_m, 2),
        along_track_m=round(along_m, 2),
        heading_diff_deg=round(diff, 1),
        remaining_m=round(remaining_m, 1),
        needed_m=round(needed_m, 1),
        runway_length_m=round(length_m, 1),
        runway_width_m=width_m,
        runway_heading_deg=round(rwy_heading, 1),
        reasons=reasons,
    )
