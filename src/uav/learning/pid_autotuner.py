"""Peregrine — PID Auto-Tuner.

After each flight, analyzes tracking errors (heading, altitude, speed)
and adjusts PID gains to reduce those errors on the next flight.

This is the AI fine-tuning layer: the autopilot gets better at flying
each specific aircraft type with every flight.

Strategy:
  - If tracking error is consistently high → increase kp
  - If oscillating → increase kd, decrease ki
  - If steady-state offset → increase ki
  - Changes are conservative (±10% per flight) to avoid instability
  - Only adjusts if enough data (>30s of tracking data per axis)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..db import local_db

log = logging.getLogger(__name__)

# Maximum gain adjustment per flight (±10%)
MAX_ADJUST = 0.10

# Minimum tracking data duration (seconds) to consider adjusting
MIN_DATA_DURATION_S = 30.0

# Error thresholds — below these, gains are "good enough"
HDG_ERROR_GOOD_DEG = 3.0      # heading error below 3° is fine
ALT_ERROR_GOOD_FT = 50.0      # altitude error below 50ft is fine
SPD_ERROR_GOOD_KTS = 3.0      # speed error below 3kts is fine

# Oscillation detection: if std/mean > this ratio, we're oscillating
OSCILLATION_RATIO = 1.5


def auto_tune(
    icao_type: str,
    tracking_data: Dict[str, Any],
    current_gains: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Dict[str, float]] | None:
    """Analyze tracking performance and suggest PID gain adjustments.

    Args:
        icao_type: Aircraft ICAO type code
        tracking_data: Dict with keys:
            heading_errors: list[float] — signed heading errors (deg) per tick
            altitude_errors: list[float] — signed altitude errors (ft) per tick
            speed_errors: list[float] — signed speed errors (kts) per tick
            dt: float — time between ticks (seconds)
        current_gains: Current PID gains dict. If None, loads from database.

    Returns:
        Updated PID gains dict, or None if no adjustment needed.
    """
    if not tracking_data:
        return None

    # Load current gains from database if not provided
    if current_gains is None:
        row = local_db.get_aircraft(icao_type)
        if not row:
            return None
        current_gains = row.get("pid_gains") or {}

    hdg_errs = tracking_data.get("heading_errors", [])
    alt_errs = tracking_data.get("altitude_errors", [])
    spd_errs = tracking_data.get("speed_errors", [])
    dt = tracking_data.get("dt", 0.1)

    adjusted = False
    new_gains = {
        "heading": dict(current_gains.get("heading", {"kp": 0.004, "ki": 0.0001, "kd": 0.008})),
        "altitude": dict(current_gains.get("altitude", {"kp": 0.001, "ki": 0.00008, "kd": 0.006})),
        "airspeed": dict(current_gains.get("airspeed", {"kp": 0.015, "ki": 0.0, "kd": 0.004})),
    }

    # ── Heading PID ──
    if len(hdg_errs) * dt > MIN_DATA_DURATION_S:
        adj = _analyze_axis(hdg_errs, HDG_ERROR_GOOD_DEG)
        if adj:
            _apply_adjustment(new_gains["heading"], adj)
            adjusted = True
            log.info(f"[AUTOTUNE] Heading: {adj}")

    # ── Altitude PID ──
    if len(alt_errs) * dt > MIN_DATA_DURATION_S:
        adj = _analyze_axis(alt_errs, ALT_ERROR_GOOD_FT)
        if adj:
            _apply_adjustment(new_gains["altitude"], adj)
            adjusted = True
            log.info(f"[AUTOTUNE] Altitude: {adj}")

    # ── Airspeed PID ──
    if len(spd_errs) * dt > MIN_DATA_DURATION_S:
        adj = _analyze_axis(spd_errs, SPD_ERROR_GOOD_KTS)
        if adj:
            _apply_adjustment(new_gains["airspeed"], adj)
            adjusted = True
            log.info(f"[AUTOTUNE] Airspeed: {adj}")

    if not adjusted:
        log.info("[AUTOTUNE] No adjustments needed — tracking is good")
        return None

    # Save to database
    try:
        local_db.update_aircraft_pid_gains(icao_type, new_gains)
        log.info(f"[AUTOTUNE] Updated PID gains for {icao_type}")
    except Exception as e:
        log.error(f"[AUTOTUNE] Failed to save gains: {e}")

    return new_gains


def _analyze_axis(errors: list[float], good_threshold: float) -> Dict[str, float] | None:
    """Analyze tracking errors for one axis and return gain adjustments.

    Returns dict of {"kp": delta, "ki": delta, "kd": delta} or None if good.
    """
    if len(errors) < 10:
        return None

    abs_errors = [abs(e) for e in errors]
    mean_err = sum(abs_errors) / len(abs_errors)

    # If error is already good, no adjustment needed
    if mean_err < good_threshold:
        return None

    # Compute statistics
    mean_signed = sum(errors) / len(errors)
    variance = sum((e - mean_signed) ** 2 for e in errors) / len(errors)
    std = variance ** 0.5

    adjustments = {"kp": 0.0, "ki": 0.0, "kd": 0.0}

    # Check for oscillation (high std relative to mean error)
    is_oscillating = std > mean_err * OSCILLATION_RATIO if mean_err > 0 else False

    if is_oscillating:
        # Oscillating: increase damping (kd), decrease integral (ki)
        adjustments["kd"] = MAX_ADJUST
        adjustments["ki"] = -MAX_ADJUST * 0.5
        adjustments["kp"] = -MAX_ADJUST * 0.3  # slightly reduce proportional too
    elif abs(mean_signed) > good_threshold * 0.5:
        # Steady-state offset: increase integral
        adjustments["ki"] = MAX_ADJUST * 0.5
    else:
        # High error but not oscillating: increase proportional
        # Scale adjustment by how far we are from "good"
        overshoot = min(mean_err / good_threshold, 3.0)
        adjustments["kp"] = MAX_ADJUST * min(overshoot * 0.3, 1.0)

    # Only return if at least one adjustment is non-zero
    if any(abs(v) > 0.001 for v in adjustments.values()):
        return adjustments
    return None


def _apply_adjustment(gains: Dict[str, float], adj: Dict[str, float]) -> None:
    """Apply percentage adjustments to gains, clamping to reasonable ranges."""
    for key in ("kp", "ki", "kd"):
        if key not in gains or key not in adj:
            continue
        current = gains[key]
        delta_pct = adj[key]

        if current == 0.0 and delta_pct > 0:
            # Bootstrap: if gain is zero and we want to increase, set a small value
            if key == "ki":
                gains[key] = 0.00001
            elif key == "kd":
                gains[key] = 0.001
            continue

        new_val = current * (1.0 + delta_pct)
        # Clamp: never go negative, never go above 10x original
        new_val = max(0.0, min(new_val, current * 10.0 if current > 0 else 1.0))
        gains[key] = round(new_val, 8)
