"""
Safety Envelope — always-on overrides that protect the aircraft.

These run AFTER the controller produces Actuators but BEFORE they're
sent to X-Plane.  They enforce hard physical limits that no flight
director or PID should ever violate.

Layers:
  1. Descent rate envelope (GPWS-inspired)
  2. Stall protection (speed floor with bank compensation)
  3. Speed limit (never exceed Vne)
  4. Bank limit (reduces near ground)

This module provides a single function: enforce_envelope() that takes
telemetry, targets, and actuators and returns corrected actuators + targets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from uav.sim.types import Telemetry, Targets, Actuators


@dataclass
class EnvelopeCorrection:
    """What the safety envelope changed, for logging."""
    descent_limited: bool = False
    stall_protected: bool = False
    overspeed_protected: bool = False
    bank_limited: bool = False
    max_descent_fpm: float = 0.0


def enforce_envelope(
    telemetry: Telemetry,
    targets: Targets,
    actuators: Actuators,
    *,
    v_stall: float = 77.0,
    v_never_exceed: float = 250.0,
    max_bank_deg: float = 30.0,
) -> tuple[Targets, Actuators, EnvelopeCorrection]:
    """Apply safety envelope overrides.

    Returns (corrected_targets, corrected_actuators, correction_info).
    The caller should use the corrected values.
    """
    correction = EnvelopeCorrection()
    agl_ft = (telemetry.agl_m * 3.28084) if not math.isnan(telemetry.agl_m) else 0.0

    # ── 1. Descent rate envelope (GPWS-inspired) ────────────────────
    # max_descent_fpm = min(1500, AGL_ft × 2.0)
    # At 50ft AGL → max 100 fpm (soft touchdown)
    # At 500ft AGL → max 1000 fpm
    # Above 750ft → capped at 1500 fpm
    max_descent = min(1500.0, max(50.0, agl_ft * 2.0))
    correction.max_descent_fpm = max_descent

    # Estimate current descent rate from pitch attitude (rough proxy).
    # If pitch is very negative and we're low, force pitch up.
    # The actual VS enforcement happens through the altitude target:
    # if the altitude target would require exceeding max descent rate,
    # we raise the target to comply.
    if targets is not None and targets.altitude_ft is not None:
        alt_error = targets.altitude_ft - telemetry.altitude_ft
        # If commanded to descend faster than envelope allows,
        # raise the altitude target to limit descent rate.
        # At 15 Hz, each tick ≈ 0.067s. If alt_error is very negative,
        # the PID will command aggressive nose-down.
        # We limit how far below current alt the target can be.
        # max_descent_per_min → max_descent_per_second → how much alt loss
        # we allow over the next ~4 seconds of PID response.
        max_alt_drop = max_descent / 60.0 * 4.0  # 4s worth of descent
        if alt_error < -max_alt_drop:
            targets = Targets(
                heading_deg=targets.heading_deg,
                altitude_ft=telemetry.altitude_ft - max_alt_drop,
                airspeed_kts=targets.airspeed_kts,
                climb_rate_fpm=targets.climb_rate_fpm,
                throttle=targets.throttle,
                brake_ratio=targets.brake_ratio,
                gear_down=targets.gear_down,
                roll_limit=targets.roll_limit,
                pitch_limit=targets.pitch_limit,
                yaw_hold=targets.yaw_hold,
                yaw_kp=targets.yaw_kp,
                yaw_ki=targets.yaw_ki,
                yaw_limit=targets.yaw_limit,
                yaw_full_deg=targets.yaw_full_deg,
                pitch_protect_kts=targets.pitch_protect_kts,
                pitch_protect_gain=targets.pitch_protect_gain,
                flap_ratio=targets.flap_ratio,
            )
            correction.descent_limited = True

    # ── 2. Stall protection ─────────────────────────────────────────
    # V_safe = Vs × 1.3 × sqrt(1/cos(bank))
    # If airspeed drops below V_safe, force nose down + add throttle.
    bank_rad = math.radians(min(abs(telemetry.roll_deg), 60.0))
    cos_bank = max(math.cos(bank_rad), 0.5)
    v_safe = v_stall * 1.3 * math.sqrt(1.0 / cos_bank)

    if telemetry.airspeed_kts < v_safe:
        # Nose-down authority: pitch down proportional to deficit
        deficit = v_safe - telemetry.airspeed_kts
        pitch_down = min(0.15, deficit * 0.01)  # gentle but effective
        actuators = Actuators(
            throttle=min(1.0, actuators.throttle + 0.3),  # add power
            roll=actuators.roll,
            pitch=actuators.pitch - pitch_down,
            yaw=actuators.yaw,
            brake_ratio=0.0,  # release brakes
            gear_down=actuators.gear_down,
            flap_ratio=actuators.flap_ratio,
        )
        correction.stall_protected = True

    # ── 3. Overspeed protection ─────────────────────────────────────
    # Never exceed Vne.  If approaching, cut throttle + pitch up.
    if telemetry.airspeed_kts > v_never_exceed * 0.95:
        excess = telemetry.airspeed_kts - v_never_exceed * 0.9
        pitch_up = min(0.10, excess * 0.005)
        throttle_cut = max(0.0, actuators.throttle - excess * 0.02)
        actuators = Actuators(
            throttle=throttle_cut,
            roll=actuators.roll,
            pitch=actuators.pitch + pitch_up,
            yaw=actuators.yaw,
            brake_ratio=actuators.brake_ratio,
            gear_down=actuators.gear_down,
            flap_ratio=actuators.flap_ratio,
        )
        correction.overspeed_protected = True

    # ── 4. Bank limit (reduces near ground) ──────────────────────────
    # Full bank above 500ft AGL, linearly reduced to 10° at 0ft AGL.
    if agl_ft < 500.0:
        max_bank_near_ground = 10.0 + (max_bank_deg - 10.0) * (agl_ft / 500.0)
    else:
        max_bank_near_ground = max_bank_deg

    if abs(telemetry.roll_deg) > max_bank_near_ground:
        # Limit roll command to prevent further banking
        sign = 1.0 if telemetry.roll_deg > 0 else -1.0
        excess_bank = abs(telemetry.roll_deg) - max_bank_near_ground
        roll_correction = -sign * min(0.10, excess_bank * 0.005)
        actuators = Actuators(
            throttle=actuators.throttle,
            roll=actuators.roll + roll_correction,
            pitch=actuators.pitch,
            yaw=actuators.yaw,
            brake_ratio=actuators.brake_ratio,
            gear_down=actuators.gear_down,
            flap_ratio=actuators.flap_ratio,
        )
        correction.bank_limited = True

    return targets, actuators, correction
