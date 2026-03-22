from __future__ import annotations

from dataclasses import dataclass

from uav.core.control.base import Controller
from uav.core.control.pid import PID
from uav.sim.types import Telemetry, Actuators, Targets


def _wrap_deg(error_deg: float) -> float:
    while error_deg > 180.0:
        error_deg -= 360.0
    while error_deg < -180.0:
        error_deg += 360.0
    return error_deg


def _clamp(value: float, limit: float) -> float:
    if limit <= 0.0:
        return value
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


# ── Default cascaded heading control constants ──────────────────
# These are used when the aircraft is uncalibrated (calibration_confidence < 0.3).
# After calibration, gains are derived from measured control sensitivity.
MAX_BANK_DEG = 30.0        # max bank angle — drone operations, faster turns
BANK_PER_HDG_ERROR = 1.5   # deg bank per deg heading error (1.5:1 — 10° off = 15° bank)
BANK_INNER_KP = 0.020      # aileron per deg of bank error (inner loop P gain)
BANK_INNER_KD = 0.008      # aileron per deg/s of bank rate (stronger damping)


@dataclass
class ControlGains:
    """Derived control gains — either from calibration or defaults."""
    bank_per_hdg_error: float = BANK_PER_HDG_ERROR
    bank_inner_kp: float = BANK_INNER_KP
    bank_inner_kd: float = BANK_INNER_KD
    max_bank_deg: float = MAX_BANK_DEG


def derive_gains(envelope) -> ControlGains:
    """Convert measured control sensitivity into PID gains.

    Args:
        envelope: AircraftEnvelope with pitch_sensitivity, roll_sensitivity, etc.

    Returns:
        ControlGains with calibration-derived values, or defaults if uncalibrated.
    """
    cal_conf = getattr(envelope, 'calibration_confidence', 0.0)
    if cal_conf < 0.3:
        return ControlGains()  # use defaults

    roll_sens = getattr(envelope, 'roll_sensitivity', 0.0)

    if roll_sens < 1.0:
        return ControlGains()  # sensitivity too low / unreliable

    # Inner loop: we want ~5 deg/s roll correction per 10° bank error.
    # bank_inner_kp = desired_rate / (roll_sensitivity * typical_error)
    # With roll_sens = 15 deg/s/unit → kp = 5 / (15 * 10) = 0.033
    # With roll_sens = 30 deg/s/unit → kp = 5 / (30 * 10) = 0.017
    # Clamp to reasonable range [0.005, 0.06]
    kp = _clamp(5.0 / (roll_sens * 10.0), 0.06)
    kp = max(kp, 0.005)

    # Damping scales inversely with sensitivity too
    kd = _clamp(2.0 / roll_sens, 0.02)
    kd = max(kd, 0.002)

    return ControlGains(
        bank_per_hdg_error=BANK_PER_HDG_ERROR,  # outer loop is aircraft-independent
        bank_inner_kp=kp,
        bank_inner_kd=kd,
        max_bank_deg=MAX_BANK_DEG,
    )


class SimpleFixedWingController(Controller):
    def __init__(
        self,
        heading_pid: PID,
        altitude_pid: PID,
        airspeed_pid: PID,
        cruise_throttle: float,
        pitch_rate_limit_per_s: float = 2.0,
        roll_rate_limit_per_s: float = 0.3,
        gains: ControlGains | None = None,
    ) -> None:
        self.heading_pid = heading_pid  # kept for API compat but no longer drives roll
        self.altitude_pid = altitude_pid
        self.airspeed_pid = airspeed_pid
        self.cruise_throttle = cruise_throttle
        self.pitch_rate_limit_per_s = pitch_rate_limit_per_s
        self.roll_rate_limit_per_s = roll_rate_limit_per_s
        self.gains = gains or ControlGains()
        self._prev_pitch_cmd = 0.0
        self._prev_roll_cmd = 0.0
        self._prev_bank_deg = 0.0   # for bank rate damping
        self._prev_hdg_deg = None    # for heading rate damping (yaw) — None = first tick

    def compute(self, telemetry: Telemetry, targets: Targets, dt: float) -> Actuators:
        hdg_error = _wrap_deg(targets.heading_deg - telemetry.heading_deg)
        alt_error = targets.altitude_ft - telemetry.altitude_ft
        spd_error = targets.airspeed_kts - telemetry.airspeed_kts

        # ── CASCADED HEADING → BANK → AILERON ──────────────────────
        # Outer loop: heading error → desired bank angle
        # Linear proportional: 10° heading error → 15° bank, capped at max_bank_deg
        g = self.gains
        target_bank_deg = _clamp(hdg_error * g.bank_per_hdg_error, g.max_bank_deg)

        # Apply roll_limit from targets (e.g. takeoff limits bank)
        if targets.roll_limit is not None:
            # roll_limit is in ratio (-1..+1), convert to approximate degrees
            # At limit 0.20 → ~20° bank max
            max_bank_from_limit = abs(targets.roll_limit) * 100.0  # rough mapping
            target_bank_deg = _clamp(target_bank_deg, min(g.max_bank_deg, max_bank_from_limit))

        # Inner loop: bank error → aileron command
        bank_error_deg = target_bank_deg - telemetry.roll_deg
        bank_rate_deg_s = (telemetry.roll_deg - self._prev_bank_deg) / dt if dt > 0 else 0.0
        self._prev_bank_deg = telemetry.roll_deg

        roll_cmd = g.bank_inner_kp * bank_error_deg - g.bank_inner_kd * bank_rate_deg_s

        # Clamp aileron output
        roll_cmd = _clamp(roll_cmd, 1.0)

        # ── ALTITUDE → PITCH  /  SPEED → THROTTLE ─────────────────
        # Philosophy: PITCH controls altitude. THROTTLE controls speed.
        # The PID has FULL authority over the elevator — no artificial
        # pitch limits during normal flight. The safety layer (SafetyLimits)
        # is the only hard stop (structural limit of the airframe).
        #
        # This is how a real pilot flies: push the nose where it needs
        # to go to hold altitude, and set power for the desired speed.

        # Reset altitude PID on phase transitions (e.g. CRUISE→APPROACH)
        # to clear accumulated integral bias that would fight the new target.
        if getattr(targets, 'reset_alt_pid', False):
            self.altitude_pid.reset()

        pitch_cmd = self.altitude_pid.update(alt_error, dt)
        throttle_cmd = self.cruise_throttle

        # CLIMB behavior: constant moderate pitch while below target.
        # Like a real pilot: set pitch and power, climb at whatever rate results.
        # Near the target, blend smoothly to level flight.
        if targets.throttle is not None and targets.climb_rate_fpm is not None:
            throttle_cmd = targets.throttle
            if alt_error > 200.0:
                # Well below target: hold steady climb pitch
                pitch_cmd = 0.07
            elif alt_error > 0:
                # Approaching target: blend from climb pitch to level (0.07 → 0.0)
                pitch_cmd = 0.07 * (alt_error / 200.0)
            else:
                # Above target: gentle push down, proportional
                pitch_cmd = max(-0.15, alt_error * 0.0003)
            self.altitude_pid.reset()
            self.airspeed_pid.reset()
        else:
            if targets.throttle is not None:
                throttle_cmd = targets.throttle
            else:
                throttle_cmd += self.airspeed_pid.update(spd_error, dt)

        # Airspeed protection: if dangerously slow (near stall), pitch
        # down to trade altitude for airspeed — survival takes priority.
        if targets.pitch_protect_kts is not None and targets.pitch_protect_gain is not None:
            if telemetry.airspeed_kts < targets.pitch_protect_kts:
                delta = (targets.pitch_protect_kts - telemetry.airspeed_kts) * targets.pitch_protect_gain
                pitch_cmd -= min(delta, 0.20)

        # Enforce pitch_limit as an UPPER bound only (nose-up cap).
        # The plane must always be free to push nose DOWN — never restrict that.
        # This prevents PID integral windup from causing sharp pitch-ups on
        # phase transitions (e.g. CRUISE→APPROACH where old integral fights descent).
        if targets.pitch_limit is not None:
            pitch_cmd = min(pitch_cmd, abs(targets.pitch_limit))

        # Roll rate limit: smooth out corrections.
        if dt > 0:
            max_roll_delta = self.roll_rate_limit_per_s * dt
            roll_delta = roll_cmd - self._prev_roll_cmd
            if roll_delta > max_roll_delta:
                roll_cmd = self._prev_roll_cmd + max_roll_delta
            elif roll_delta < -max_roll_delta:
                roll_cmd = self._prev_roll_cmd - max_roll_delta
        self._prev_roll_cmd = roll_cmd

        # Pitch rate limit to avoid oscillation/hunting.
        if dt > 0:
            max_delta = self.pitch_rate_limit_per_s * dt
            delta = pitch_cmd - self._prev_pitch_cmd
            if delta > max_delta:
                pitch_cmd = self._prev_pitch_cmd + max_delta
            elif delta < -max_delta:
                pitch_cmd = self._prev_pitch_cmd - max_delta
        self._prev_pitch_cmd = pitch_cmd

        # ── YAW (takeoff/low-altitude only) ─────────────────────────
        # Purely proportional + derivative. NO bang-bang, NO integral windup.
        # The correction scales linearly with the error and tapers to zero
        # as the plane converges on the target heading.
        #
        # Derivative term (heading rate damping) prevents overshoot:
        # as the plane starts turning back toward target, the rate opposes
        # the correction → smooth deceleration into the target heading.
        yaw_cmd = 0.0
        if targets.yaw_hold:
            yaw_kp = targets.yaw_kp if targets.yaw_kp is not None else 0.02
            yaw_limit = targets.yaw_limit if targets.yaw_limit is not None else 0.5

            # Heading rate: how fast heading is changing (deg/s)
            # First tick: no rate info yet, just use proportional.
            if self._prev_hdg_deg is None:
                hdg_rate = 0.0
            elif dt > 0:
                hdg_rate = _wrap_deg(telemetry.heading_deg - self._prev_hdg_deg) / dt
                # Cap rate to physical limits (~30°/s max for a small jet)
                hdg_rate = max(-30.0, min(30.0, hdg_rate))
            else:
                hdg_rate = 0.0
            self._prev_hdg_deg = telemetry.heading_deg

            # PD controller: proportional to error, damped by heading rate.
            # As error shrinks → proportional shrinks → correction tapers.
            # As plane turns toward target → rate opposes → prevents overshoot.
            yaw_kd = 0.008  # rudder per deg/s of heading rate
            yaw_cmd = yaw_kp * hdg_error - yaw_kd * hdg_rate

            yaw_cmd = _clamp(yaw_cmd, yaw_limit)
        else:
            self._prev_hdg_deg = telemetry.heading_deg

        brake_ratio = 0.0
        if targets.brake_ratio is not None:
            brake_ratio = targets.brake_ratio

        gear_down = targets.gear_down if targets.gear_down is not None else True
        flap_ratio = targets.flap_ratio if targets.flap_ratio is not None else 0.0

        return Actuators(
            throttle=throttle_cmd,
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=yaw_cmd,
            brake_ratio=brake_ratio,
            gear_down=gear_down,
            flap_ratio=flap_ratio,
        )
