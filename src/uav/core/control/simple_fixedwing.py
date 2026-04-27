"""
Dumb-soldier controller.

Commander–soldier model: the ribbon gives orders, this controller executes
them without second-guessing. Only hardware truths are enforced (±1 on
surfaces, [0,1] on throttle/brakes). No phase logic. No rate limiters. No
hardcoded pitch curves. No stall/overspeed protection. No altitude-dependent
roll scheduling. If the ribbon says "bank 90°," we bank 90°.

Control laws:
  heading → bank → aileron (cascaded)
  altitude → pitch (single PID)
  speed    → throttle (single PID; bypassed if throttle is ribbon-commanded)

Optional passthrough clamps from Targets: roll_limit, pitch_limit. These are
commander-issued orders. The soldier honors them because the ribbon asked, not
because it has policy of its own.
"""
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


# Cascaded heading-loop defaults. Used when calibration is absent; gains may
# be overridden via derive_gains() from a learned envelope.
BANK_PER_HDG_ERROR = 1.5   # degrees bank commanded per degree heading error
BANK_INNER_KP = 0.020      # aileron per degree of bank error
BANK_INNER_KD = 0.010      # aileron damping per deg/s of bank rate


@dataclass
class ControlGains:
    bank_per_hdg_error: float = BANK_PER_HDG_ERROR
    bank_inner_kp: float = BANK_INNER_KP
    bank_inner_kd: float = BANK_INNER_KD


def derive_gains(envelope) -> ControlGains:
    """Convert measured roll sensitivity into inner-loop gains.

    With no calibration (confidence < 0.3) we return defaults.
    """
    cal_conf = getattr(envelope, 'calibration_confidence', 0.0)
    if cal_conf < 0.3:
        return ControlGains()

    roll_sens = getattr(envelope, 'roll_sensitivity', 0.0)
    if roll_sens < 1.0:
        return ControlGains()

    # kp targets ~5 deg/s correction per 10° bank error
    kp = max(0.005, min(0.06, 5.0 / (roll_sens * 10.0)))
    kd = max(0.002, min(0.02, 2.0 / roll_sens))
    return ControlGains(
        bank_per_hdg_error=BANK_PER_HDG_ERROR,
        bank_inner_kp=kp,
        bank_inner_kd=kd,
    )


class SimpleFixedWingController(Controller):
    def __init__(
        self,
        heading_pid: PID,
        altitude_pid: PID,
        airspeed_pid: PID,
        cruise_throttle: float,
        gains: ControlGains | None = None,
    ) -> None:
        # heading_pid kept for API compat — inner loop is cascaded, not a PID
        self.heading_pid = heading_pid
        self.altitude_pid = altitude_pid
        self.airspeed_pid = airspeed_pid
        self.cruise_throttle = cruise_throttle
        self.gains = gains or ControlGains()
        self._prev_bank_deg = 0.0
        self._prev_hdg_deg: float | None = None

    def compute(self, telemetry: Telemetry, targets: Targets, dt: float) -> Actuators:
        hdg_error = _wrap_deg(targets.heading_deg - telemetry.heading_deg)
        alt_error = targets.altitude_ft - telemetry.altitude_ft
        spd_error = targets.airspeed_kts - telemetry.airspeed_kts

        # ── Heading → Bank → Aileron ─────────────────────────────────
        g = self.gains
        target_bank_deg = hdg_error * g.bank_per_hdg_error

        # Commander-issued roll clamp (optional). roll_limit is in [-1,+1]
        # actuator units; we map 1.0 → 90° of commanded bank.
        if targets.roll_limit is not None:
            max_bank = abs(targets.roll_limit) * 90.0
            target_bank_deg = _clamp(target_bank_deg, max_bank)

        bank_error = target_bank_deg - telemetry.roll_deg
        bank_rate = (telemetry.roll_deg - self._prev_bank_deg) / dt if dt > 0 else 0.0
        self._prev_bank_deg = telemetry.roll_deg

        roll_cmd = g.bank_inner_kp * bank_error - g.bank_inner_kd * bank_rate
        roll_cmd = _clamp(roll_cmd, 1.0)  # hardware truth

        # ── Altitude → Pitch ─────────────────────────────────────────
        pitch_cmd = self.altitude_pid.update(alt_error, dt)

        # Commander-issued pitch clamp. Nose-up cap is always honored; nose-down
        # cap is opt-in (only honored if pitch_down_limit is set, so phases
        # that need full nose-down authority for stall recovery are unaffected).
        # Descent keyframes set pitch_down_limit to prevent a glideslope-chase
        # dive from converting altitude into speed past structural limits.
        if targets.pitch_limit is not None:
            pitch_cmd = min(pitch_cmd, abs(targets.pitch_limit))
        if targets.pitch_down_limit is not None:
            pitch_cmd = max(pitch_cmd, -abs(targets.pitch_down_limit))
        pitch_cmd = _clamp(pitch_cmd, 1.0)  # hardware truth

        # ── Speed → Throttle ─────────────────────────────────────────
        # If ribbon commanded throttle directly, honor it. Otherwise PID.
        if targets.throttle is not None:
            throttle_cmd = targets.throttle
        else:
            throttle_cmd = self.cruise_throttle + self.airspeed_pid.update(spd_error, dt)
        throttle_cmd = max(0.0, min(1.0, throttle_cmd))  # hardware truth

        # ── Yaw (ribbon-driven yaw-hold only; no standalone yaw PID) ─
        yaw_cmd = 0.0
        if targets.yaw_hold:
            yaw_kp = targets.yaw_kp if targets.yaw_kp is not None else 0.02
            yaw_limit = targets.yaw_limit if targets.yaw_limit is not None else 0.5

            if self._prev_hdg_deg is None or dt <= 0:
                hdg_rate = 0.0
            else:
                hdg_rate = _wrap_deg(telemetry.heading_deg - self._prev_hdg_deg) / dt
            self._prev_hdg_deg = telemetry.heading_deg

            yaw_cmd = _clamp(yaw_kp * hdg_error - 0.008 * hdg_rate, yaw_limit)
        else:
            self._prev_hdg_deg = telemetry.heading_deg

        brake_ratio = targets.brake_ratio if targets.brake_ratio is not None else 0.0
        brake_ratio = max(0.0, min(1.0, brake_ratio))  # hardware truth
        gear_down = targets.gear_down if targets.gear_down is not None else True
        flap_ratio = targets.flap_ratio if targets.flap_ratio is not None else 0.0
        flap_ratio = max(0.0, min(1.0, flap_ratio))  # hardware truth

        return Actuators(
            throttle=throttle_cmd,
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=yaw_cmd,
            brake_ratio=brake_ratio,
            gear_down=gear_down,
            flap_ratio=flap_ratio,
        )
