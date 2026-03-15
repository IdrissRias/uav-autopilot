from __future__ import annotations

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


class SimpleFixedWingController(Controller):
    def __init__(
        self,
        heading_pid: PID,
        altitude_pid: PID,
        airspeed_pid: PID,
        cruise_throttle: float,
        pitch_rate_limit_per_s: float = 0.6,
    ) -> None:
        self.heading_pid = heading_pid
        self.altitude_pid = altitude_pid
        self.airspeed_pid = airspeed_pid
        self.cruise_throttle = cruise_throttle
        self.pitch_rate_limit_per_s = pitch_rate_limit_per_s
        self._yaw_integral = 0.0
        self._prev_pitch_cmd = 0.0

    def compute(self, telemetry: Telemetry, targets: Targets, dt: float) -> Actuators:
        hdg_error = _wrap_deg(targets.heading_deg - telemetry.heading_deg)
        alt_error = targets.altitude_ft - telemetry.altitude_ft
        spd_error = targets.airspeed_kts - telemetry.airspeed_kts

        roll_cmd = self.heading_pid.update(hdg_error, dt)
        pitch_cmd = self.altitude_pid.update(alt_error, dt)
        throttle_cmd = self.cruise_throttle

        # CLIMB behavior: if mode sets a throttle and climb_rate_fpm, hold airspeed with pitch.
        if targets.throttle is not None and targets.climb_rate_fpm is not None:
            throttle_cmd = targets.throttle
            # Use airspeed PID to command pitch: if we're fast, pitch up; if slow, pitch down.
            pitch_cmd = -self.airspeed_pid.update(spd_error, dt)
            # Altitude PID not used in this regime.
            self.altitude_pid.reset()
        else:
            if targets.throttle is not None:
                throttle_cmd = targets.throttle
            else:
                throttle_cmd += self.airspeed_pid.update(spd_error, dt)

        if targets.roll_limit is not None:
            roll_cmd = _clamp(roll_cmd, targets.roll_limit)
        if targets.pitch_limit is not None:
            pitch_cmd = _clamp(pitch_cmd, targets.pitch_limit)

        # Pitch rate limit to avoid oscillation/hunting.
        if dt > 0:
            max_delta = self.pitch_rate_limit_per_s * dt
            delta = pitch_cmd - self._prev_pitch_cmd
            if delta > max_delta:
                pitch_cmd = self._prev_pitch_cmd + max_delta
            elif delta < -max_delta:
                pitch_cmd = self._prev_pitch_cmd - max_delta
        self._prev_pitch_cmd = pitch_cmd

        # Airspeed protection: if we're slow, bias pitch down to regain speed.
        if targets.pitch_protect_kts is not None and targets.pitch_protect_gain is not None:
            if telemetry.airspeed_kts < targets.pitch_protect_kts:
                delta = (targets.pitch_protect_kts - telemetry.airspeed_kts) * targets.pitch_protect_gain
                pitch_cmd -= min(delta, 0.3)

        yaw_cmd = 0.0
        if targets.yaw_hold:
            yaw_kp = targets.yaw_kp if targets.yaw_kp is not None else 0.02
            yaw_ki = targets.yaw_ki if targets.yaw_ki is not None else 0.0
            yaw_limit = targets.yaw_limit if targets.yaw_limit is not None else 0.5
            yaw_full_deg = targets.yaw_full_deg if targets.yaw_full_deg is not None else 8.0
            self._yaw_integral += hdg_error * dt
            yaw_cmd = yaw_kp * hdg_error + yaw_ki * self._yaw_integral
            if abs(hdg_error) >= yaw_full_deg:
                yaw_cmd = yaw_limit if hdg_error > 0 else -yaw_limit
            yaw_cmd = _clamp(yaw_cmd, yaw_limit)
        else:
            self._yaw_integral = 0.0

        brake_ratio = 0.0
        if targets.brake_ratio is not None:
            brake_ratio = targets.brake_ratio

        return Actuators(
            throttle=throttle_cmd,
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=yaw_cmd,
            brake_ratio=brake_ratio,
        )
