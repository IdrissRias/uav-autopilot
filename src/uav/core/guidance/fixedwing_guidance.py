from __future__ import annotations

from uav.core.guidance.base import Guidance
from uav.sim.types import Telemetry, Targets


class FixedWingGuidance(Guidance):
    def compute(self, telemetry: Telemetry, desired: Targets) -> Targets:
        heading = desired.heading_deg
        altitude = desired.altitude_ft
        airspeed = desired.airspeed_kts

        if heading is None:
            heading = telemetry.heading_deg
        if altitude is None:
            altitude = telemetry.altitude_ft
        if airspeed is None:
            airspeed = telemetry.airspeed_kts

        # Pass all mode-set fields through — only resolve the three nav fields.
        return Targets(
            heading_deg=heading,
            altitude_ft=altitude,
            airspeed_kts=airspeed,
            climb_rate_fpm=desired.climb_rate_fpm,
            throttle=desired.throttle,
            brake_ratio=desired.brake_ratio,
            gear_down=desired.gear_down,
            roll_limit=desired.roll_limit,
            pitch_limit=desired.pitch_limit,
            yaw_hold=desired.yaw_hold,
            yaw_kp=desired.yaw_kp,
            yaw_ki=desired.yaw_ki,
            yaw_limit=desired.yaw_limit,
            yaw_full_deg=desired.yaw_full_deg,
            pitch_protect_kts=desired.pitch_protect_kts,
            pitch_protect_gain=desired.pitch_protect_gain,
            flap_ratio=desired.flap_ratio,
            reset_alt_pid=desired.reset_alt_pid,
        )
