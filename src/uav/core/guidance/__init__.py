from __future__ import annotations

from .base import Guidance
from uav.sim.types import Targets, Telemetry


class PassthroughGuidance(Guidance):
    """Fills in None setpoints from current telemetry. No policy."""

    def compute(self, telemetry: Telemetry, desired: Targets) -> Targets:
        return Targets(
            heading_deg=desired.heading_deg if desired.heading_deg is not None else telemetry.heading_deg,
            altitude_ft=desired.altitude_ft if desired.altitude_ft is not None else telemetry.altitude_ft,
            airspeed_kts=desired.airspeed_kts if desired.airspeed_kts is not None else telemetry.airspeed_kts,
            throttle=desired.throttle,
            brake_ratio=desired.brake_ratio,
            gear_down=desired.gear_down,
            flap_ratio=desired.flap_ratio,
            roll_limit=desired.roll_limit,
            pitch_limit=desired.pitch_limit,
            pitch_down_limit=desired.pitch_down_limit,
            yaw_hold=desired.yaw_hold,
            yaw_kp=desired.yaw_kp,
            yaw_limit=desired.yaw_limit,
        )


__all__ = ["Targets", "Guidance", "PassthroughGuidance"]
