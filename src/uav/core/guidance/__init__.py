from __future__ import annotations

from dataclasses import replace

from .base import Guidance
from uav.sim.types import Targets, Telemetry


class PassthroughGuidance(Guidance):
    """Fills in None setpoints from current telemetry. No policy.

    Uses dataclasses.replace so EVERY field of Targets passes through
    untouched. The previous version rebuilt the object by enumerating
    fields and silently dropped any field it didn't know about — that
    swallowed throttle_for_alt / throttle_base / vs_target_fpm for a
    whole day of flights (crash 20260706_131530: the controller never
    saw the coupling flags, dove after the steep slope in classic
    alt→pitch mode, and hit the ground at 225 kts).
    """

    def compute(self, telemetry: Telemetry, desired: Targets) -> Targets:
        return replace(
            desired,
            heading_deg=(desired.heading_deg
                         if desired.heading_deg is not None
                         else telemetry.heading_deg),
            altitude_ft=(desired.altitude_ft
                         if desired.altitude_ft is not None
                         else telemetry.altitude_ft),
            airspeed_kts=(desired.airspeed_kts
                          if desired.airspeed_kts is not None
                          else telemetry.airspeed_kts),
        )


__all__ = ["Targets", "Guidance", "PassthroughGuidance"]
