from __future__ import annotations

from .base import Guidance
from uav.sim.types import Telemetry, Targets


class SimpleGuidance(Guidance):
    def compute(self, telemetry: Telemetry, desired: Targets) -> Targets:
        heading = desired.heading_deg
        altitude = desired.altitude_ft
        airspeed = desired.airspeed_kts
        throttle = desired.throttle
        climb_fpm = desired.climb_rate_fpm

        if heading is None:
            heading = telemetry.heading_deg
        if altitude is None:
            altitude = telemetry.altitude_ft
        if airspeed is None:
            airspeed = telemetry.airspeed_kts

        return Targets(
            heading_deg=heading,
            altitude_ft=altitude,
            airspeed_kts=airspeed,
            climb_rate_fpm=climb_fpm,
            throttle=throttle,
        )
