from __future__ import annotations

from uav.core.modes.base import Mode
from uav.sim.types import Telemetry, Targets


class Mode(Mode):
    name = "ABORT"

    def enter(self, ctx: dict) -> None:
        pass

    def step(self, ctx: dict, telemetry: Telemetry) -> Targets:
        # Only slam brakes when we're basically stopped; otherwise it can create weird behavior.
        brake = 1.0 if telemetry.airspeed_kts < 5.0 else 0.0
        return Targets(
            heading_deg=None,
            altitude_ft=None,
            airspeed_kts=None,
            throttle=0.0,
            brake_ratio=brake,
        )

    def exit(self, ctx: dict) -> None:
        pass
