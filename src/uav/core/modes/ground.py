from __future__ import annotations

from uav.core.modes.base import Mode
from uav.sim.types import Telemetry, Targets


class Mode(Mode):
    name = "GROUND"

    def enter(self, ctx: dict) -> None:
        pass

    def step(self, ctx: dict, telemetry: Telemetry) -> Targets:
        return Targets(
            heading_deg=None,
            altitude_ft=None,
            airspeed_kts=None,
            throttle=0.0,
            brake_ratio=1.0,
        )

    def exit(self, ctx: dict) -> None:
        pass
