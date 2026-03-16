from __future__ import annotations

from uav.core.modes.base import Mode
from uav.sim.types import Telemetry, Targets


class Mode(Mode):
    name = "LAND"

    def enter(self, ctx: dict) -> None:
        pass

    def step(self, ctx: dict, telemetry: Telemetry) -> Targets:
        airframe = ctx.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        throttle_cfg = airframe.get("throttle", {})
        thresholds = airframe.get("thresholds", {})

        v_land = speeds.get("v_land", ctx["targets"]["target_airspeed_kts"])
        on_ground = telemetry.agl_m < 5.0
        brake = 1.0 if on_ground else 0.0

        hold_hdg = ctx.get("mode_state", {}).get("hold_heading_deg", ctx["targets"]["target_hdg_deg"])
        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=telemetry.altitude_ft,
            airspeed_kts=v_land,
            throttle=throttle_cfg.get("idle", 0.0),
            brake_ratio=brake,
            gear_down=True,
        )

    def exit(self, ctx: dict) -> None:
        pass
