from __future__ import annotations

from uav.core.modes.base import Mode
from uav.sim.types import Telemetry, Targets


class Mode(Mode):
    name = "CRUISE"

    def enter(self, ctx: dict) -> None:
        pass

    def step(self, ctx: dict, telemetry: Telemetry) -> Targets:
        airframe = ctx.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        throttle_cfg = airframe.get("throttle", {})

        # If we have a destination, steer toward it.
        hold_hdg = ctx.get("mode_state", {}).get("hold_heading_deg", telemetry.heading_deg)
        dest = ctx.get("destination")
        if dest and hasattr(telemetry, "has_position") and telemetry.has_position():
            try:
                from uav.nav.geo import bearing_deg

                hold_hdg = bearing_deg(telemetry.lat_deg, telemetry.lon_deg, float(dest["lat"]), float(dest["lon"]))
            except Exception:
                pass

        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=ctx["targets"]["target_alt_ft"],
            airspeed_kts=speeds.get("v_cruise", ctx["targets"]["target_airspeed_kts"]),
            throttle=throttle_cfg.get("cruise", ctx["controller"]["cruise_throttle"]),
            brake_ratio=0.0,
        )

    def exit(self, ctx: dict) -> None:
        pass
