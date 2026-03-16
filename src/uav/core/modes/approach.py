from __future__ import annotations

import time

from uav.core.modes.base import Mode
from uav.sim.types import Telemetry, Targets


class Mode(Mode):
    name = "APPROACH"

    def enter(self, ctx: dict) -> None:
        state = ctx.setdefault("mode_state", {})
        # Record the MSL altitude we want to reach by end of approach.
        thresholds = ctx.get("airframe", {}).get("thresholds", {})
        land_alt = thresholds.get("land_transition_alt_ft", 200.0)
        ground_msl = ctx.get("takeoff_alt_ft", ctx["targets"].get("target_alt_ft", 1000.0) - 1500.0)
        state["approach_final_alt_ft"] = ground_msl + land_alt
        # Lock the approach heading at entry so we don't oscillate on final.
        # The bearing is computed once here; recomputing each tick causes oscillation.
        dest = ctx.get("destination")
        locked_hdg = state.get("hold_heading_deg", ctx["targets"].get("target_hdg_deg", 0.0))
        if dest:
            try:
                from uav.nav.geo import bearing_deg
                from uav.sim.types import Telemetry as _T  # noqa: F401
                # We don't have telemetry here, so heading is locked in step() on first call.
                state["approach_hdg_locked"] = False
            except Exception:
                pass
        state["approach_hdg_locked"] = False

    def step(self, ctx: dict, telemetry: Telemetry) -> Targets:
        airframe = ctx.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        throttle_cfg = airframe.get("throttle", {})
        state = ctx.setdefault("mode_state", {})

        final_alt = state.get("approach_final_alt_ft", telemetry.altitude_ft - 500.0)

        # Lock heading on first step — compute bearing once, then hold it.
        # Recomputing each tick causes oscillation as the plane converges on the runway.
        if not state.get("approach_hdg_locked", False):
            dest = ctx.get("destination")
            if dest and telemetry.has_position():
                try:
                    from uav.nav.geo import bearing_deg
                    state["approach_locked_hdg"] = bearing_deg(
                        telemetry.lat_deg, telemetry.lon_deg,
                        float(dest["lat"]), float(dest["lon"]),
                    )
                except Exception:
                    state["approach_locked_hdg"] = state.get("hold_heading_deg", ctx["targets"]["target_hdg_deg"])
            else:
                state["approach_locked_hdg"] = state.get("hold_heading_deg", ctx["targets"]["target_hdg_deg"])
            state["approach_hdg_locked"] = True

        hold_hdg = state["approach_locked_hdg"]

        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=final_alt,
            airspeed_kts=speeds.get("v_approach", ctx["targets"]["target_airspeed_kts"]),
            throttle=throttle_cfg.get("approach", ctx["controller"]["cruise_throttle"] * 0.7),
            pitch_limit=0.10,   # gentle nose-down only — no zoom climbs during descent
            brake_ratio=0.0,
            gear_down=True,
        )

    def exit(self, ctx: dict) -> None:
        state = ctx.setdefault("mode_state", {})
        state.pop("approach_last_ts", None)
        state.pop("approach_target_alt_ft", None)
