from __future__ import annotations

import time

from uav.core.modes.base import Mode
from uav.sim.types import Telemetry, Targets


class Mode(Mode):
    name = "APPROACH"

    def enter(self, ctx: dict) -> None:
        state = ctx.setdefault("mode_state", {})
        state["approach_last_ts"] = time.time()
        state["approach_target_alt_ft"] = None

    def step(self, ctx: dict, telemetry: Telemetry) -> Targets:
        airframe = ctx.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        throttle_cfg = airframe.get("throttle", {})
        rates = airframe.get("rates_fpm", {})
        thresholds = airframe.get("thresholds", {})

        state = ctx.setdefault("mode_state", {})
        now = time.time()
        last_ts = state.get("approach_last_ts", now)
        dt = max(now - last_ts, 1e-3)
        state["approach_last_ts"] = now

        descent_fpm = abs(rates.get("descent", 400.0))
        min_alt = thresholds.get("land_transition_alt_ft", telemetry.altitude_ft)

        if state.get("approach_target_alt_ft") is None:
            state["approach_target_alt_ft"] = telemetry.altitude_ft

        next_alt = state["approach_target_alt_ft"] - (descent_fpm / 60.0) * dt
        if next_alt < min_alt:
            next_alt = min_alt
        state["approach_target_alt_ft"] = next_alt

        hold_hdg = ctx.get("mode_state", {}).get("hold_heading_deg", ctx["targets"]["target_hdg_deg"])
        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=state["approach_target_alt_ft"],
            airspeed_kts=speeds.get("v_approach", ctx["targets"]["target_airspeed_kts"]),
            climb_rate_fpm=-descent_fpm,
            throttle=throttle_cfg.get("approach", ctx["controller"]["cruise_throttle"] * 0.7),
            brake_ratio=0.0,
            gear_down=True,  # gear down for approach and landing
        )

    def exit(self, ctx: dict) -> None:
        state = ctx.setdefault("mode_state", {})
        state.pop("approach_last_ts", None)
        state.pop("approach_target_alt_ft", None)
