from __future__ import annotations

import time

from uav.core.modes.base import Mode
from uav.sim.types import Telemetry, Targets


class Mode(Mode):
    name = "CLIMB"

    def enter(self, ctx: dict) -> None:
        state = ctx.setdefault("mode_state", {})
        state["climb_last_ts"] = time.time()
        state["climb_target_alt_ft"] = None
        state["climb_last_alt_ft"] = None

    def step(self, ctx: dict, telemetry: Telemetry) -> Targets:
        airframe = ctx.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        throttle_cfg = airframe.get("throttle", {})
        rates = airframe.get("rates_fpm", {})

        climb_cfg = ctx.get("climb", {})
        max_roll = climb_cfg.get("max_roll_cmd", 0.15)
        max_pitch = climb_cfg.get("max_pitch_cmd", 0.15)
        roll_freeze_alt = float(climb_cfg.get("roll_freeze_alt_ft", 0.0) or 0.0)
        if roll_freeze_alt > 0.0 and telemetry.altitude_ft < roll_freeze_alt:
            max_roll = min(max_roll, 0.02)

        state = ctx.setdefault("mode_state", {})
        now = time.time()
        last_ts = state.get("climb_last_ts", now)
        dt = max(now - last_ts, 1e-3)
        state["climb_last_ts"] = now

        target_alt = ctx["targets"]["target_alt_ft"]
        base_climb_fpm = rates.get("climb", 500.0)
        recover_fpm = climb_cfg.get("recover_fpm", base_climb_fpm * 1.8)

        last_alt = state.get("climb_last_alt_ft")
        if last_alt is None:
            last_alt = telemetry.altitude_ft
        vs_fpm = (telemetry.altitude_ft - last_alt) / dt * 60.0
        state["climb_last_alt_ft"] = telemetry.altitude_ft

        climb_fpm = base_climb_fpm
        if vs_fpm < 0.0:
            climb_fpm = max(base_climb_fpm, recover_fpm)

        if state.get("climb_target_alt_ft") is None:
            state["climb_target_alt_ft"] = telemetry.altitude_ft

        next_alt = state["climb_target_alt_ft"] + (climb_fpm / 60.0) * dt
        if next_alt > target_alt:
            next_alt = target_alt
        state["climb_target_alt_ft"] = next_alt

        # Use a lower climb speed by default to improve climb performance.
        climb_spd = speeds.get("v_climb", ctx["targets"].get("climb_airspeed_kts", ctx["targets"]["target_airspeed_kts"]))

        hold_hdg = ctx.get("mode_state", {}).get("hold_heading_deg", ctx["targets"]["target_hdg_deg"])
        # In climb, do not let throttle collapse: keep a minimum throttle to maintain climb.
        throttle_min = float(climb_cfg.get("throttle_min", throttle_cfg.get("climb", ctx["controller"]["cruise_throttle"])))
        # But also avoid runaway overspeed: if we're well above climb speed, pull throttle back.
        throttle_cap = float(climb_cfg.get("throttle_cap", 1.0))
        overspeed_kts = float(climb_cfg.get("overspeed_kts", 10.0))
        if telemetry.airspeed_kts > (climb_spd + overspeed_kts):
            throttle_cmd = min(throttle_min, 0.55)
        else:
            throttle_cmd = throttle_min

        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=state["climb_target_alt_ft"],
            airspeed_kts=climb_spd,
            climb_rate_fpm=climb_fpm,
            throttle=min(throttle_cap, throttle_cmd),
            brake_ratio=0.0,
            gear_down=False,  # retract gear once climb mode is active
            roll_limit=max_roll,
            pitch_limit=max_pitch,
            pitch_protect_kts=climb_spd - 10.0,
            pitch_protect_gain=climb_cfg.get("pitch_protect_gain", 0.03),
        )

    def exit(self, ctx: dict) -> None:
        state = ctx.setdefault("mode_state", {})
        state.pop("climb_last_ts", None)
        state.pop("climb_target_alt_ft", None)
