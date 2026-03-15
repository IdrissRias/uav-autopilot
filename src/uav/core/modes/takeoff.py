from __future__ import annotations

from uav.core.modes.base import Mode
from uav.sim.types import Telemetry, Targets


class Mode(Mode):
    name = "TAKEOFF"

    def enter(self, ctx: dict) -> None:
        ctx.setdefault("mode_state", {})
        ctx["mode_state"].pop("runway_heading_deg", None)

    def step(self, ctx: dict, telemetry: Telemetry) -> Targets:
        state = ctx.setdefault("mode_state", {})
        if "runway_heading_deg" not in state:
            state["runway_heading_deg"] = telemetry.heading_deg
        if state.get("hold_heading_deg") is None:
            state["hold_heading_deg"] = state["runway_heading_deg"]

        takeoff_cfg = ctx.get("takeoff", {})
        airframe = ctx.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        throttle_cfg = airframe.get("throttle", {})
        thresholds = airframe.get("thresholds", {})

        target_spd = takeoff_cfg.get(
            "target_airspeed_kts",
            speeds.get("v_takeoff", ctx["targets"]["target_airspeed_kts"]),
        )
        throttle = takeoff_cfg.get("throttle", throttle_cfg.get("takeoff", 1.0))
        vr_kts = speeds.get("v_rotate", thresholds.get("takeoff_to_climb_speed_kts", 55.0))
        rotate_gain_ft = thresholds.get("rotate_altitude_gain_ft", 200.0)

        target_alt = telemetry.altitude_ft
        if telemetry.airspeed_kts >= 0.8 * vr_kts:
            target_alt = min(telemetry.altitude_ft + rotate_gain_ft, ctx["targets"]["target_alt_ft"])

        max_roll_air = takeoff_cfg.get("max_roll_cmd", 0.08)
        max_roll = 0.0 if telemetry.airspeed_kts < vr_kts else max_roll_air
        max_pitch = takeoff_cfg.get("max_pitch_cmd", 0.12)

        # Yaw/steering scheduling:
        # - strong at very low speed
        # - taper down as we accelerate to avoid fishtail
        yaw_kp = takeoff_cfg.get("yaw_kp", 0.02)
        yaw_ki = takeoff_cfg.get("yaw_ki", 0.00)
        base_yaw_limit = float(takeoff_cfg.get("yaw_limit", 0.35))
        # taper from 1.0 -> 0.35 between 0..50 kts (floor at 0.25)
        taper = 1.0 - min(max(telemetry.airspeed_kts, 0.0), 50.0) / 50.0
        yaw_limit = max(0.25, base_yaw_limit * (0.35 + 0.65 * taper))
        # don't go full deflection unless big error
        yaw_full_deg = float(takeoff_cfg.get("yaw_full_deg", 5.0))

        return Targets(
            heading_deg=state.get("hold_heading_deg", state["runway_heading_deg"]),
            altitude_ft=target_alt,
            airspeed_kts=target_spd,
            climb_rate_fpm=takeoff_cfg.get("climb_rate_fpm"),
            throttle=throttle,
            brake_ratio=0.0,
            roll_limit=max_roll,
            pitch_limit=max_pitch,
            yaw_hold=True,
            yaw_kp=yaw_kp,
            yaw_ki=yaw_ki,
            yaw_limit=yaw_limit,
            yaw_full_deg=yaw_full_deg,
            pitch_protect_kts=target_spd - 8.0,
            pitch_protect_gain=0.02,
        )

    def exit(self, ctx: dict) -> None:
        pass
