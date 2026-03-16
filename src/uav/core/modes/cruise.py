from __future__ import annotations

from uav.core.modes.base import Mode
from uav.sim.types import Telemetry, Targets


def _wrap_deg(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


class Mode(Mode):
    name = "CRUISE"

    # At 15 Hz loop rate → 3°/step = 45°/s max turn rate.
    # Keeps heading error small → proportional bank, not full ±roll_limit slam.
    _MAX_TURN_DEG_PER_STEP = 3.0

    def enter(self, ctx: dict) -> None:
        pass

    def step(self, ctx: dict, telemetry: Telemetry) -> Targets:
        airframe = ctx.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        state = ctx.setdefault("mode_state", {})

        # Rate-limited heading toward destination bearing.
        dest = ctx.get("destination")
        if dest and hasattr(telemetry, "has_position") and telemetry.has_position():
            try:
                from uav.nav.geo import bearing_deg
                true_bearing = bearing_deg(
                    telemetry.lat_deg, telemetry.lon_deg,
                    float(dest["lat"]), float(dest["lon"]),
                )
                prev_cmd = state.get("cruise_cmd_hdg", telemetry.heading_deg)
                error = _wrap_deg(true_bearing - prev_cmd)
                advance = max(-self._MAX_TURN_DEG_PER_STEP, min(self._MAX_TURN_DEG_PER_STEP, error))
                cmd_hdg = (prev_cmd + advance) % 360.0
                state["cruise_cmd_hdg"] = cmd_hdg
                hold_hdg = cmd_hdg
            except Exception:
                hold_hdg = state.get("cruise_cmd_hdg", state.get("hold_heading_deg", telemetry.heading_deg))
        else:
            hold_hdg = state.get("cruise_cmd_hdg", state.get("hold_heading_deg", telemetry.heading_deg))

        v_stall = speeds.get("v_stall", 48.0)
        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=ctx["targets"]["target_alt_ft"],
            airspeed_kts=speeds.get("v_cruise", ctx["targets"]["target_airspeed_kts"]),
            throttle=None,              # let airspeed PID regulate throttle for speed stability
            pitch_limit=0.08,           # gentle altitude hold — prevents PID saturation/oscillation
            roll_limit=0.12,            # gentle bank — reduces altitude loss during turns
            pitch_protect_kts=v_stall + 20.0,
            pitch_protect_gain=0.03,
            brake_ratio=0.0,
        )

    def exit(self, ctx: dict) -> None:
        pass
