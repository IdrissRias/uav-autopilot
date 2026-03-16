from __future__ import annotations

from typing import Dict

from uav.sim.types import Telemetry, Targets
from uav.core.modes import Ground, Takeoff, Climb, Cruise, Approach, Land, Abort


class ModeManager:
    def __init__(self, start_mode: str, ctx: dict, auto_start: bool, has_destination: bool) -> None:
        self.ctx = ctx
        self.start_mode = start_mode.upper()
        self.auto_start = auto_start
        self.has_destination = has_destination
        self.modes: Dict[str, object] = {
            "GROUND": Ground(),
            "TAKEOFF": Takeoff(),
            "CLIMB": Climb(),
            "CRUISE": Cruise(),
            "APPROACH": Approach(),
            "LAND": Land(),
            "ABORT": Abort(),
        }
        self.current = self.modes["GROUND"]
        self.current.enter(self.ctx)
        self._started = False

    @property
    def name(self) -> str:
        return self.current.name

    def transition(self, mode_name: str) -> None:
        mode_name = mode_name.upper()
        if mode_name not in self.modes:
            return
        if self.current.name == mode_name:
            return
        self.current.exit(self.ctx)
        self.current = self.modes[mode_name]
        self.current.enter(self.ctx)

    def reset(self) -> None:
        # Bring the state machine back to ground and allow auto-start again.
        self._started = False
        self.ctx.pop("mode_state", None)
        self.ctx.pop("destination", None)  # recalculate from fresh position after reset
        self.transition("GROUND")

    def step(self, telemetry: Telemetry, stale: bool) -> Targets:
        if stale:
            self.transition("ABORT")
            return self.current.step(self.ctx, telemetry)

        mode_cfg = self.ctx.get("mode", {})
        demo_sequence = bool(mode_cfg.get("demo_sequence", False))
        lock_heading = bool(mode_cfg.get("lock_heading", False))
        if lock_heading:
            state = self.ctx.setdefault("mode_state", {})
            if state.get("hold_heading_deg") is None and telemetry.is_valid():
                state["hold_heading_deg"] = telemetry.heading_deg
        if (
            not self._started
            and telemetry.is_valid()
            and self.auto_start
            and (self.has_destination or demo_sequence)
        ):
            # Default: always start with TAKEOFF; other start modes can be added later.
            self.transition("TAKEOFF")
            self._started = True

        takeoff_cfg = self.ctx.get("takeoff", {})
        climb_cfg = self.ctx.get("climb", {})
        airframe = self.ctx.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        thresholds = airframe.get("thresholds", {})
        target_alt = self.ctx["targets"]["target_alt_ft"]
        vr_kts = takeoff_cfg.get(
            "vr_kts",
            thresholds.get("takeoff_to_climb_speed_kts", speeds.get("v_rotate", 55.0)),
        )
        alt_tol = climb_cfg.get(
            "altitude_tolerance_ft",
            thresholds.get("climb_to_cruise_altitude_tolerance_ft", 100.0),
        )
        landing_requested = bool(mode_cfg.get("landing_requested", False))
        land_alt_ft = thresholds.get("land_transition_alt_ft", 200.0)

        prev_mode = self.current.name

        if self.current.name == "TAKEOFF" and telemetry.airspeed_kts >= vr_kts:
            self.transition("CLIMB")

        if self.current.name == "CLIMB" and telemetry.altitude_ft >= target_alt - alt_tol:
            self.transition("CRUISE")

        # If we just entered CRUISE this tick, record the time and don't immediately jump to APPROACH.
        if prev_mode != "CRUISE" and self.current.name == "CRUISE":
            import time as _time
            self.ctx.setdefault("mode_state", {})["cruise_entered_at"] = _time.time()
            return self.current.step(self.ctx, telemetry)

        if self.current.name == "CRUISE":
            should_approach = landing_requested or demo_sequence
            if not should_approach:
                import time as _time
                cruise_entered_at = self.ctx.get("mode_state", {}).get("cruise_entered_at", 0.0)
                cruise_stable = (_time.time() - cruise_entered_at) >= 5.0
                dest = self.ctx.get("destination")
                if cruise_stable and dest and telemetry.has_position():
                    try:
                        from uav.nav.geo import haversine_m
                        dist_nm = haversine_m(
                            telemetry.lat_deg, telemetry.lon_deg,
                            float(dest["lat"]), float(dest["lon"]),
                        ) / 1852.0
                        agl_ft = telemetry.agl_m * 3.28084
                        approach_dist_nm = max(3.0, agl_ft / 300.0)
                        should_approach = dist_nm <= approach_dist_nm
                    except Exception:
                        pass
            if should_approach:
                self.transition("APPROACH")
        if self.current.name == "APPROACH" and telemetry.agl_m < 15.0:
            self.transition("LAND")

        return self.current.step(self.ctx, telemetry)
