from __future__ import annotations

import math
import time
from typing import Optional

from uav.sim.types import Telemetry, Targets


def _wrap_deg(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


class ReactiveFlightDirector:
    """
    Situation-aware flight director.

    Replaces ModeManager + per-mode classes with a single step() that decides
    what to do based on current flight state every tick.  No named transitions,
    no missed windows, no "stuck-in-mode" bugs.

    Interface is drop-in compatible with ModeManager:
      .name   — current phase label (string)
      .ctx    — shared context dict
      .step() — returns Targets
      .reset()— called after a flight reset
    """

    _MAX_TURN_DEG_PER_STEP = 3.0  # heading advance per loop tick (45°/s at 15 Hz)

    def __init__(self, ctx: dict) -> None:
        self.ctx = ctx
        self._phase = "GROUND"
        self._st: dict = {}  # small persistent sub-state (heading ramps, timers)

    # ------------------------------------------------------------------
    # Public interface (matches ModeManager)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._phase

    def reset(self) -> None:
        """Return to ground state after a flight reset."""
        self._phase = "GROUND"
        self._st.clear()
        self.ctx.pop("mode_state", None)
        self.ctx.pop("destination", None)
        # approach_initiated lives in _st which was just cleared — nothing extra needed

    def step(self, telemetry: Telemetry, stale: bool = False) -> Targets:
        if stale:
            self._phase = "ABORT"
            return Targets(
                heading_deg=telemetry.heading_deg,
                altitude_ft=telemetry.altitude_ft,
                airspeed_kts=0.0,
                throttle=0.0,
                brake_ratio=1.0,
                gear_down=True,
            )

        cfg = self.ctx
        airframe = cfg.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        throttle_cfg = airframe.get("throttle", {})
        thresholds = airframe.get("thresholds", {})
        takeoff_cfg = cfg.get("takeoff", {})
        climb_cfg = cfg.get("climb", {})
        mode_cfg = cfg.get("mode", {})

        v_rotate = float(speeds.get("v_rotate", thresholds.get("takeoff_to_climb_speed_kts", 55.0)))
        v_climb = float(speeds.get("v_climb", 75.0))
        v_cruise = float(speeds.get("v_cruise", 105.0))
        v_approach = float(speeds.get("v_approach", 70.0))
        v_land = float(speeds.get("v_land", 60.0))
        v_stall = float(speeds.get("v_stall", 48.0))

        cruise_target_ft = float(cfg["targets"]["target_alt_ft"])
        agl_ft = (telemetry.agl_m * 3.28084) if not math.isnan(telemetry.agl_m) else 0.0

        # ── Auto-start guard ──────────────────────────────────────────────────
        # Don't begin the flight sequence unless requested.
        auto_start = bool(mode_cfg.get("auto_start", True))
        has_dest = cfg.get("destination") is not None
        demo = bool(mode_cfg.get("demo_sequence", False))
        if not auto_start or (not has_dest and not demo):
            self._phase = "GROUND"
            return self._ground_idle(telemetry)

        # ── Destination geometry ──────────────────────────────────────────────
        dist_nm: float = math.inf
        true_bearing: Optional[float] = None
        dest = cfg.get("destination")
        if dest and telemetry.has_position():
            try:
                from uav.nav.geo import haversine_m, bearing_deg
                dist_nm = haversine_m(
                    telemetry.lat_deg, telemetry.lon_deg,
                    float(dest["lat"]), float(dest["lon"]),
                ) / 1852.0
                true_bearing = bearing_deg(
                    telemetry.lat_deg, telemetry.lon_deg,
                    float(dest["lat"]), float(dest["lon"]),
                )
            except Exception:
                pass

        # Dynamic approach trigger: start descending earlier the higher we are.
        # Trigger distance is computed once and latched — never shrinks mid-descent,
        # which would cause near_dest to flip False and re-trigger CLIMB.
        approach_dist_nm = max(3.0, agl_ft / 300.0)
        if not self._st.get("approach_initiated", False) and dist_nm <= approach_dist_nm:
            self._st["approach_initiated"] = True
        near_dest = self._st.get("approach_initiated", False)

        # ── Phase selection ───────────────────────────────────────────────────
        # Pure function of current flight state — re-evaluated every tick.
        on_ground = agl_ft < 5.0
        at_cruise = telemetry.altitude_ft >= cruise_target_ft - 50.0
        flaring = agl_ft < 15.0 and near_dest

        if flaring:
            phase = "LAND"
        elif on_ground:
            phase = "GROUND"
        elif not at_cruise and not near_dest:
            phase = "CLIMB"
        elif near_dest:
            phase = "APPROACH"
        else:
            phase = "CRUISE"

        # Capture runway heading on first ground tick so CLIMB and APPROACH can use it.
        if on_ground and "runway_hdg" not in self._st:
            self._st["runway_hdg"] = telemetry.heading_deg

        # Clear approach heading lock whenever we leave APPROACH (e.g. go-around).
        prev_phase = self._phase
        self._phase = phase
        if prev_phase == "APPROACH" and phase not in ("APPROACH", "LAND"):
            self._st.pop("approach_hdg_locked", None)
            self._st.pop("approach_locked_hdg", None)
            self._st.pop("approach_initiated", None)

        # ── Dispatch to phase implementation ─────────────────────────────────
        if phase == "GROUND":
            return self._ground(
                telemetry, v_rotate, v_climb, cruise_target_ft,
                takeoff_cfg, throttle_cfg, thresholds,
            )
        if phase == "CLIMB":
            return self._climb(
                telemetry, v_climb, v_stall, cruise_target_ft,
                climb_cfg, throttle_cfg,
            )
        if phase == "CRUISE":
            return self._cruise(telemetry, v_cruise, v_stall, cruise_target_ft, true_bearing, agl_ft)
        if phase == "APPROACH":
            return self._approach(
                telemetry, v_approach, v_stall, agl_ft, dist_nm, true_bearing, throttle_cfg,
            )
        # LAND
        return self._land(telemetry, v_land, agl_ft, throttle_cfg)

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _ground_idle(self, telemetry: Telemetry) -> Targets:
        """Hold on ground, engines idle — used when auto-start is disabled."""
        return Targets(
            heading_deg=telemetry.heading_deg,
            altitude_ft=telemetry.altitude_ft,
            airspeed_kts=0.0,
            throttle=0.0,
            brake_ratio=1.0,
            gear_down=True,
        )

    def _ground(
        self,
        telemetry: Telemetry,
        v_rotate: float,
        v_climb: float,
        cruise_target_ft: float,
        takeoff_cfg: dict,
        throttle_cfg: dict,
        thresholds: dict,
    ) -> Targets:
        """
        Takeoff roll and rotation.

        Holds runway heading with rudder; zero aileron until past Vr;
        pitches to a target altitude once at 80% of Vr.
        """
        if "runway_hdg" not in self._st:
            self._st["runway_hdg"] = telemetry.heading_deg
        runway_hdg = self._st["runway_hdg"]

        throttle = float(takeoff_cfg.get("throttle", throttle_cfg.get("takeoff", 1.0)))
        rotate_gain_ft = float(thresholds.get("rotate_altitude_gain_ft", 200.0))

        if telemetry.airspeed_kts >= 0.8 * v_rotate:
            target_alt = min(telemetry.altitude_ft + rotate_gain_ft, cruise_target_ft)
        else:
            target_alt = telemetry.altitude_ft

        max_roll = 0.0 if telemetry.airspeed_kts < v_rotate else float(takeoff_cfg.get("max_roll_cmd", 0.08))
        max_pitch = float(takeoff_cfg.get("max_pitch_cmd", 0.12))

        # Yaw authority tapers from full at 0 kts down to 35% at 50 kts.
        base_yaw_limit = float(takeoff_cfg.get("yaw_limit", 0.35))
        taper = 1.0 - min(max(telemetry.airspeed_kts, 0.0), 50.0) / 50.0
        yaw_limit = max(0.25, base_yaw_limit * (0.35 + 0.65 * taper))

        return Targets(
            heading_deg=runway_hdg,
            altitude_ft=target_alt,
            airspeed_kts=v_climb,
            throttle=throttle,
            brake_ratio=0.0,
            gear_down=True,
            roll_limit=max_roll,
            pitch_limit=max_pitch,
            yaw_hold=True,
            yaw_kp=float(takeoff_cfg.get("yaw_kp", 0.02)),
            yaw_ki=float(takeoff_cfg.get("yaw_ki", 0.00)),
            yaw_limit=yaw_limit,
            yaw_full_deg=float(takeoff_cfg.get("yaw_full_deg", 5.0)),
            pitch_protect_kts=v_climb - 8.0,
            pitch_protect_gain=0.02,
        )

    def _climb(
        self,
        telemetry: Telemetry,
        v_climb: float,
        v_stall: float,
        cruise_target_ft: float,
        climb_cfg: dict,
        throttle_cfg: dict,
    ) -> Targets:
        """
        Climb to cruise altitude.

        Ramps the target altitude at a fixed fpm so the altitude PID always
        sees a small, proportional error rather than a 5000-ft cliff.
        """
        now = time.time()
        rates = self.ctx.get("airframe", {}).get("rates_fpm", {})
        base_fpm = float(rates.get("climb", 500.0))
        recover_fpm = float(climb_cfg.get("recover_fpm", base_fpm * 1.8))

        last_ts = self._st.get("climb_ts", now)
        dt = max(now - last_ts, 1e-3)
        self._st["climb_ts"] = now

        # Measure actual vertical speed to decide whether to use recover rate.
        last_alt = self._st.get("climb_alt", telemetry.altitude_ft)
        vs_fpm = (telemetry.altitude_ft - last_alt) / dt * 60.0
        self._st["climb_alt"] = telemetry.altitude_ft

        climb_fpm = recover_fpm if vs_fpm < 0.0 else base_fpm

        prev_tgt = self._st.get("climb_tgt_alt", telemetry.altitude_ft)
        next_tgt = min(prev_tgt + (climb_fpm / 60.0) * dt, cruise_target_ft)
        self._st["climb_tgt_alt"] = next_tgt

        # Hold runway heading during climb; don't start the destination turn yet.
        hold_hdg = self._st.get("runway_hdg", self.ctx["targets"]["target_hdg_deg"])

        agl_m = telemetry.agl_m if not math.isnan(telemetry.agl_m) else 0.0
        gear_down = agl_m <= 15.0

        throttle_min = float(climb_cfg.get("throttle_min", throttle_cfg.get("climb", 0.85)))
        overspeed_kts = float(climb_cfg.get("overspeed_kts", 10.0))
        throttle_cmd = min(throttle_min, 0.55) if telemetry.airspeed_kts > (v_climb + overspeed_kts) else throttle_min

        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=next_tgt,
            airspeed_kts=v_climb,
            climb_rate_fpm=climb_fpm,
            throttle=min(float(climb_cfg.get("throttle_cap", 1.0)), throttle_cmd),
            brake_ratio=0.0,
            gear_down=gear_down,
            roll_limit=float(climb_cfg.get("max_roll_cmd", 0.15)),
            pitch_limit=float(climb_cfg.get("max_pitch_cmd", 0.15)),
            pitch_protect_kts=v_climb - 10.0,
            pitch_protect_gain=float(climb_cfg.get("pitch_protect_gain", 0.03)),
        )

    def _cruise(
        self,
        telemetry: Telemetry,
        v_cruise: float,
        v_stall: float,
        cruise_target_ft: float,
        true_bearing: Optional[float],
        agl_ft: float,
    ) -> Targets:
        """
        Cruise at altitude toward destination.

        Heading is advanced at most _MAX_TURN_DEG_PER_STEP per tick so the
        heading PID always sees a small error and rolls stay proportional.

        Terrain avoidance: if AGL drops below 300ft during cruise (rising terrain),
        the altitude target is raised to maintain at least 300ft AGL clearance.
        """
        if true_bearing is not None:
            prev_cmd = self._st.get("cruise_cmd_hdg", telemetry.heading_deg)
            error = _wrap_deg(true_bearing - prev_cmd)
            advance = max(-self._MAX_TURN_DEG_PER_STEP, min(self._MAX_TURN_DEG_PER_STEP, error))
            cmd_hdg = (prev_cmd + advance) % 360.0
            self._st["cruise_cmd_hdg"] = cmd_hdg
        else:
            cmd_hdg = self._st.get("cruise_cmd_hdg", telemetry.heading_deg)

        # Terrain floor: if rising terrain brings AGL below 300ft, push target up.
        terrain_floor_ft = (telemetry.altitude_ft - agl_ft) + 300.0
        effective_target = max(cruise_target_ft, terrain_floor_ft)

        # Pre-approach speed bleed: ramp airspeed target from v_cruise down to v_approach
        # over the last 8nm so the plane arrives at approach already slow.
        approach_dist_nm = max(3.0, agl_ft / 300.0)
        bleed_start_nm = approach_dist_nm + 8.0
        dest = self.ctx.get("destination")
        v_approach = float(self.ctx.get("airframe", {}).get("speeds_kts", {}).get("v_approach", 70.0))
        if dest is not None:
            try:
                from uav.nav.geo import haversine_m, bearing_deg as _bd
                dist_to_dest = haversine_m(
                    telemetry.lat_deg, telemetry.lon_deg,
                    float(dest["lat"]), float(dest["lon"]),
                ) / 1852.0
                if dist_to_dest < bleed_start_nm:
                    t = max(0.0, min(1.0, (bleed_start_nm - dist_to_dest) / 8.0))
                    cmd_spd = v_cruise - t * (v_cruise - v_approach)
                else:
                    cmd_spd = v_cruise
            except Exception:
                cmd_spd = v_cruise
        else:
            cmd_spd = v_cruise

        return Targets(
            heading_deg=cmd_hdg,
            altitude_ft=effective_target,
            airspeed_kts=cmd_spd,
            throttle=None,       # airspeed PID regulates throttle for speed stability
            brake_ratio=0.0,
            gear_down=False,
            pitch_limit=0.08,    # prevents PID saturation during altitude hold
            roll_limit=0.12,     # gentle bank → small altitude loss during turns
            pitch_protect_kts=v_stall + 20.0,
            pitch_protect_gain=0.03,
        )

    def _approach(
        self,
        telemetry: Telemetry,
        v_approach: float,
        v_stall: float,
        agl_ft: float,
        dist_nm: float,
        true_bearing: Optional[float],
        throttle_cfg: dict,
    ) -> Targets:
        """
        Descend and line up with the runway.

        Approach heading is locked on first tick — recomputing each tick
        causes heading oscillation as the aircraft converges on the runway.

        Altitude target is a 3° glidepath anchored to the terrain currently
        under the aircraft: terrain_msl + dist_nm * 318ft + 50ft flare height.
        This works regardless of departure/destination elevation difference, and
        prevents the old bug where the target was computed from takeoff_alt_ft
        (departure MSL) and ended up underground at a higher-elevation destination.
        """
        if not self._st.get("approach_hdg_locked", False):
            if true_bearing is not None:
                locked = true_bearing
            else:
                locked = self._st.get(
                    "cruise_cmd_hdg",
                    self.ctx["targets"]["target_hdg_deg"],
                )
            self._st["approach_locked_hdg"] = locked
            self._st["approach_hdg_locked"] = True

        hold_hdg = self._st["approach_locked_hdg"]

        # Terrain MSL estimate under the aircraft right now.
        # As the plane flies toward the destination on final, this converges
        # toward destination airport elevation.
        terrain_msl_ft = telemetry.altitude_ft - agl_ft

        # 3° glidepath: 318ft of descent per NM.  Arrive at 50ft AGL at threshold.
        glidepath_ft = terrain_msl_ft + max(0.0, dist_nm) * 318.0 + 50.0

        # Only command descent — never zoom-climb up to a glidepath that's above us.
        target_alt_ft = min(telemetry.altitude_ft, glidepath_ft)

        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=target_alt_ft,
            airspeed_kts=v_approach,
            throttle=None,       # airspeed PID manages throttle — prevents runaway speed on descent
            brake_ratio=0.0,
            gear_down=True,
            pitch_limit=0.12,
            pitch_protect_kts=v_stall + 10.0,
            pitch_protect_gain=0.03,
        )

    def _land(
        self,
        telemetry: Telemetry,
        v_land: float,
        agl_ft: float,
        throttle_cfg: dict,
    ) -> Targets:
        """
        Flare and rollout.

        Idle throttle; hold the locked approach heading; apply brakes on touchdown.

        Flare: target an altitude 20ft above current terrain so the altitude PID
        commands a gentle nose-up that arrests the descent rate before touchdown.
        Without this the plane hits the ground at full approach sink rate → hard landing.
        """
        hold_hdg = self._st.get(
            "approach_locked_hdg",
            self._st.get("runway_hdg", telemetry.heading_deg),
        )
        on_ground = agl_ft < 5.0

        # Terrain MSL estimate under us right now.
        terrain_msl_ft = telemetry.altitude_ft - agl_ft

        # Target 20ft AGL — altitude PID pitches nose up to reach it, arresting descent.
        # Once on the ground (agl < 5), hold current alt so PID relaxes.
        flare_target_ft = terrain_msl_ft + 20.0 if not on_ground else telemetry.altitude_ft

        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=flare_target_ft,
            airspeed_kts=v_land,
            throttle=float(throttle_cfg.get("idle", 0.0)),
            brake_ratio=1.0 if on_ground else 0.0,
            gear_down=True,
            pitch_limit=0.10,   # caps nose-up so we don't balloon
        )
