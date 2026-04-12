"""
V2 Flight Engine — follow the ribbon.

Drop-in replacement for ReactiveFlightDirector.
Same interface: .name, .ctx, .step(telemetry, stale), .reset()

One job: at each tick, find where we are on the ribbon, look ahead,
and produce Targets that steer the controller toward the lookahead point.

Phase is a PROPERTY of the ribbon point, not a runtime decision.
No state machines, no phase transitions, no mode logic.
"""
from __future__ import annotations

import math
import time
from typing import Optional

from uav.nav.flight_plan_v2 import RibbonPath, PathPoint, plan_path, format_ribbon
from uav.nav.geo import haversine_m, bearing_deg
from uav.sim.types import Telemetry, Targets


def _wrap180(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


class FlightEngine:
    """Follow a pre-computed ribbon path.

    Interface contract (matches ReactiveFlightDirector / ModeManager):
      .name   → current phase label (string)
      .ctx    → shared context dict
      .step() → returns Targets
      .reset()→ called after a flight reset
    """

    # Lookahead distance: how far ahead on the ribbon to steer toward.
    # Shorter = tighter tracking, longer = smoother but lazier.
    _LOOKAHEAD_NM = 0.5       # cruise/climb
    _LOOKAHEAD_APPROACH = 0.3  # approach/descent — tighter tracking
    _LOOKAHEAD_GROUND = 0.15   # takeoff roll — very tight

    # Max heading advance per tick (same as ReactiveFlightDirector)
    _MAX_TURN_DEG_PER_STEP = 3.0

    def __init__(self, ctx: dict) -> None:
        self.ctx = ctx
        self._ribbon: Optional[RibbonPath] = None
        self._idx: int = 0          # current position on ribbon
        self._phase: str = "GROUND"
        self._built: bool = False   # has ribbon been built for this flight?
        self._st: dict = {}         # small persistent state

    # ── Public interface ─────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._phase

    def reset(self) -> None:
        self._ribbon = None
        self._idx = 0
        self._phase = "GROUND"
        self._built = False
        self._st.clear()
        self.ctx.pop("mode_state", None)
        self.ctx.pop("destination", None)

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
        takeoff_cfg = cfg.get("takeoff", {})
        throttle_cfg = airframe.get("throttle", {})
        mode_cfg = cfg.get("mode", {})

        # Sanity cap: v_approach must never exceed v_land + 15.
        # Bad calibration data can produce absurd approach speeds.
        v_land_cap = float(speeds.get("v_land", 77.0))
        v_app_raw = float(speeds.get("v_approach", 80.0))
        if v_app_raw > v_land_cap + 15.0:
            speeds = dict(speeds)
            speeds["v_approach"] = v_land_cap + 15.0

        # ── Auto-start guard ────────────────────────────────────────
        auto_start = bool(mode_cfg.get("auto_start", True))
        has_dest = cfg.get("destination") is not None
        demo = bool(mode_cfg.get("demo_sequence", False))
        if not auto_start or (not has_dest and not demo):
            self._phase = "GROUND"
            return self._ground_idle(telemetry)

        # ── Build ribbon on first tick with a destination ────────────
        if not self._built and has_dest:
            self._build_ribbon(telemetry)

        # ── No ribbon yet — fall back to V1-style ground hold ───────
        if self._ribbon is None:
            self._phase = "GROUND"
            return self._ground_hold(telemetry, takeoff_cfg, throttle_cfg, speeds)

        # ── Follow the ribbon ───────────────────────────────────────
        return self._follow(telemetry, takeoff_cfg, throttle_cfg, speeds)

    # ── Ribbon construction ──────────────────────────────────────────

    def _build_ribbon(self, telemetry: Telemetry) -> None:
        """Build the ribbon path from current position to destination."""
        cfg = self.ctx
        dest = cfg["destination"]
        airframe = cfg.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        rates = airframe.get("rates_fpm", {})

        cruise_alt = float(cfg.get("targets", {}).get("target_alt_ft", 5000.0))

        # Destination runway info
        dest_rwy = cfg.get("dest_runway")
        rwy_hdg = dest_rwy["heading"] if dest_rwy else None
        thr_lat = dest_rwy["threshold_lat"] if dest_rwy else None
        thr_lon = dest_rwy["threshold_lon"] if dest_rwy else None
        dest_alt = float(dest_rwy["elevation_ft"]) if dest_rwy and dest_rwy.get("elevation_ft") else telemetry.altitude_ft

        # Departure heading: prefer detected runway
        dep_hdg = telemetry.heading_deg
        if "runway_hdg" in self._st:
            dep_hdg = self._st["runway_hdg"]

        # Sanity cap v_approach to v_land + 15.
        # Bad calibration can produce absurd approach speeds.
        v_land_val = float(speeds.get("v_land", 77.0))
        v_approach_raw = float(speeds.get("v_approach", 83.0))
        v_approach_capped = min(v_approach_raw, v_land_val + 15.0)
        if v_approach_capped != v_approach_raw:
            print(f"[FLIGHT_ENGINE] Capping v_approach {v_approach_raw:.0f} → {v_approach_capped:.0f} kts")
            # Update ctx so later reads also get the capped value
            speeds["v_approach"] = v_approach_capped

        try:
            self._ribbon = plan_path(
                dep_lat=telemetry.lat_deg,
                dep_lon=telemetry.lon_deg,
                dep_alt_ft=telemetry.altitude_ft,
                dep_heading=dep_hdg,
                dest_lat=float(dest["lat"]),
                dest_lon=float(dest["lon"]),
                dest_alt_ft=dest_alt,
                dest_rwy_heading=rwy_hdg,
                dest_threshold_lat=thr_lat,
                dest_threshold_lon=thr_lon,
                dest_rwy_length_ft=float(dest_rwy.get("length_ft", 6000.0)) if dest_rwy else 6000.0,
                cruise_alt_ft=cruise_alt,
                v_rotate=float(speeds.get("v_rotate", 90.0)),
                v_climb=float(speeds.get("v_climb", 160.0)),
                v_cruise=float(speeds.get("v_cruise", 200.0)),
                v_approach=v_approach_capped,
                v_land=v_land_val,
                climb_fpm=float(rates.get("climb", 1600.0)),
                takeoff_roll_ft=float(airframe.get("takeoff_roll_ft", 2000.0)),
            )
            self._idx = 0
            self._built = True
            print(format_ribbon(self._ribbon))
        except Exception as e:
            print(f"[FLIGHT_ENGINE] Ribbon build failed: {e}")
            import traceback; traceback.print_exc()
            self._ribbon = None

    # ── Core: follow the ribbon ──────────────────────────────────────

    def _follow(self, telemetry: Telemetry, takeoff_cfg: dict,
                throttle_cfg: dict, speeds: dict) -> Targets:
        """Track the ribbon — the heart of V2."""
        ribbon = self._ribbon
        assert ribbon is not None

        # 1. Find nearest point on ribbon (monotonic forward search)
        self._idx = ribbon.nearest_ahead(
            telemetry.lat_deg, telemetry.lon_deg,
            telemetry.heading_deg, self._idx,
        )
        cur = ribbon.points[self._idx]
        self._phase = self._map_phase(cur.phase)

        # 2. Choose lookahead distance based on phase
        if cur.phase == "GROUND":
            la_nm = self._LOOKAHEAD_GROUND
        elif cur.phase in ("APPROACH", "DESCENT", "FLARE"):
            la_nm = self._LOOKAHEAD_APPROACH
        else:
            la_nm = self._LOOKAHEAD_NM

        la_idx = ribbon.lookahead(self._idx, la_nm)
        la = ribbon.points[la_idx]

        # 3. Compute bearing to lookahead point
        if telemetry.has_position():
            target_bearing = bearing_deg(
                telemetry.lat_deg, telemetry.lon_deg,
                la.lat, la.lon,
            )
        else:
            target_bearing = la.heading_deg

        # 4. Rate-limit heading changes (smooth turns)
        prev_cmd_hdg = self._st.get("cmd_hdg", telemetry.heading_deg)
        hdg_err = _wrap180(target_bearing - prev_cmd_hdg)
        advance = max(-self._MAX_TURN_DEG_PER_STEP,
                      min(self._MAX_TURN_DEG_PER_STEP, hdg_err))
        cmd_hdg = (prev_cmd_hdg + advance) % 360.0
        self._st["cmd_hdg"] = cmd_hdg

        # 5. Altitude target from ribbon point (with terrain floor protection)
        agl_ft = (telemetry.agl_m * 3.28084) if not math.isnan(telemetry.agl_m) else 0.0
        target_alt = la.alt_ft

        # Terrain floor: never go below 300ft AGL during cruise/descent
        if cur.phase in ("CRUISE", "DESCENT"):
            terrain_floor = (telemetry.altitude_ft - agl_ft) + 300.0
            target_alt = max(target_alt, terrain_floor)

        # 6. Speed target from ribbon
        target_speed = la.speed_kts

        # 7. Phase-specific overrides
        return self._phase_targets(
            telemetry, cur, la, cmd_hdg, target_alt, target_speed,
            agl_ft, takeoff_cfg, throttle_cfg, speeds,
        )

    def _phase_targets(
        self, telemetry: Telemetry,
        cur: PathPoint, la: PathPoint,
        cmd_hdg: float, target_alt: float, target_speed: float,
        agl_ft: float,
        takeoff_cfg: dict, throttle_cfg: dict, speeds: dict,
    ) -> Targets:
        """Pure ribbon follower — read the ribbon, set those targets.

        The ONLY reactive rules:
          1. Throttle: if ribbon says None → PID manages.
             If ribbon says 0.0 but we're below target speed → PID adds power.
          2. AGL safety: if on ground (< 3ft AGL) → brakes + idle regardless of ribbon.
          3. Ground roll: yaw hold for directional control on the runway.
        """
        phase = cur.phase
        on_ground = agl_ft < 3.0

        # ── AGL safety override — on the ground, always brake ────────
        if on_ground and phase in ("FLARE", "ROLLOUT", "APPROACH"):
            hold_hdg = self._st.get("approach_hdg_locked",
                        self._st.get("runway_hdg", telemetry.heading_deg))
            return Targets(
                heading_deg=hold_hdg,
                altitude_ft=telemetry.altitude_ft,
                airspeed_kts=0.0,
                throttle=0.0,
                brake_ratio=1.0,
                gear_down=True,
                flap_ratio=1.0,
                roll_limit=0.02,
            )

        # ── Throttle logic ───────────────────────────────────────────
        # Ribbon throttle: None = PID manages, 0.0 = idle, 1.0 = full
        # Reactive rule: if ribbon says idle (0.0) but we're BELOW target
        # speed by more than 5 kts, let PID add power to prevent stall.
        ribbon_throttle = cur.throttle
        if ribbon_throttle is not None and ribbon_throttle < 0.01:
            # Ribbon wants idle — but check if we need survival power
            if telemetry.airspeed_kts < (target_speed - 5.0):
                ribbon_throttle = None  # let PID save us

        # ── Ground roll: yaw hold ────────────────────────────────────
        yaw_hold = False
        yaw_kp = yaw_ki = yaw_limit_val = yaw_full = None
        if phase == "GROUND":
            if "runway_hdg" not in self._st:
                self._st["runway_hdg"] = telemetry.heading_deg
            cmd_hdg = self._st["runway_hdg"]
            yaw_hold = True
            yaw_kp = float(takeoff_cfg.get("yaw_kp", 0.02))
            yaw_ki = 0.0
            base_yaw_limit = float(takeoff_cfg.get("yaw_limit", 0.35))
            taper = 1.0 - min(max(telemetry.airspeed_kts, 0.0), 50.0) / 50.0
            yaw_limit_val = max(0.25, base_yaw_limit * (0.35 + 0.65 * taper))
            yaw_full = float(takeoff_cfg.get("yaw_full_deg", 5.0))
        elif phase == "CLIMB" and agl_ft < 600.0:
            yaw_hold = True
            yaw_kp = float(takeoff_cfg.get("yaw_kp", 0.07))
            yaw_ki = float(takeoff_cfg.get("yaw_ki", 0.02))
            yaw_limit_val = float(takeoff_cfg.get("yaw_limit", 0.5))
            yaw_full = float(takeoff_cfg.get("yaw_full_deg", 5.0))

        # ── Approach heading lock ────────────────────────────────────
        if phase in ("APPROACH", "FLARE", "ROLLOUT"):
            if not self._st.get("approach_hdg_locked"):
                self._st["approach_hdg_locked"] = True
                self._st["approach_hdg_locked"] = la.heading_deg
            cmd_hdg = self._st.get("approach_hdg_locked", cmd_hdg)

        # ── Roll limits by AGL ───────────────────────────────────────
        if phase == "GROUND":
            roll_lim = 0.08 if agl_ft > 10.0 else 0.0
        elif phase == "CLIMB" and agl_ft < 600.0:
            roll_lim = 0.35 * max(0.0, (agl_ft - 300.0) / 300.0)
        elif phase in ("FLARE", "ROLLOUT"):
            roll_lim = 0.03
        elif phase == "APPROACH":
            roll_lim = 0.15  # enough to correct heading on approach
        else:
            roll_lim = 0.35  # cruise/descent — full authority

        # ── Build targets from ribbon ────────────────────────────────
        return Targets(
            heading_deg=cmd_hdg,
            altitude_ft=target_alt,
            airspeed_kts=target_speed,
            throttle=ribbon_throttle,
            brake_ratio=0.0,
            gear_down=cur.gear_down,
            flap_ratio=cur.flap_ratio,
            roll_limit=roll_lim,
            pitch_limit=0.12 if phase in ("GROUND", "CLIMB", "CRUISE") else 0.08,
            yaw_hold=yaw_hold,
            yaw_kp=yaw_kp,
            yaw_ki=yaw_ki,
            yaw_limit=yaw_limit_val,
            yaw_full_deg=yaw_full,
        )

    # ── Phase mapping ────────────────────────────────────────────────

    def _map_phase(self, ribbon_phase: str) -> str:
        """Map ribbon phase names to the phase names autopilot.py expects.

        The autopilot watches for specific phase strings for scoring,
        DB logging, and phase-limit watchdog.
        """
        mapping = {
            "GROUND": "GROUND",
            "CLIMB": "CLIMB",
            "CRUISE": "CRUISE",
            "DESCENT": "CRUISE",    # watchdog treats descent like cruise
            "APPROACH": "APPROACH",
            "FLARE": "LAND",
            "ROLLOUT": "LAND",
        }
        return mapping.get(ribbon_phase, ribbon_phase)

    # ── Fallback modes ───────────────────────────────────────────────

    def _ground_idle(self, telemetry: Telemetry) -> Targets:
        return Targets(
            heading_deg=telemetry.heading_deg,
            altitude_ft=telemetry.altitude_ft,
            airspeed_kts=0.0,
            throttle=0.0,
            brake_ratio=1.0,
            gear_down=True,
        )

    def _ground_hold(self, telemetry: Telemetry, takeoff_cfg: dict,
                     throttle_cfg: dict, speeds: dict) -> Targets:
        """Hold on ground before ribbon is built — same as V1 GROUND."""
        if "runway_hdg" not in self._st:
            self._st["runway_hdg"] = telemetry.heading_deg
        return self._ground_idle(telemetry)
