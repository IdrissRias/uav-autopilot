"""
V2 Flight Engine — pure ribbon follower.

One rule: every tick, read the ribbon point at the current cursor, and
command its targets verbatim. No phase logic, no smoothness, no soft
limits. All smoothness, pacing, and smarts come from the ribbon itself.

The engine has ONLY two non-negotiable physical exceptions:
  1. Wheels-on-ground braking during landing rollout (physics)
  2. Yaw hold on the takeoff roll (rudder coordination while rolling)

Everything else — roll limits, pitch limits, terrain floor, heading
rate-limit, AGL gating — is now the ribbon's responsibility.

Interface contract (matches ReactiveFlightDirector / ModeManager):
  .name   → current phase label (string)
  .ctx    → shared context dict
  .step() → returns Targets
  .reset()→ called after a flight reset
"""
from __future__ import annotations

import math
from typing import Optional

from uav.nav.flight_plan_v2 import RibbonPath, PathPoint, plan_path, format_ribbon
from uav.nav.geo import bearing_deg
from uav.sim.types import Telemetry, Targets


class FlightEngine:
    # Fixed steering lookahead — use the bearing to a point this far along
    # the ribbon as the commanded heading. This is what makes the aircraft
    # track the line, rather than perpetually flying toward "the current point"
    # (which is by definition wherever the plane already is).
    _LOOKAHEAD_NM = 0.3

    def __init__(self, ctx: dict) -> None:
        self.ctx = ctx
        self._ribbon: Optional[RibbonPath] = None
        self._idx: int = 0
        self._phase: str = "GROUND"
        self._built: bool = False
        self._st: dict = {}

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
        mode_cfg = cfg.get("mode", {})

        # Auto-start guard
        auto_start = bool(mode_cfg.get("auto_start", True))
        has_dest = cfg.get("destination") is not None
        demo = bool(mode_cfg.get("demo_sequence", False))
        if not auto_start or (not has_dest and not demo):
            self._phase = "GROUND"
            return self._ground_idle(telemetry)

        # Build ribbon once on first tick with a destination
        if not self._built and has_dest:
            self._build_ribbon(telemetry)

        if self._ribbon is None:
            self._phase = "GROUND"
            return self._ground_idle(telemetry)

        return self._follow(telemetry)

    # ── Ribbon construction ──────────────────────────────────────────

    def _build_ribbon(self, telemetry: Telemetry) -> None:
        cfg = self.ctx
        dest = cfg["destination"]
        airframe = cfg.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        rates = airframe.get("rates_fpm", {})

        cruise_alt = float(cfg.get("targets", {}).get("target_alt_ft", 5000.0))

        dest_rwy = cfg.get("dest_runway")
        rwy_hdg = dest_rwy["heading"] if dest_rwy else None
        thr_lat = dest_rwy["threshold_lat"] if dest_rwy else None
        thr_lon = dest_rwy["threshold_lon"] if dest_rwy else None
        dest_alt = float(dest_rwy["elevation_ft"]) if dest_rwy and dest_rwy.get("elevation_ft") else telemetry.altitude_ft

        dep_hdg = telemetry.heading_deg
        if "runway_hdg" in self._st:
            dep_hdg = self._st["runway_hdg"]

        # Sanity cap v_approach — bad calibration can produce absurd speeds.
        v_land_val = float(speeds.get("v_land", 77.0))
        v_approach_raw = float(speeds.get("v_approach", 83.0))
        v_approach_capped = min(v_approach_raw, v_land_val + 15.0)
        if v_approach_capped != v_approach_raw:
            print(f"[FLIGHT_ENGINE] Capping v_approach {v_approach_raw:.0f} → {v_approach_capped:.0f} kts")
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

    # ── Core: pure follower ──────────────────────────────────────────

    def _follow(self, telemetry: Telemetry) -> Targets:
        """Read ribbon, emit its targets verbatim.

        Two exceptions:
          (a) On-ground during landing  → brakes locked, throttle idle.
          (b) Takeoff roll               → yaw hold on departure heading.
        """
        r = self._ribbon
        assert r is not None

        # 1. Advance cursor
        self._idx = r.nearest_ahead(
            telemetry.lat_deg, telemetry.lon_deg,
            telemetry.heading_deg, self._idx,
        )
        p = r.points[self._idx]
        self._phase = self._map_phase(p.phase)

        # 2. Heading lookahead — steer toward a point ahead on the ribbon
        la_idx = r.lookahead(self._idx, self._LOOKAHEAD_NM)
        la = r.points[la_idx]
        if telemetry.has_position():
            target_heading = bearing_deg(
                telemetry.lat_deg, telemetry.lon_deg, la.lat, la.lon,
            )
        else:
            target_heading = la.heading_deg

        # 3. AGL (for on-ground override only)
        agl_ft = (telemetry.agl_m * 3.28084) if not math.isnan(telemetry.agl_m) else 0.0

        # ── Exception (a): wheels on ground during landing ──────────
        if agl_ft < 3.0 and p.phase in ("ROLLOUT", "FLARE"):
            return Targets(
                heading_deg=p.heading_deg,
                altitude_ft=telemetry.altitude_ft,
                airspeed_kts=0.0,
                throttle=0.0,
                brake_ratio=1.0,
                gear_down=True,
                flap_ratio=1.0,
                roll_limit=0.02,
            )

        # ── Exception (b): takeoff roll yaw hold ─────────────────────
        yaw_hold = False
        yaw_kp = yaw_ki = yaw_limit_val = yaw_full = None
        if p.phase == "GROUND":
            if "runway_hdg" not in self._st:
                self._st["runway_hdg"] = telemetry.heading_deg
            target_heading = self._st["runway_hdg"]
            takeoff_cfg = self.ctx.get("takeoff", {})
            yaw_hold = True
            yaw_kp = float(takeoff_cfg.get("yaw_kp", 0.02))
            yaw_ki = 0.0
            yaw_limit_val = float(takeoff_cfg.get("yaw_limit", 0.35))
            yaw_full = float(takeoff_cfg.get("yaw_full_deg", 5.0))

        # ── Pure follower: emit the ribbon point verbatim ────────────
        return Targets(
            heading_deg=target_heading,
            altitude_ft=p.alt_ft,
            airspeed_kts=p.speed_kts,
            throttle=p.throttle,
            brake_ratio=0.0,
            gear_down=p.gear_down,
            flap_ratio=p.flap_ratio,
            roll_limit=0.5,      # full authority — ribbon owns smoothness
            pitch_limit=0.15,    # full authority
            yaw_hold=yaw_hold,
            yaw_kp=yaw_kp,
            yaw_ki=yaw_ki,
            yaw_limit=yaw_limit_val,
            yaw_full_deg=yaw_full,
        )

    # ── Phase mapping ────────────────────────────────────────────────

    def _map_phase(self, ribbon_phase: str) -> str:
        """Map ribbon phase names to what autopilot.py expects for scoring
        and watchdog logic."""
        mapping = {
            "GROUND": "GROUND",
            "CLIMB": "CLIMB",
            "CRUISE": "CRUISE",
            "DESCENT": "CRUISE",
            "APPROACH": "APPROACH",
            "FLARE": "LAND",
            "ROLLOUT": "LAND",
        }
        return mapping.get(ribbon_phase, ribbon_phase)

    # ── Idle fallback ────────────────────────────────────────────────

    def _ground_idle(self, telemetry: Telemetry) -> Targets:
        return Targets(
            heading_deg=telemetry.heading_deg,
            altitude_ft=telemetry.altitude_ft,
            airspeed_kts=0.0,
            throttle=0.0,
            brake_ratio=1.0,
            gear_down=True,
        )
