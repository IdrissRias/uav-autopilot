"""
V2 Flight Engine — keyframe walker.

The ribbon is a list of ~12 keyframes.  Each keyframe has a stable
command block and a trigger that advances to the next frame.  The engine
does exactly three things every tick:

  1. Evaluate the current keyframe's trigger.  If it fired, advance.
  2. Resolve the current keyframe into a concrete Targets object
     (handling dynamic commands: glideslope altitude, aim-at heading,
     altitude-scaled cruise throttle).
  3. Emit it.

No policy, no smoothness tricks, no soft limits.  If the ribbon commands
a 90° bank and 2000 kts at 500 ft — the ribbon is wrong, not the engine.

Interface contract (consumed by Autopilot as `mode_manager`):
  .name   → current phase label (string)
  .ctx    → shared context dict
  .step() → returns Targets
  .reset()→ called after a flight reset
"""
from __future__ import annotations

import math
from typing import Optional

from uav.nav.flight_plan_v2 import (
    Keyframe, Ribbon, plan_path, format_ribbon,
)
from uav.nav.geo import bearing_deg, haversine_m
from uav.sim.types import Telemetry, Targets
from uav.core.guidance.track_follower import TrackFollower, TrackState


class FlightEngine:
    def __init__(self, ctx: dict) -> None:
        self.ctx = ctx
        self._ribbon: Optional[Ribbon] = None
        self._idx: int = 0
        self._phase: str = "GROUND"
        self._built: bool = False
        self._prev_targets: Optional[Targets] = None
        # L1 track follower. Rebuilt each time a new ribbon is installed.
        # All aim_at keyframes route heading through this follower so the
        # plane tracks the ribbon LINE, not individual aim points.
        self._follower: Optional[TrackFollower] = None
        self._last_track: Optional[TrackState] = None

    # ── Public interface ─────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Mapped phase label used by scoring + heartbeat (legacy contract)."""
        return self._phase

    @property
    def keyframe_name(self) -> str:
        """Real keyframe name (e.g. CLIMB_GEAR_UP).  For diagnostic prints."""
        if self._ribbon is None:
            return self._phase
        if self._idx >= len(self._ribbon.keyframes):
            return self._phase
        return self._ribbon.keyframes[self._idx].name

    def reset(self) -> None:
        self._ribbon = None
        self._idx = 0
        self._phase = "GROUND"
        self._built = False
        self._prev_targets = None
        self._follower = None
        self._last_track = None
        self.ctx.pop("mode_state", None)
        self.ctx.pop("destination", None)
        self.ctx.pop("_aim_passed_kf", None)
        self.ctx.pop("track_state", None)

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

        mode_cfg = self.ctx.get("mode", {})
        auto_start = bool(mode_cfg.get("auto_start", True))
        has_dest = self.ctx.get("destination") is not None
        demo = bool(mode_cfg.get("demo_sequence", False))
        if not auto_start or (not has_dest and not demo):
            self._phase = "GROUND"
            return self._ground_idle(telemetry)

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

        cruise_alt = float(cfg.get("targets", {}).get("target_alt_ft", 5000.0))

        dest_rwy = cfg.get("dest_runway")
        rwy_hdg = dest_rwy["heading"] if dest_rwy else None
        thr_lat = dest_rwy["threshold_lat"] if dest_rwy else None
        thr_lon = dest_rwy["threshold_lon"] if dest_rwy else None
        dest_alt = (float(dest_rwy["elevation_ft"])
                    if dest_rwy and dest_rwy.get("elevation_ft")
                    else telemetry.altitude_ft)

        dep_hdg = telemetry.heading_deg

        # Ribbon's Vs = landing reference.  Prefer v_land (the speed we
        # actually fly at touchdown) over the theoretical v_stall, so the
        # ratios (Vrotate=1.17×, Vapp=1.08×…) produce usable numbers.
        # Default 77 kts (Cirrus SF50 approach reference).
        v_stall = float(speeds.get("v_land", 77.0))

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
                cruise_alt_ft=cruise_alt,
                v_stall=v_stall,
            )
            self._idx = 0
            self._built = True
            self._install_follower()
            print(format_ribbon(self._ribbon))
        except Exception as e:
            print(f"[FLIGHT_ENGINE] Ribbon build failed: {e}")
            import traceback; traceback.print_exc()
            self._ribbon = None
            self._follower = None

    def _install_follower(self) -> None:
        """Wrap the current ribbon's polyline in a TrackFollower. Called
        after ribbon build (here) and also from autopilot when the flight
        plan is pre-built externally (we detect that case by seeing a
        ribbon set on self._ribbon but no follower yet)."""
        if self._ribbon is None or not self._ribbon.points:
            self._follower = None
            return
        try:
            self._follower = TrackFollower(self._ribbon.points)
            print(f"[FOLLOWER] L1 track follower armed — "
                  f"{len(self._ribbon.points)} polyline points, "
                  f"{self._follower.cum_lengths[-1]:.2f} nm total")
        except Exception as e:
            print(f"[FOLLOWER] Failed to arm: {e}")
            self._follower = None

    # ── Core: keyframe walker ────────────────────────────────────────

    def _follow(self, telemetry: Telemetry) -> Targets:
        r = self._ribbon
        assert r is not None

        # Lazy-arm the follower if the ribbon was pre-built externally
        # (autopilot._build_and_broadcast_plan assigns self._ribbon
        # directly without going through _build_ribbon).
        if self._follower is None:
            self._install_follower()

        agl_ft = ((telemetry.agl_m * 3.28084)
                  if not math.isnan(telemetry.agl_m) else 0.0)

        # Advance keyframe if trigger fired.  Loop so multiple triggers can
        # cascade (e.g. first tick of CRUISE when we already overshot
        # descent_start).
        while self._idx < len(r.keyframes) - 1:
            kf = r.keyframes[self._idx]
            if kf.trigger.fired(telemetry, agl_ft):
                self._idx += 1
                new_kf = r.keyframes[self._idx]
                print(f"[RIBBON] advance → {new_kf.name} [{new_kf.phase}]  "
                      f"(agl={agl_ft:.0f}ft spd={telemetry.airspeed_kts:.0f}kts "
                      f"alt={telemetry.altitude_ft:.0f}ft)")
            else:
                break

        kf = r.keyframes[self._idx]
        self._phase = self._map_phase(kf.phase)

        targets = self._resolve(kf, telemetry, r)
        self._prev_targets = targets
        return targets

    def _resolve(self, kf: Keyframe, t: Telemetry, r: Ribbon) -> Targets:
        """Turn a Keyframe into a concrete Targets for this tick."""
        g = r.geometry
        prev = self._prev_targets

        # ── Heading ──────────────────────────────────────────────────
        # Strict per-phase heading sources:
        #   dep_runway  → runway takeoff heading (ignores follower;
        #                 we do NOT steer off the centerline on the
        #                 ground for any reason)
        #   dest_runway → runway landing heading (protects flare /
        #                 rollout from cross-track wobble during the
        #                 critical final ~50 ft AGL)
        #   fixed       → keyframe-specified absolute bearing
        #   aim_at      → L1 TRACK FOLLOWER — the plane follows the
        #                 ribbon polyline with cross-track capture and
        #                 turn anticipation. Replaces the old
        #                 bearing-to-aim + past-aim-guard logic which
        #                 could not handle overshoot or drift.
        #   hold        → last tick's heading (inertial hold)
        if kf.heading_mode == "dep_runway":
            hdg = g.dep_heading
        elif kf.heading_mode == "dest_runway":
            hdg = g.rwy_heading
        elif kf.heading_mode == "aim_at" and t.has_position() and self._follower is not None:
            track = self._follower.update(t)
            self._last_track = track
            self.ctx["track_state"] = {
                "cross_track_nm": track.cross_track_nm,
                "along_track_nm": track.along_track_nm,
                "segment_idx": track.segment_idx,
                "segment_progress": track.segment_progress,
                "ribbon_length_nm": track.ribbon_length_nm,
                "lookahead_lat": track.lookahead_lat,
                "lookahead_lon": track.lookahead_lon,
            }
            hdg = track.heading_deg
        elif kf.heading_mode == "fixed" and kf.target_heading_deg is not None:
            hdg = kf.target_heading_deg
        else:  # "hold"
            hdg = prev.heading_deg if prev else t.heading_deg

        # ── Altitude ─────────────────────────────────────────────────
        if kf.alt_mode == "glideslope" and t.has_position():
            # Glideslope altitude: threshold_alt + slope × distance_to_threshold.
            # Slope MUST match the planner's _GLIDE_FT_PER_NM, otherwise the
            # ribbon's descent waypoints and the engine's commanded alt
            # disagree and the alt PID fights itself. Imported from the
            # planner module so the single source of truth lives there.
            #
            # Sign-aware: if plane has passed the threshold (we're on the
            # runway-heading side), clamp d_nm to 0 so we don't command a
            # climb-back. Capped at cruise_alt as an upper bound.
            from uav.nav.flight_plan_v2 import _GLIDE_FT_PER_NM as _SLOPE_FT_PER_NM
            d_nm = haversine_m(t.lat_deg, t.lon_deg,
                               g.thr_lat, g.thr_lon) / 1852.0
            brng_from_thr = bearing_deg(g.thr_lat, g.thr_lon,
                                         t.lat_deg, t.lon_deg)
            # Approach side = bearing roughly matches (rwy_heading + 180).
            back_hdg = (g.rwy_heading + 180.0) % 360.0
            ang_diff = abs(((brng_from_thr - back_hdg + 180.0) % 360.0) - 180.0)
            if ang_diff > 90.0:
                # Plane is on the runway-departing side of threshold.
                d_nm = 0.0
            alt = g.thr_alt_ft + d_nm * _SLOPE_FT_PER_NM
            alt = min(alt, g.cruise_alt_ft)
            # During FLARE, clamp so we don't command negative AGL
            if kf.phase == "FLARE":
                alt = max(alt, g.thr_alt_ft + 2.0)
        elif kf.alt_mode == "target" and kf.target_alt_ft is not None:
            alt = kf.target_alt_ft
        else:  # "hold"
            alt = prev.altitude_ft if prev and prev.altitude_ft is not None else t.altitude_ft

        # ── Speed target ─────────────────────────────────────────────
        speed = kf.target_speed_kts  # None = no speed regulation

        # ── Throttle ─────────────────────────────────────────────────
        if kf.throttle_mode == "alt_scaled":
            # Dense air at low alt needs less thrust for cruise; thinner
            # air at high alt needs more.  At 2.6kft → 0.59, 10kft → 0.70,
            # 20kft → 0.85.  Capped so we never float above 85% at cruise.
            alt_kft = t.altitude_ft / 1000.0
            throttle = max(0.55, min(0.85, 0.55 + 0.015 * alt_kft))
        elif kf.throttle_mode == "idle":
            throttle = 0.0
        elif kf.throttle_mode == "speed_pid":
            # Let the speed target + controller regulate throttle.
            throttle = None
        elif kf.throttle_mode == "explicit":
            throttle = kf.throttle
        else:
            throttle = prev.throttle if prev else None

        # ── Levers (None inherits) ───────────────────────────────────
        gear = (kf.gear_down if kf.gear_down is not None
                else (prev.gear_down if prev and prev.gear_down is not None else True))
        flap = (kf.flap_ratio if kf.flap_ratio is not None
                else (prev.flap_ratio if prev and prev.flap_ratio is not None else 0.0))
        brake = kf.brake_ratio if kf.brake_ratio is not None else 0.0

        return Targets(
            heading_deg=hdg,
            altitude_ft=alt,
            airspeed_kts=speed,
            throttle=throttle,
            brake_ratio=brake,
            gear_down=gear,
            flap_ratio=flap,
            roll_limit=kf.roll_limit,
            pitch_limit=kf.pitch_limit,
            pitch_down_limit=kf.pitch_down_limit,
            yaw_hold=kf.yaw_hold,
            yaw_kp=kf.yaw_kp,
            yaw_limit=kf.yaw_limit,
            throttle_for_alt=kf.throttle_for_alt,
        )

    # ── Phase mapping ────────────────────────────────────────────────

    def _map_phase(self, ribbon_phase: str) -> str:
        """Map ribbon phases to what autopilot.py expects for scoring/
        watchdog logic.

        Historical note: DESCENT used to collapse to CRUISE here so the
        old scoring code (which only knew GROUND/CLIMB/CRUISE/APPROACH/
        LAND) didn't crash. Grep now shows no such dependency, and the
        collapse caused the app to display "CRUISE" during active DESCENT
        keyframes — confusing during approach. Restored DESCENT→DESCENT
        so the app and CSV log the true phase.
        """
        mapping = {
            "GROUND": "GROUND",
            "CLIMB": "CLIMB",
            "CRUISE": "CRUISE",
            "DESCENT": "DESCENT",
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
