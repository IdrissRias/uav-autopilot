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

# Flaps are an altitude-control device on the glideslope (see flap
# logic in _resolve): they add lift, so they deploy only when the plane
# has sunk BELOW the slope and needs pulling back up. On-slope or above,
# the wing stays clean. Hysteresis gap so they don't cycle on noise:
#   ≥ _FLAP_DEPLOY_BELOW_FT below slope → deploy the keyframe's
#     scheduled setting
#   ≤ _FLAP_CLEAN_BELOW_FT below slope (incl. any amount above it)
#     → retract to clean (speed permitting)
#   in between → hold current setting
_FLAP_DEPLOY_BELOW_FT = 100.0
_FLAP_CLEAN_BELOW_FT = 50.0


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
                v_cruise_kts=(float(speeds["v_cruise"])
                              if speeds.get("v_cruise") else None),
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
            # Runway heading + a small centerline correction. A pure
            # heading hold let any residual cross-track at flare entry
            # (or crosswind drift) carry the plane toward the runway
            # edge — nothing measured lateral offset below 30 ft AGL.
            hdg = g.rwy_heading
            if t.has_position():
                d_nm = haversine_m(t.lat_deg, t.lon_deg,
                                   g.thr_lat, g.thr_lon) / 1852.0
                brg_thr_plane = bearing_deg(g.thr_lat, g.thr_lon,
                                            t.lat_deg, t.lon_deg)
                back_hdg = (g.rwy_heading + 180.0) % 360.0
                diff = ((brg_thr_plane - back_hdg + 180.0) % 360.0) - 180.0
                # Perpendicular offset from the extended centerline.
                # diff > 0 → plane displaced left of the approach course
                # (facing the runway) → steer right (positive correction).
                cross_nm = d_nm * math.sin(math.radians(diff))
                correction = max(-8.0, min(8.0, cross_nm * 40.0))
                hdg = (g.rwy_heading + correction) % 360.0
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
            # PIECEWISE glideslope, both slopes imported from the planner
            # (single source of truth). The last _APPROACH_ALT_AGL_FT of
            # descent flies the shallow _FINAL_FT_PER_NM slope so the
            # flare starts from an arrestable sink rate (~600 fpm, not
            # ~1700); everything above that flies the steep
            # _GLIDE_FT_PER_NM descent slope.
            #
            # Sign-aware: if plane has passed the threshold (we're on the
            # runway-heading side), clamp d_nm to 0 so we don't command a
            # climb-back. Capped at cruise_alt as an upper bound.
            from uav.nav.flight_plan_v2 import (
                _GLIDE_FT_PER_NM as _STEEP,
                _FINAL_FT_PER_NM as _FINAL,
                _APPROACH_ALT_AGL_FT as _APP_AGL,
                _FLARE_ALT_AGL_FT as _FLARE_AGL,
            )
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
            final_len_nm = (_APP_AGL - _FLARE_AGL) / _FINAL
            if d_nm <= final_len_nm:
                alt = g.thr_alt_ft + d_nm * _FINAL
            else:
                alt = (g.thr_alt_ft + final_len_nm * _FINAL
                       + (d_nm - final_len_nm) * _STEEP)
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

        # ── Flare sink-rate command ──────────────────────────────────
        # During FLARE, pitch tracks a sink rate instead of an altitude.
        # Target decays with AGL: -420 fpm entering at 30 ft, -220 at
        # 10 ft, -120 at the pavement — an exponential-style arrest to
        # a gentle touchdown regardless of what the approach delivered.
        # (The old alt-PID flare commanded nose-DOWN at 30 ft because
        # the clamped target alt sat below the plane.)
        vs_target = None
        if kf.phase == "FLARE":
            agl_ft = ((t.agl_m * 3.28084)
                      if not math.isnan(t.agl_m) else 30.0)
            vs_target = -(120.0 + max(0.0, agl_ft) * 10.0)
            vs_target = max(vs_target, -600.0)  # never command a dive

        # ── Throttle baseline for throttle_for_alt ───────────────────
        # On glideslope phases the alt→throttle P-law pivots around near
        # idle: on-slope (alt_error ≈ 0) means gravity does the work, and
        # thrust only comes in when we sink below the slope. The old
        # baseline was cruise_throttle (~0.55), which flew the descent
        # hot — 180+ ft above slope before throttle even hit zero.
        throttle_base = 0.12 if (kf.throttle_for_alt
                                 and kf.alt_mode == "glideslope") else None

        # ── Levers (None inherits) ───────────────────────────────────
        gear = (kf.gear_down if kf.gear_down is not None
                else (prev.gear_down if prev and prev.gear_down is not None else True))
        flap = (kf.flap_ratio if kf.flap_ratio is not None
                else (prev.flap_ratio if prev and prev.flap_ratio is not None else 0.0))
        # ── Situational flap control ─────────────────────────────────
        # Flaps are LIFT. On the glideslope they deploy only when the
        # plane has sunk below the slope and needs pulling back up:
        #   ≥ _FLAP_DEPLOY_BELOW_FT below slope → deploy the keyframe's
        #     scheduled setting (the schedule still caps how much flap
        #     this phase may use, so Vfe protection is untouched).
        #   ≤ _FLAP_CLEAN_BELOW_FT below slope, or anywhere above it
        #     → retract to clean, IF speed is at/above v_approach
        #     (retracting raises stall speed; never clean up slow).
        #   in between → hold current setting (hysteresis, no cycling).
        # FLARE is exempt — it always gets its full flaps. Gear is NOT
        # gated: gear is drag without lift, which always helps slow us.
        if (kf.alt_mode == "glideslope" and kf.phase != "FLARE"
                and prev is not None and prev.flap_ratio is not None):
            alt_below_ft = alt - t.altitude_ft  # positive = below slope
            if alt_below_ft >= _FLAP_DEPLOY_BELOW_FT:
                pass  # keep the scheduled `flap` — we need the lift
            elif alt_below_ft <= _FLAP_CLEAN_BELOW_FT:
                if (prev.flap_ratio == 0.0
                        or (not math.isnan(t.airspeed_kts)
                            and t.airspeed_kts >= g.v_approach)):
                    flap = 0.0
                else:
                    flap = prev.flap_ratio  # too slow to clean up
            else:
                flap = prev.flap_ratio  # hysteresis band: hold
        brake = kf.brake_ratio if kf.brake_ratio is not None else 0.0
        # Progressive braking: slamming parkbrake + full wheel brakes at
        # touchdown speed (~98 kts) is how tires blow. Ramp from 30% at
        # 80+ kts to 100% at 40 kts.
        if brake > 0.0 and not math.isnan(t.airspeed_kts):
            spd = t.airspeed_kts
            if spd > 40.0:
                scale = 0.3 + 0.7 * max(0.0, min(1.0, (80.0 - spd) / 40.0))
                brake = brake * scale

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
            throttle_base=throttle_base,
            vs_target_fpm=vs_target,
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
