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
import time
from typing import Optional

from uav.nav.flight_plan_v2 import (
    Keyframe, Ribbon, plan_path, format_ribbon,
)
from uav.nav.geo import bearing_deg, haversine_m
from uav.sim.types import Telemetry, Targets
from uav.core.guidance.track_follower import TrackFollower, TrackState

# Flap deployment is speed-staged and EARLY (see flap logic in
# _resolve): all drag comes out at the top of the descent, the
# VS-tracking pitch counters the lift spike, and nothing deploys low.


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
        # Landing commitment latch (see _resolve). Once armed the plane
        # stops flying and starts landing: one-way, no exit until wheels.
        self._land_committed = False
        self._commit_vs_fpm: Optional[float] = None
        # Command-continuity ramps: each phase CONTINUES from where the
        # last one left the plane. Keyframe targets may step (descent
        # trigger fires 0.3 nm late → alt target steps 300 ft; gear
        # keyframe steps the speed target 24 kts) — the EMITTED command
        # never does. Alt ≤ 50 ft/s, speed ≤ 2.5 kts/s.
        self._cmd_alt_smooth: Optional[float] = None
        self._cmd_spd_smooth: Optional[float] = None
        self._ramp_ts: Optional[float] = None
        # Centerline-correction integrator: a P-only law can't remove a
        # STEADY lateral offset (crosswind, geometry bias) — the plane
        # lands parallel to the runway, beside it. Slow integral trims
        # the residual to zero. Degrees; clamped ±5.
        self._cl_int_deg = 0.0
        self._cl_cross_prev = None
        self._cl_cross_rate = 0.0
        self._cruise_thr_trim = 0.0
        self._td_ticks = 0

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
        self._land_committed = False
        self._commit_vs_fpm = None
        self._cmd_alt_smooth = None
        self._cmd_spd_smooth = None
        self._ramp_ts = None
        self._cl_int_deg = 0.0
        self._cl_cross_prev = None
        self._cl_cross_rate = 0.0
        self._cruise_thr_trim = 0.0
        self._td_ticks = 0
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
        _ts = self.ctx.get("track_state") or {}
        along_nm = _ts.get("along_track_nm")
        while self._idx < len(r.keyframes) - 1:
            kf = r.keyframes[self._idx]
            if kf.trigger.fired(telemetry, agl_ft, along_nm):
                # Touchdown debounce: FLARE→ROLLOUT latched once on a
                # single transient AGL=0 frame while 22 ft up (bounced
                # landing, derotation active mid-air). The wheels-down
                # advance needs 3 consecutive fired ticks.
                if kf.name == "FLARE":
                    self._td_ticks += 1
                    if self._td_ticks < 3:
                        break
                else:
                    self._td_ticks = 0
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
        # Wall-clock dt for command ramps / integrators (bounded so a
        # hiccup can't produce a giant step).
        _now = time.time()
        ramp_dt = (min(0.5, max(0.0, _now - self._ramp_ts))
                   if self._ramp_ts is not None else 0.05)
        self._ramp_ts = _now

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
            # Runway heading + centerline correction, mirroring the
            # landing side. A bare heading hold let any initial
            # misalignment accumulate sideways until the plane mowed
            # the grass (loop flight 96786124's takeoff roll). Anchor:
            # the planned departure point along the departure heading.
            # cross > 0 = drifted right of the centerline → steer left.
            hdg = g.dep_heading
            if t.has_position():
                d_nm = haversine_m(t.lat_deg, t.lon_deg,
                                   g.dep_lat, g.dep_lon) / 1852.0
                if d_nm > 0.002:  # <4 m from anchor: bearing is noise
                    brg = bearing_deg(g.dep_lat, g.dep_lon,
                                      t.lat_deg, t.lon_deg)
                    diff = ((brg - g.dep_heading + 180.0) % 360.0) - 180.0
                    cross_m = d_nm * 1852.0 * math.sin(math.radians(diff))
                    # METER-scale gain. The old 60°/nm gave <1° for a
                    # 10 m drift — a whisper; the plane left the pavement
                    # with the loop nominally "working". On a 45 m-wide
                    # runway, meters are the unit that matters:
                    # 0.8°/m → 8° at 10 m off, capped 15°.
                    hdg = (g.dep_heading
                           - max(-15.0, min(15.0, cross_m * 0.8))) % 360.0
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
                # The INTEGRAL term kills what P never can: a STEADY
                # offset (crosswind / geometry bias) that had the plane
                # landing parallel to the runway, beside it.
                self._cl_int_deg += cross_nm * 10.0 * ramp_dt
                self._cl_int_deg = max(-5.0, min(5.0, self._cl_int_deg))
                cross_m = cross_nm * 1852.0
                on_ground = (not math.isnan(t.agl_m)) and t.agl_m < 5.0
                if on_ground:
                    # Rollout: meter-scale, assertive (0.8°/m, cap 15°) —
                    # nm-scale gains are whispers at runway width.
                    p_term = max(-15.0, min(15.0, cross_m * 0.8))
                    correction = p_term + self._cl_int_deg
                else:
                    # Airborne final: METER-scale, damped. The old 60/nm
                    # was 0.032°/m — 25 m off the centerline (in the grass)
                    # bought <1° of correction, so the plane touched down
                    # off the pavement and the ground steering hauled it
                    # over. New gain ~0.18°/m gives a heading offset that
                    # closes the lateral gap on a ~6 s time constant at
                    # approach speed; the cross-track RATE damper stops it
                    # overshooting into an S-turn. Cap 15°.
                    if self._cl_cross_prev is not None and ramp_dt > 0:
                        rate = (cross_m - self._cl_cross_prev) / ramp_dt
                        self._cl_cross_rate = (0.4 * rate
                                               + 0.6 * self._cl_cross_rate)
                    self._cl_cross_prev = cross_m
                    p_term = max(-15.0, min(15.0, cross_m * 0.18))
                    d_term = max(-6.0, min(6.0, self._cl_cross_rate * 1.2))
                    correction = p_term + d_term + self._cl_int_deg
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
        # Altitude has authority over speed on the glideslope: the VS
        # law flies the path with pitch, config provides the drag, and
        # speed is EMERGENT (bounded by the stall floor below and flap
        # overspeed protection above). No pitch-up bleed, no speed
        # retargeting — superseded by configure-early + pitch-down.
        speed = kf.target_speed_kts  # None = no speed regulation
        # (cruise_hold is computed below; speed is cleared there —
        # power walks altitude and speed is EMERGENT, per doctrine.)

        # ── Cruise = ALTITUDE-COORDINATED, EMERGENT SPEED ────────────
        # In every level CRUISE-phase keyframe, don't chase a speed
        # number (chasing an unreachable target floored the King Air's
        # throttle and it climbed 10,000 ft). Instead, BOTH controls
        # defend altitude, coordinated:
        #   • PITCH holds altitude precisely (vs cascade below) — fast,
        #     tight, so the altitude error stays small.
        #   • THROTTLE is REACTIVE to altitude (throttle_for_alt): above
        #     target → ease off (slow down / sink), below → add power.
        #     Slower energy support; because pitch keeps the error small
        #     it mostly just trims, so the two don't fight into a phugoid.
        # Speed is emergent: whatever that power holds at altitude.
        # DECELERATE keeps its idle (it wants to slow down).
        cruise_hold = (kf.phase == "CRUISE" and kf.alt_mode == "target"
                       and kf.target_alt_ft is not None
                       and kf.throttle_mode != "idle")

        # DESCENT DECOUPLING (same cure as cruise). On the glideslope,
        # PITCH flies the slope (vs_target below) and THROTTLE must hold
        # APPROACH SPEED, not chase altitude. Throttle-on-altitude here
        # slammed full power below the slope and the plane arrived HIGH
        # AND SLOW — the worst corner — where the stall floor then fired
        # full power and ballooned it (flight 193023: 231 ft high at 90 kt,
        # throttle 1.0). Holding speed makes it arrive at v_approach, so
        # the stall floor never trips and it can actually come down. The
        # low side is still covered: pitch sinks slower than the slope to
        # catch it, and the stall floor remains for genuine danger. FLARE
        # keeps its own idle + arrest.
        glide_decouple = (kf.alt_mode == "glideslope"
                          and kf.phase != "FLARE")
        # (Under TECS every phase commands a speed — cruise no longer runs
        # "emergent speed"; TECS holds v_cruise while power/pitch trade energy.)

        # ── Throttle ─────────────────────────────────────────────────
        # No fixed cruise law and NO fixed bases/caps anywhere (user
        # doctrine): closed-loop phases hand throttle=None to the
        # controller, whose single slow-walk integrator ranges the FULL
        # 0–100% and settles on whatever power the plane needs. In
        # cruise_hold the walk is driven by ALTITUDE (speed emergent);
        # on the glideslope by the approach speed target.
        if cruise_hold:
            throttle = None
        elif kf.throttle_mode == "alt_scaled":
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

        # ── ALTITUDE CAPTURE (climb → cruise) ────────────────────────
        # The climb held FULL power right up to cruise altitude, so the
        # plane arrived with ~2000 fpm of momentum and blasted 875 ft
        # through it before pitch could arrest it (the "initial bump").
        # Bleed the climb power over the last CAPTURE_FT so it eases onto
        # the target: full climb power at the band edge, cruise power at
        # the target. Pitch (the alt PID) flattens in step, so the plane
        # rounds off onto cruise alt instead of rocketing past.
        if (kf.phase == "CLIMB" and kf.alt_mode == "target"
                and kf.target_alt_ft is not None and throttle is not None):
            CAPTURE_FT = 700.0
            CRUISE_PWR = 0.50
            remaining = kf.target_alt_ft - t.altitude_ft
            if remaining < CAPTURE_FT and throttle > CRUISE_PWR:
                # frac 1 at band edge → 0 at/above target (never full
                # power once past the target waiting for the trigger).
                frac = max(0.0, min(1.0, remaining / CAPTURE_FT))
                throttle = CRUISE_PWR + (throttle - CRUISE_PWR) * frac

        pitch_cap = kf.pitch_limit
        pitch_down_cap = kf.pitch_down_limit
        # ── Sink-rate commands (pitch flies the path) ────────────────
        # FLARE: target decays with AGL: -420 fpm entering at 30 ft,
        # -220 at 10 ft, -120 at the pavement — an exponential-style
        # arrest regardless of what the approach delivered.
        #
        # DESCENT/APPROACH (glideslope): pitch tracks the REQUIRED sink
        # rate directly — local slope gradient × ground speed, plus a
        # convergence term when off the slope. This is the "counter the
        # config instantly" law: a flap balloon shows up as a VS error
        # the same tick and the elevator eats it, instead of the plane
        # gaining 50 ft while a speed-error law wakes up. Never commands
        # a climb (throttle owns the low side).
        vs_target = None
        if kf.phase == "FLARE":
            # Deepened arrest (was 120 + 10/ft): touchdowns were firm.
            # -80 fpm at the pavement, gentler slope so the hold starts
            # earlier and the plane is nearly floating at contact.
            agl_ft = ((t.agl_m * 3.28084)
                      if not math.isnan(t.agl_m) else 30.0)
            vs_target = -(80.0 + max(0.0, agl_ft) * 8.0)
            vs_target = max(vs_target, -600.0)  # never command a dive
        elif kf.phase == "ROLLOUT":
            # DEROTATION. Without this, rollout fell back to the alt-hold
            # PID whose wound-up integral yanked full nose-UP on the
            # runway (P=+1.00 in every rollout log) — a tail strike
            # waiting to happen. A small negative VS demand on the ground
            # (vs ≈ 0) resolves to steady gentle FORWARD stick: the
            # nosewheel comes down and stays down.
            #
            # BOUNCE GUARD: flight 946a68cb latched ROLLOUT on a
            # transient AGL=0 while still 22 ft up, then derotation
            # pushed the nose DOWN through three bounce cycles — slam,
            # bounce, slam. Airborne again (> 5 ft) means we are NOT
            # rolling out, whatever the keyframe says: fly the flare
            # arrest until genuinely down.
            agl_ro_ft = ((t.agl_m * 3.28084)
                         if not math.isnan(t.agl_m) else 0.0)
            if agl_ro_ft > 5.0:
                vs_target = -150.0
                pitch_cap = 0.25   # give the arrest a real nose-up range
            else:
                vs_target = -100.0
        elif kf.alt_mode == "glideslope" and t.has_position():
            from uav.nav.flight_plan_v2 import (
                _GLIDE_FT_PER_NM as _STEEP2,
                _FINAL_FT_PER_NM as _FINAL2,
                _APPROACH_ALT_AGL_FT as _APP_AGL2,
                _FLARE_ALT_AGL_FT as _FLARE_AGL2,
            )
            d_thr_nm = haversine_m(t.lat_deg, t.lon_deg,
                                   g.thr_lat, g.thr_lon) / 1852.0
            local_slope = (_FINAL2
                           if d_thr_nm <= (_APP_AGL2 - _FLARE_AGL2) / _FINAL2
                           else _STEEP2)
            gs_kts = (t.groundspeed_kts
                      if not math.isnan(t.groundspeed_kts) and t.groundspeed_kts > 30
                      else t.airspeed_kts)
            gs_nm_min = max(0.0, gs_kts) / 60.0
            required_fpm = -(local_slope * gs_nm_min)
            # vs_target is now the BASELINE slope sink rate ONLY — the sink
            # that flies PARALLEL to the glideslope at the current ground
            # speed. It carries NO altitude-convergence term any more.
            #
            # Convergence back onto the line (pull up when low / nose down
            # when high) moved into the controller's glideslope PITCH loop
            # as a STEP-AND-CHECK — the same "nudge, wait 1 s, re-check the
            # altitude, nudge again" feedback loop the throttle and yaw
            # already run, checking the SAME altitude error the throttle
            # checks (Idriss, 2026-07-12: "pulling checking alt, pulling
            # checking alt, we did that every 1 s in the params").
            #
            # Why here, not a continuous conv_gain: the old convergence was
            # a per-tick proportional formula (conv_gain * off_slope), which
            # is exactly the continuous law that let power be added below the
            # line while the nose kept commanding the full descent — energy
            # into speed, not climb, so it "accelerated straight down." A
            # step-and-check that OBSERVES the altitude before nudging again
            # can't do that: below the line, throttle steps power up AND
            # pitch steps the nose up, both re-based on the current error
            # each second, so power increase is PAIRED with pulling up. It
            # also inherently avoids the dive→release balloon the old
            # tangential-capture PD was patching, because it never winds up a
            # demand it hasn't seen the result of yet.
            vs_target = max(-1500.0, min(250.0, required_fpm))
        elif cruise_hold:
            # POWER FOR ALTITUDE, PITCH FOR ATTITUDE (user doctrine). The
            # yoke holds a STEADY, near-level attitude — it does NOT chase
            # altitude with big VS demands (that was the twitchy, constant
            # stick). Just a whisper of VS toward the target (±90 fpm) so
            # the plane doesn't drift, but mostly it holds level. THROTTLE
            # walks the altitude to the target (see the cruise throttle
            # block: power up → the plane rises, power down → it falls) and
            # settles into the power band that holds it. Steady stick,
            # slow power, speed emergent.
            alt_err_ft = alt - t.altitude_ft   # + = below target
            vs_target = max(-90.0, min(90.0, alt_err_ft * 0.6))
        elif (kf.phase == "CLIMB" and kf.alt_mode == "target"
                and kf.target_alt_ft is not None
                and kf.name != "CLIMB_ROTATE"):
            # STEADY CLIMB (user doctrine): pitch holds ONE climb rate all
            # the way up — the yoke does not chase the altitude error. The
            # commanded rate tapers over the last 600 ft so the plane
            # rounds onto cruise altitude (in step with the throttle
            # capture taper below), then cruise_hold's steady-level law
            # takes over. Rate comes from the airframe (rates_fpm.climb).
            # CLIMB_ROTATE keeps the alt-PID: rotation needs the nose
            # yanked up, not a rate hold.
            rates_af = self.ctx.get("airframe", {}).get("rates_fpm", {})
            climb_fpm = float(rates_af.get("climb", 1500.0) or 1500.0)
            climb_fpm = max(500.0, min(2200.0, climb_fpm))
            remaining_c = kf.target_alt_ft - t.altitude_ft
            if remaining_c > 0:
                vs_target = max(150.0,
                                climb_fpm * min(1.0, remaining_c / 600.0))
            else:
                vs_target = 0.0   # at/above target: level, wait for trigger

        # ── Throttle baseline for throttle_for_alt ───────────────────
        # Glideslope descents now fly CONFIGURED (gear + flaps out from
        # the top), so the baseline is a spooled 0.30 — the drag exceeds
        # the slope's needs and the engine works against it, giving the
        # alt law authority in BOTH directions (old near-idle base could
        # only fix "too low"; "too high" hit the idle stop).
        # No fixed bases: the controller's walk integrator finds the
        # power band on its own (user doctrine — nothing is pinned).
        throttle_base = None

        # ── Levers (None inherits) ───────────────────────────────────
        gear = (kf.gear_down if kf.gear_down is not None
                else (prev.gear_down if prev and prev.gear_down is not None else True))
        # ── Drag ladder: gear leads the descent ──────────────────────
        # The 1000 ft/nm slope needs ~1700 fpm of sink at descent speed;
        # a clean airframe mushing at idle gives ~850 (measured, flight
        # 914edfc6) — it can NEVER capture the slope, which also locks
        # out the flaps (they deploy only below the slope). Gear is drag
        # without lift: no balloon, and it moves the "slow and draggy"
        # regime away from the clean wing-drop that spiralled that same
        # flight. Deploy as soon as the descent begins and speed allows;
        # never retracted once out on the way down.
        if (kf.alt_mode == "glideslope"
                and kf.phase in ("DESCENT", "APPROACH")
                and not math.isnan(t.airspeed_kts)
                and t.airspeed_kts <= g.gear_safe_kts):
            gear = True
        flap = (kf.flap_ratio if kf.flap_ratio is not None
                else (prev.flap_ratio if prev and prev.flap_ratio is not None else 0.0))
        # ── Configure EARLY: speed-staged flap deployment ────────────
        # Every mid-descent flap event went wrong (balloon at 120 kts,
        # deploy-lockout until the flare, low-altitude config churn).
        # New philosophy: ALL drag comes out at the top of the descent,
        # where there's 4000 ft of margin and the VS-tracking pitch
        # counters the lift spike the same tick. Only SPEED stages the
        # notches — half the moment it's legal, full once the drag has
        # bled the plane under full-flap speed (seconds later):
        #   speed > flap_safe          → no new flap (hold current)
        #   flap_safe ≥ speed > v_app+10 → up to HALF
        #   speed ≤ v_app+10           → full scheduled setting
        # Once out, flaps stay out (monotonic) — except the hard
        # overspeed retract below. FLARE keeps its schedule as-is.
        if (kf.alt_mode == "glideslope" and kf.phase != "FLARE"
                and prev is not None and prev.flap_ratio is not None
                and not math.isnan(t.airspeed_kts)):
            full_flap_safe = g.v_approach + 10.0
            if t.airspeed_kts > g.flap_safe_kts:
                allowed = prev.flap_ratio       # too fast for anything new
            elif t.airspeed_kts > full_flap_safe:
                allowed = max(prev.flap_ratio, min(flap, 0.5))
            else:
                allowed = max(prev.flap_ratio, flap)
            flap = allowed

        # ── Flap SPEED protection (all phases, overrides everything) ─
        # Flaps while fast is how the last crash happened: deployed at
        # speed, the lift spike ballooned the plane away from the runway
        # and the recovery dive hit the ground. The altitude logic above
        # decides whether flaps are WANTED; this block decides whether
        # they're SAFE. Above flap_safe no new flap deploys (including
        # FLARE — a fast flare entry keeps its current setting); past
        # ~8% over flap_safe anything still out gets pulled back in
        # (retracting while fast has no stall risk — fast IS the
        # protection). The gap between the two thresholds is hysteresis.
        if not math.isnan(t.airspeed_kts):
            prev_flap = (prev.flap_ratio
                         if prev is not None and prev.flap_ratio is not None
                         else 0.0)
            if t.airspeed_kts > g.flap_safe_kts:
                flap = min(flap, prev_flap)   # block any new deployment
            if t.airspeed_kts > g.flap_safe_kts * 1.08:
                flap = 0.0                    # structural: pull them in

        brake = kf.brake_ratio if kf.brake_ratio is not None else 0.0
        # Progressive braking: slamming parkbrake + full wheel brakes at
        # touchdown speed (~98 kts) is how tires blow. Ramp from 30% at
        # 80+ kts to 100% at 40 kts.
        if brake > 0.0 and not math.isnan(t.airspeed_kts):
            spd = t.airspeed_kts
            if spd > 40.0:
                scale = 0.3 + 0.7 * max(0.0, min(1.0, (80.0 - spd) / 40.0))
                brake = brake * scale

        # Stall floor: in-flight phases get a hard "power below this
        # speed" guarantee, whatever the alt-priority coupling wants.
        # FLARE/ROLLOUT/GROUND are excluded — slow there is by design.
        stall_floor = (g.v_land
                       if kf.phase in ("CLIMB", "CRUISE", "DESCENT", "APPROACH")
                       else None)

        # ── LANDING COMMITMENT ───────────────────────────────────────
        # Above the gate the plane negotiates; below it, it executes.
        # Modeled on real autoland: flare latches at ~50 ft radio alt
        # with auto-retard to idle, and the stabilized-approach doctrine
        # gates entry — unstable at the gate means NO latch (airline
        # rule would be a go-around; we keep flying the approach laws).
        #
        # Gate (all must hold): AGL ≤ 50 ft, within 0.6 nm of the
        # threshold, heading within 15° of the runway, sink < 1000 fpm,
        # speed ≤ v_land + 25.
        #
        # Once latched (one-way until wheels / flight reset):
        #   • throttle locked idle (retard), stall floor off
        #   • sink target follows the flare curve as a RATCHET — it only
        #     ever gets shallower; an AGL blip can't re-steepen it, and
        #     nothing ever commands up
        #   • emergency arrest: actual sink past 1000 fpm near the
        #     ground bypasses the ratchet for a full-authority arrest
        #     (commit to landing, not to impact)
        #   • bank capped ~9° (wing-strike protection near the ground)
        roll_lim = kf.roll_limit
        agl_ft_commit = ((t.agl_m * 3.28084)
                         if not math.isnan(t.agl_m) else float("inf"))
        if kf.phase in ("APPROACH", "FLARE") and t.has_position():
            if not self._land_committed:
                d_thr_nm = haversine_m(t.lat_deg, t.lon_deg,
                                       g.thr_lat, g.thr_lon) / 1852.0
                hdg_off = abs(((t.heading_deg - g.rwy_heading + 180.0)
                               % 360.0) - 180.0)
                sink_ok = (math.isnan(t.vs_fpm) or t.vs_fpm > -1000.0)
                speed_ok = (math.isnan(t.airspeed_kts)
                            or t.airspeed_kts <= g.v_land + 25.0)
                if (agl_ft_commit <= 50.0 and d_thr_nm <= 0.6
                        and hdg_off <= 15.0 and sink_ok and speed_ok):
                    self._land_committed = True
                    self._commit_vs_fpm = None
                    print(f"[LANDING] COMMITTED — agl={agl_ft_commit:.0f}ft "
                          f"spd={t.airspeed_kts:.0f}kts "
                          f"sink={t.vs_fpm:.0f}fpm. One way down.")
            if self._land_committed:
                throttle = 0.0          # retard — and it stays there
                throttle_base = None
                stall_floor = None
                roll_lim = 0.10         # ~9° bank cap near the ground
                # BALLOON AUTHORITY: the flare's tail-strike pitch-down
                # cap (0.05) trapped flight 8baa5ae1 in a +1000 fpm
                # balloon zoom it couldn't push out of — it hung, bled
                # 92→68 kts, and fell 110 ft on a dead elevator.
                # Climbing while committed opens the nose-down cap;
                # stopping a balloon IS respecting V/S.
                if (not math.isnan(t.vs_fpm) and t.vs_fpm > 100.0):
                    pitch_down_cap = 0.25
                curve = max(-600.0, -(80.0 + max(0.0, agl_ft_commit) * 8.0))
                if self._commit_vs_fpm is None:
                    self._commit_vs_fpm = curve
                else:  # ratchet: shallower only, never re-steepen
                    self._commit_vs_fpm = max(self._commit_vs_fpm, curve)
                vs_target = self._commit_vs_fpm
                if (not math.isnan(t.vs_fpm) and t.vs_fpm < -1000.0):
                    vs_target = -150.0  # emergency arrest, ratchet bypassed

        # ── Command continuity ramps ─────────────────────────────────
        # Phases CONTINUE each other: the emitted alt/speed commands are
        # rate-limited (alt ≤ 50 ft/s, speed ≤ 2.5 kts/s) so a keyframe
        # advance can never step the plane's orders — the wobble at
        # every transition was the controller flinching at target steps.
        # Near-ground phases bypass: those targets must be instant truth.
        if kf.phase in ("FLARE", "ROLLOUT", "GROUND") or self._land_committed:
            self._cmd_alt_smooth = alt
            self._cmd_spd_smooth = speed
        else:
            if alt is not None:
                if self._cmd_alt_smooth is None:
                    self._cmd_alt_smooth = alt
                else:
                    step = 50.0 * ramp_dt
                    self._cmd_alt_smooth = max(
                        self._cmd_alt_smooth - step,
                        min(self._cmd_alt_smooth + step, alt))
                alt = self._cmd_alt_smooth
            else:
                self._cmd_alt_smooth = None
            if speed is not None:
                if self._cmd_spd_smooth is None:
                    self._cmd_spd_smooth = speed
                else:
                    step = 2.5 * ramp_dt
                    self._cmd_spd_smooth = max(
                        self._cmd_spd_smooth - step,
                        min(self._cmd_spd_smooth + step, speed))
                speed = self._cmd_spd_smooth
            else:
                self._cmd_spd_smooth = None

        # Overspeed ceiling for the energy law's protection floor: the
        # flap-safe speed when flaps are out, a conservative clean Vne proxy
        # otherwise (no explicit Vne in the seed envelope).
        v_max = g.flap_safe_kts if flap > 0.0 else 2.3 * g.v_stall

        return Targets(
            heading_deg=hdg,
            altitude_ft=alt,
            airspeed_kts=speed,
            throttle=throttle,
            brake_ratio=brake,
            gear_down=gear,
            flap_ratio=flap,
            roll_limit=roll_lim,
            pitch_limit=pitch_cap,
            pitch_down_limit=pitch_down_cap,
            yaw_hold=kf.yaw_hold,
            yaw_kp=kf.yaw_kp,
            yaw_limit=kf.yaw_limit,
            # Cruise DECOUPLING: pitch (vs cascade) owns ALTITUDE, and the
            # throttle owns SPEED (throttle_for_alt cleared → speed PID).
            # Throttle-on-ALTITUDE drove a violent phugoid: the engine's
            # lag turned its VS-damping into a driver, both loops fought
            # over altitude, and the plane rang ±200 ft the whole cruise
            # (flight abf5d1fa). Throttle-on-SPEED is classic autothrottle
            # and DAMPS the phugoid (balloon up → speed bleeds → power
            # comes in → pulled back). Reachable speed target required, or
            # the throttle floors and climbs away.
            throttle_for_alt=(kf.throttle_for_alt
                              and not cruise_hold and not glide_decouple),
            throttle_base=throttle_base,
            # Cap cruise throttle so the speed capture is gentle: full
            # power to accelerate from the slow level-off speed to cruise
            # climbs the plane faster than pitch can hold (1300 ft
            # overshoot). 0.75 still reaches 175 kt (level needs ~0.55).
            throttle_max=None,  # full 0–100% range in every phase
            vs_target_fpm=vs_target,
            stall_floor_kts=stall_floor,
            v_max_kts=v_max,
            on_glideslope=glide_decouple,
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
