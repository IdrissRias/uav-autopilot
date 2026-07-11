"""
TECSController — the rebuilt soldier.

Longitudinal control is TECS (energy) + a pitch attitude inner loop. Lateral
(heading→bank→aileron) and yaw are the proven cascades kept verbatim from the
old controller — those were never the problem. Three phases are NOT pure TECS
and stay as explicit overrides: ground/takeoff (throttle commanded, nose eases
up to rotate), and flare (idle + sink-rate arrest). Everything between —
climb, cruise, descent, approach, decelerate — is one TECS law fed a
(altitude, speed) demand by the ribbon.

Drop-in for SimpleFixedWingController: same compute(telemetry, targets, dt)
-> Actuators signature.
"""
from __future__ import annotations

import math

from uav.core.control.base import Controller
from uav.core.control.pid import PID
from uav.core.control.simple_fixedwing import ControlGains, _wrap_deg, _clamp
from uav.core.control.tecs import TECS, TECSParams
from uav.core.control.attitude import default_pitch_axis
from uav.sim.types import Telemetry, Actuators, Targets

FT_TO_M = 0.3048
MS_TO_KT = 1.94384
FPM_TO_MS = 0.00508   # ft/min → m/s

# Flare = hold ONE gentle nose-up attitude and let her settle, idle power.
# NOT a sink-rate chase — that PIO'd on the real plane's noisy VS (ballooned
# 6→46 ft, bled to the stall, dropped in). A fixed held degree is stable.
# Tune this one number if the flare is too firm (lower) or floats (higher).
FLARE_PITCH_DEG = 5.0


class TECSController(Controller):
    def __init__(
        self,
        heading_pid: PID | None = None,
        altitude_pid: PID | None = None,
        airspeed_pid: PID | None = None,
        cruise_throttle: float = 0.45,
        gains: ControlGains | None = None,
        tecs_params: TECSParams | None = None,
    ) -> None:
        # PIDs accepted for constructor compatibility with main.py; unused —
        # TECS and the attitude loop replace them.
        self.gains = gains or ControlGains()
        params = tecs_params or TECSParams()
        params.thr_cruise = cruise_throttle
        self.tecs = TECS(params)
        # Pitch attitude inner loop: the PROVEN X-Plane-tuned one from the old
        # controller (soft 0.05/deg spring + 0.055 rate damper with filtering),
        # NOT the sim-tuned attitude.py — that railed at rotation on the real
        # plane (flight after f3594a7) because X-Plane's elevator bites harder
        # than my sim modelled. This loop flew every takeoff today.
        self._prev_pitch_deg: float | None = None
        self._pitch_rate_filt = 0.0
        # glideslope-split state (pitch=slope, throttle=speed); re-seeded on entry
        self._gs_theta_ref: float | None = None
        self._gs_vs_filt = 0.0
        self._gs_thr: float | None = None
        # lateral state
        self._prev_bank_deg = 0.0
        self._prev_hdg_deg: float | None = None
        # longitudinal state
        self._prev_V_ms: float | None = None
        self._prev_throttle = 0.0

    # ── helpers ──────────────────────────────────────────────────────
    def _airspeed_ms(self, tel: Telemetry) -> float:
        v = tel.airspeed_kts
        if math.isnan(v):
            v = 0.0
        return max(v, 0.0) / MS_TO_KT

    def compute(self, telemetry: Telemetry, targets: Targets, dt: float) -> Actuators:
        g = self.gains

        # ── LATERAL: heading → bank → aileron (proven cascade, unchanged) ─
        if targets.heading_deg is not None:
            hdg_error = _wrap_deg(targets.heading_deg - telemetry.heading_deg)
        else:
            hdg_error = 0.0
        target_bank_deg = hdg_error * g.bank_per_hdg_error
        if targets.roll_limit is not None:
            target_bank_deg = _clamp(target_bank_deg, abs(targets.roll_limit) * 90.0)
        bank_rate = ((telemetry.roll_deg - self._prev_bank_deg) / dt
                     if dt > 0 else 0.0)
        self._prev_bank_deg = telemetry.roll_deg
        roll_cmd = _clamp(
            g.bank_inner_kp * (target_bank_deg - telemetry.roll_deg)
            - g.bank_inner_kd * bank_rate, 1.0)

        # ── LONGITUDINAL ────────────────────────────────────────────────
        V = self._airspeed_ms(telemetry)
        h = (telemetry.altitude_ft * FT_TO_M
             if not math.isnan(telemetry.altitude_ft) else 0.0)
        hdot = (telemetry.vs_fpm * FPM_TO_MS
                if not math.isnan(telemetry.vs_fpm) else 0.0)
        if self._prev_V_ms is None:
            self._prev_V_ms = V
        vdot = (V - self._prev_V_ms) / dt if dt > 0 else 0.0
        self._prev_V_ms = V

        pitch_dmd_deg = 0.0
        # Clear glideslope trim state whenever we're not on the slope, so it
        # re-seeds cleanly the next approach.
        if not getattr(targets, "on_glideslope", False):
            self._gs_theta_ref = None
            self._gs_thr = None

        if targets.throttle is not None and targets.vs_target_fpm is not None:
            # FLARE: idle power (explicit), and hold ONE steady nose-up flare
            # attitude — the proven inner loop eases the nose up to it and
            # keeps it there. She decelerates and settles onto the mains
            # instead of hunting. (Replaced the sink-rate chase that PIO'd on
            # the real plane's noisy VS and ballooned into a stall-drop.)
            throttle_cmd = targets.throttle
            pitch_dmd_deg = FLARE_PITCH_DEG
            self.tecs.reset()

        elif targets.throttle is not None:
            # GROUND / TAKEOFF / any explicit-throttle phase without a sink
            # target. Honour the throttle; ease the nose toward a gentle climb
            # attitude scaled by how far below target we are (0 on the ground
            # where h≈h_dmd is false but speed is ~0, so the surface just holds
            # back-pressure and she rotates at Vr). Bounded so it can't yank.
            throttle_cmd = targets.throttle
            if targets.altitude_ft is not None:
                below = (targets.altitude_ft - telemetry.altitude_ft)
                pitch_dmd_deg = max(0.0, min(8.0, below * 0.004))
            self.tecs.reset()

        elif getattr(targets, "on_glideslope", False) and targets.airspeed_kts is not None:
            # ── GLIDESLOPE: ALTITUDE ONLY. SPEED PLAYS NO ROLE, EVER. ────
            # (Idriss, 2026-07-11.) Throttle serves TARGET ALTITUDE and
            # NOTHING ELSE — no speed target, no stall guard. Speed is a
            # RESULT of pitch + configuration, never a goal, never a reason
            # to add or remove power.
            #
            # SYMMETRIC on purpose (fixed after it landed short, flight
            # c677d781): a one-sided version that could only CUT power
            # (never restore it below the line) meant that any dip under
            # the glideslope was permanent — nothing pulled it back onto
            # the path, and it sank into the ground before the runway.
            # "Accurate as fuck on altitude — that's what guarantees a
            # landing" (Idriss). Above the line → power off, firmly. Below
            # the line → power returns to climb back onto it. Both
            # directions are pure altitude error; neither looks at speed.
            # SAME gain both ways (0.006) — tight, symmetric tracking, not
            # "firm above, gentle below." The real fix for the short landing
            # was matching the glideslope angle to what the King Air can
            # actually fly (see _GLIDE_FT_PER_NM); this symmetric gain is
            # the safety margin on top, so any transient dip below the line
            # is corrected with the same urgency as an excursion above it.
            alt_err_ft = targets.altitude_ft - telemetry.altitude_ft  # + = below target
            if self._gs_thr is None:
                self._gs_thr = self._prev_throttle
            drive = alt_err_ft * 0.006
            self._gs_thr += drive * dt
            self._gs_thr = max(0.0, min(1.0, self._gs_thr))
            throttle_cmd = self._gs_thr

            # PITCH: unchanged — the damped attitude-trim that killed the
            # 987d1262 PIO (filtered VS, slow trim, rate-damped inner loop).
            # Flies the commanded sink rate down; nothing here reacts to
            # speed either.
            vs_now = telemetry.vs_fpm if not math.isnan(telemetry.vs_fpm) else 0.0
            pnow = telemetry.pitch_deg if not math.isnan(telemetry.pitch_deg) else 0.0
            if self._gs_theta_ref is None:   # seed on entry
                self._gs_theta_ref = pnow
                self._gs_vs_filt = vs_now
            self._gs_vs_filt += min(1.0, dt / 0.6) * (vs_now - self._gs_vs_filt)
            vs_ref = (targets.vs_target_fpm
                      if targets.vs_target_fpm is not None else -500.0)
            vs_err = vs_ref - self._gs_vs_filt
            self._gs_theta_ref += vs_err * 0.0018 * dt
            self._gs_theta_ref = max(-8.0, min(6.0, self._gs_theta_ref))
            pitch_dmd_deg = max(-10.0, min(8.0,
                                self._gs_theta_ref + vs_err * 0.0008))
            self.tecs.reset()

        elif targets.airspeed_kts is not None and targets.altitude_ft is not None:
            # TECS — climb / cruise / decelerate (level-ish energy phases).
            h_dmd = targets.altitude_ft * FT_TO_M
            V_dmd = targets.airspeed_kts / MS_TO_KT
            sw = 1.7 if getattr(targets, "pitch_for_speed", False) else 1.0
            throttle_cmd, pitch_dmd_deg = self.tecs.update(
                h, V, hdot, vdot, h_dmd, V_dmd, dt, spdweight=sw)
        else:
            # No speed target (shouldn't happen with the TECS ribbon) — hold
            # the last throttle and a level attitude rather than do anything
            # surprising.
            throttle_cmd = self._prev_throttle
            pitch_dmd_deg = 2.0

        throttle_cmd = max(0.0, min(1.0, throttle_cmd))
        # THE ENGINE IS NOT A SWITCH. TECS on the real (noisy) plane wanted to
        # slam full↔idle every frame; slew-limit to 0.5/s so power moves like
        # a throttle lever, not a light switch. Explicit orders (takeoff full,
        # flare idle) are exempt — those are meant to be immediate.
        if targets.throttle is None and dt > 0:
            step = 0.5 * dt
            throttle_cmd = max(self._prev_throttle - step,
                               min(self._prev_throttle + step, throttle_cmd))
        self._prev_throttle = throttle_cmd

        # Pitch attitude inner loop — hold the TECS-demanded degree. Soft
        # spring + timely rate damper (the old controller's X-Plane-proven
        # gains). This is deliberately GENTLE so it can't rail on the real
        # elevator the way the sim-tuned loop did.
        pdeg = telemetry.pitch_deg if not math.isnan(telemetry.pitch_deg) else 0.0
        if self._prev_pitch_deg is not None and dt > 0:
            raw = (pdeg - self._prev_pitch_deg) / dt
            self._pitch_rate_filt = 0.5 * raw + 0.5 * self._pitch_rate_filt
        self._prev_pitch_deg = pdeg
        pitch_cmd = 0.05 * (pitch_dmd_deg - pdeg) - 0.055 * self._pitch_rate_filt
        # Commander surface clamps (hardware orders).
        if targets.pitch_limit is not None:
            pitch_cmd = min(pitch_cmd, abs(targets.pitch_limit))
        if targets.pitch_down_limit is not None:
            pitch_cmd = max(pitch_cmd, -abs(targets.pitch_down_limit))
        pitch_cmd = _clamp(pitch_cmd, 1.0)

        # ── YAW (ribbon-driven yaw-hold, unchanged) ─────────────────────
        yaw_cmd = 0.0
        if targets.yaw_hold:
            yaw_kp = targets.yaw_kp if targets.yaw_kp is not None else 0.02
            yaw_limit = targets.yaw_limit if targets.yaw_limit is not None else 0.5
            if self._prev_hdg_deg is None or dt <= 0:
                hdg_rate = 0.0
            else:
                hdg_rate = _wrap_deg(telemetry.heading_deg - self._prev_hdg_deg) / dt
            self._prev_hdg_deg = telemetry.heading_deg
            yaw_cmd = _clamp(yaw_kp * hdg_error - 0.008 * hdg_rate, yaw_limit)
        else:
            self._prev_hdg_deg = telemetry.heading_deg

        brake_ratio = max(0.0, min(1.0, targets.brake_ratio
                                   if targets.brake_ratio is not None else 0.0))
        gear_down = targets.gear_down if targets.gear_down is not None else True
        flap_ratio = max(0.0, min(1.0, targets.flap_ratio
                                  if targets.flap_ratio is not None else 0.0))

        return Actuators(
            throttle=throttle_cmd,
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=yaw_cmd,
            brake_ratio=brake_ratio,
            gear_down=gear_down,
            flap_ratio=flap_ratio,
        )
