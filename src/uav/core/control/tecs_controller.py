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
        # step-and-check timer for the glideslope throttle loop (below)
        self._gs_thr_wait = 0.0
        # step-and-check state for yaw (below)
        self._yaw_cmd_held = 0.0
        self._yaw_wait = 0.0
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
        throttle_self_limited = False  # True = the branch below already rate-limited itself
        # Clear glideslope trim state whenever we're not on the slope, so it
        # re-seeds cleanly the next approach.
        if not getattr(targets, "on_glideslope", False):
            self._gs_theta_ref = None
            self._gs_thr = None
            self._gs_thr_wait = 0.0

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
            # landing" (Idriss). Above the line → power eases off. Below
            # the line → power eases back in. Both directions are pure
            # altitude error; neither looks at speed.
            #
            # STEP-AND-CHECK, not a continuous formula (Idriss, 2026-07-11:
            # "the alt needs to be increased by 0.05, check alt again, if
            # still below increase again, if not don't, if up decrease").
            # This replaces the continuous proportional walk entirely. Every
            # THROTTLE_CHECK_S seconds: look at the altitude error, take ONE
            # fixed +-THROTTLE_STEP nudge in the direction that helps, then
            # go quiet and let the airplane's own inertia show the result
            # BEFORE judging again. This is also why the earlier windup
            # incident (theta_ref accumulating during a 36 s deficit, then
            # ballooning +217 ft on the way back — see the Notion audit)
            # can't happen to the throttle side any more: there is no
            # accumulator to wind up, just a bounded step taken periodically
            # off the CURRENT observed error. A small deadband (+-20 ft)
            # means it holds once close instead of chattering by 0.05 every
            # cycle forever. Checking THROTTLE_CHECK_S (3 s) apart, not
            # every tick, gives the engine + aerodynamic response time to
            # actually show up before the next decision — a tick-by-tick
            # check would just be re-judging a change that hasn't landed
            # yet, the same "loop faster than the airplane" mistake that
            # phugoided cruise earlier today.
            # TIERED step size (Idriss, 2026-07-11, final correction): the
            # step now scales with how far off the line we are — fine near
            # the target, urgent far from it. A flat 0.01 step would take
            # 1000+ seconds to correct a genuine 1000 ft deficit (too slow
            # for a ~1-3 min descent); a flat 0.05 was too hot once close
            # (part of what caused the windup/balloon incident). Checked
            # every THROTTLE_CHECK_S (1 s), same as before — enough time for
            # the engine + aerodynamic response to actually show up before
            # judging again. The decision is ALWAYS re-based on the CURRENT
            # observed altitude error at each check, never a blind
            # continuation — "not empty increasing" — which is also what
            # keeps this immune to the theta_ref-style windup that caused
            # the earlier +217 ft balloon: there is no accumulator, just a
            # bounded step taken periodically off what's true right now.
            THROTTLE_CHECK_S = 1.0
            THROTTLE_DEADBAND_FT = 5.0   # noise floor only, not a "close enough" zone
            alt_err_ft = targets.altitude_ft - telemetry.altitude_ft  # + = below target
            abs_err = abs(alt_err_ft)
            if abs_err <= 100.0:
                THROTTLE_STEP = 0.01
            elif abs_err <= 500.0:
                THROTTLE_STEP = 0.05
            else:   # <=1000 ft and beyond — no band defined past 1000, so
                THROTTLE_STEP = 0.2   # hold at the outermost step rather than leave it undefined
            if self._gs_thr is None:
                self._gs_thr = self._prev_throttle
                self._gs_thr_wait = 0.0
            self._gs_thr_wait += dt
            if self._gs_thr_wait >= THROTTLE_CHECK_S:
                self._gs_thr_wait = 0.0
                if alt_err_ft > THROTTLE_DEADBAND_FT:
                    self._gs_thr = min(1.0, self._gs_thr + THROTTLE_STEP)
                elif alt_err_ft < -THROTTLE_DEADBAND_FT:
                    self._gs_thr = max(0.0, self._gs_thr - THROTTLE_STEP)
                # else: within the noise-floor deadband — hold, no step.
            throttle_cmd = self._gs_thr
            throttle_self_limited = True   # the step-and-check IS the rate limit;
            # the generic 0.05/s outer slew below would otherwise silently
            # cap the 0.05/0.2 tiers down to its own rate, defeating the
            # whole point of having bigger steps for bigger errors (found
            # on the bench: tier2/tier3/beyond ALL measured +0.025 instead
            # of +0.05/+0.2/+0.2 — the outer slew was the true bottleneck).

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
            # Trim gain halved (0.0018 -> 0.0008) alongside the flight_engine
            # conv_gain cut — the trim was winding up past what the
            # pitch_limit=0.25 nose-up cap can actually deliver, and
            # unwinding slowly enough to overshoot on the way back. Gentler
            # trim + a gentler upstream demand together, not either alone.
            self._gs_theta_ref += vs_err * 0.0008 * dt
            self._gs_theta_ref = max(-8.0, min(6.0, self._gs_theta_ref))
            pitch_dmd_deg = max(-10.0, min(8.0,
                                self._gs_theta_ref + vs_err * 0.00035))
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
        # THE ENGINE IS NOT A SWITCH. Cap tightened 0.5/s -> 0.05/s (Idriss,
        # 2026-07-11, after the crash audit): a stalled control tick has an
        # uncapped dt, and the slew step is proportional to dt, so a 2 s
        # stall legally permitted a 100% throttle jump under the old 0.5/s
        # cap — confirmed in the crash log (row 941: 0.42->1.00 in 0.098 s,
        # a 12x violation). A 10x tighter rate shrinks ANY stall's max jump
        # by the same 10x, protecting against this cause and any other
        # source of a large dt, not just the ones already found. Explicit
        # orders (takeoff full, flare idle) are exempt — those are meant to
        # be immediate. Full 0-100% now takes >=20 s in closed-loop phases
        # (cruise only now — glideslope is exempt, see throttle_self_limited
        # above: its own tiered step-and-check already rate-limits it, and
        # a second flat 0.05/s cap on top of that was silently defeating
        # the whole point of the bigger tiers for bigger errors).
        if targets.throttle is None and dt > 0 and not throttle_self_limited:
            step = 0.05 * dt
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

        # ── YAW (ribbon-driven yaw-hold): STEP-AND-CHECK, SPEED-SCALED ───
        # (Idriss, 2026-07-11.) Same philosophy as the throttle loop above —
        # a fixed nudge, then wait and observe, rather than a continuous
        # formula reacting every tick — but with YAW's own, much faster
        # physics: the nose starts responding to rudder within about a
        # second, nothing like altitude's multi-second lag, so this checks
        # far more often than the throttle loop (YAW_CHECK_S vs
        # THROTTLE_CHECK_S) — same structure, different timing, because the
        # two controls are not the same speed of animal.
        #
        # Still speed-scaled: the rudder is an aerodynamic surface — the
        # SAME deflection produces LESS actual turning force as airspeed
        # drops (dynamic pressure falls with V^2), and the old law used one
        # fixed step across the WHOLE yaw_hold speed range (TAKEOFF_ROLL
        # 0->~105kt; ROLLOUT touchdown ~85-108kt down to a stop) with no
        # awareness of that. X-Plane holds whatever we send (a persistent
        # DataRef write, not a pulse) — the bottleneck was never "can we
        # hold it," it was that we never asked for MORE as authority
        # weakened. The step itself (not a continuous gain now) is scaled
        # by (V/Vref)^2, same pattern as attitude.py's pitch/roll axes:
        # floored/capped so it can't blow up near a stop or get suppressed
        # to nothing at speed.
        YAW_REF_KTS = 60.0      # mid-range of the takeoff-roll/rollout envelope
        # Revised finer (Idriss, 2026-07-11): step 0.05->0.01, check
        # 0.75s->1.0s (now uniform with the throttle loop's cadence). The
        # step is still speed-scaled below (/q_ratio), so the EFFECTIVE
        # step at low speed (0-60kt, where authority is weakest) is closer
        # to ~0.029 — full rudder range in ~21s at the low-authority end,
        # far slower (and less needed) at high speed.
        YAW_STEP = 0.01
        YAW_CHECK_S = 1.0
        YAW_DEADBAND_DEG = 2.0
        yaw_cmd = 0.0
        if targets.yaw_hold:
            yaw_limit = targets.yaw_limit if targets.yaw_limit is not None else 0.5
            self._prev_hdg_deg = telemetry.heading_deg
            gs = (telemetry.groundspeed_kts
                  if not math.isnan(telemetry.groundspeed_kts) and telemetry.groundspeed_kts > 0
                  else telemetry.airspeed_kts)
            gs = gs if not math.isnan(gs) else YAW_REF_KTS
            q_ratio = max(0.35, min(2.5, (max(gs, 15.0) / YAW_REF_KTS) ** 2))
            self._yaw_wait += dt
            if self._yaw_wait >= YAW_CHECK_S:
                self._yaw_wait = 0.0
                step = YAW_STEP / q_ratio   # bigger nudge at low speed, smaller at high
                if hdg_error > YAW_DEADBAND_DEG:
                    self._yaw_cmd_held = min(yaw_limit, self._yaw_cmd_held + step)
                elif hdg_error < -YAW_DEADBAND_DEG:
                    self._yaw_cmd_held = max(-yaw_limit, self._yaw_cmd_held - step)
                # else: within the deadband — hold, no step.
            yaw_cmd = _clamp(self._yaw_cmd_held, yaw_limit)
        else:
            self._prev_hdg_deg = telemetry.heading_deg
            self._yaw_cmd_held = 0.0
            self._yaw_wait = 0.0

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
