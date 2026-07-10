"""
Dumb-soldier controller.

Commander–soldier model: the ribbon gives orders, this controller executes
them without second-guessing. Only hardware truths are enforced (±1 on
surfaces, [0,1] on throttle/brakes). No phase logic. No rate limiters. No
hardcoded pitch curves. No stall/overspeed protection. No altitude-dependent
roll scheduling. If the ribbon says "bank 90°," we bank 90°.

Control laws:
  heading → bank → aileron (cascaded)
  altitude → pitch (single PID)
  speed    → throttle (single PID; bypassed if throttle is ribbon-commanded)

Optional passthrough clamps from Targets: roll_limit, pitch_limit. These are
commander-issued orders. The soldier honors them because the ribbon asked, not
because it has policy of its own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from uav.core.control.base import Controller
from uav.core.control.pid import PID
from uav.sim.types import Telemetry, Actuators, Targets


def _wrap_deg(error_deg: float) -> float:
    while error_deg > 180.0:
        error_deg -= 360.0
    while error_deg < -180.0:
        error_deg += 360.0
    return error_deg


def _clamp(value: float, limit: float) -> float:
    if limit <= 0.0:
        return value
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


# Cascaded heading-loop defaults. Used when calibration is absent; gains may
# be overridden via derive_gains() from a learned envelope.
BANK_PER_HDG_ERROR = 1.5   # degrees bank commanded per degree heading error
BANK_INNER_KP = 0.020      # aileron per degree of bank error
BANK_INNER_KD = 0.010      # aileron damping per deg/s of bank rate

# ── Altitude → Throttle law (TUNABLE) ────────────────────────────────
# THE formula that decides how much power to add for how much altitude
# we need to gain/lose, in level throttle-defends-altitude flight:
#
#   throttle = cruise_power
#            + KP · alt_error            (immediate: power per foot low)
#            + KI · ∫ alt_error dt        (trim: settles on the ONE steady
#                                          power the altitude needs)
#            − KD · vertical_speed        (damping: back off before the
#                                          error flips, so it can't porpoise)
#
# alt_error > 0 means BELOW target (add power); < 0 means above (reduce).
# KP is the direct "throttle per foot" knob. It is deliberately GENTLE:
# a big KP lunges to 95 % for a trim-sized error and porpoises. KI does
# the settling. Override per-airframe from the yaml (alt_throttle:).
ALT_THROTTLE_KP = 0.0008   # throttle per foot low  (100 ft → +0.08)
ALT_THROTTLE_KI = 0.00016  # trim integrator rate
ALT_THROTTLE_KD = 0.00030  # throttle per fpm of vertical speed (damping)
ALT_THROTTLE_P_CLAMP = 0.15  # max |P| contribution (anti-lunge)


@dataclass
class ControlGains:
    bank_per_hdg_error: float = BANK_PER_HDG_ERROR
    bank_inner_kp: float = BANK_INNER_KP
    bank_inner_kd: float = BANK_INNER_KD
    # Altitude → throttle law (see constants above). Tunable per-airframe.
    alt_throttle_kp: float = ALT_THROTTLE_KP
    alt_throttle_ki: float = ALT_THROTTLE_KI
    alt_throttle_kd: float = ALT_THROTTLE_KD
    alt_throttle_p_clamp: float = ALT_THROTTLE_P_CLAMP


def derive_gains(envelope) -> ControlGains:
    """Convert measured roll sensitivity into inner-loop gains.

    With no calibration (confidence < 0.3) we return defaults.
    """
    cal_conf = getattr(envelope, 'calibration_confidence', 0.0)
    if cal_conf < 0.3:
        return ControlGains()

    roll_sens = getattr(envelope, 'roll_sensitivity', 0.0)
    if roll_sens < 1.0:
        return ControlGains()

    # kp targets ~5 deg/s correction per 10° bank error
    kp = max(0.005, min(0.06, 5.0 / (roll_sens * 10.0)))
    kd = max(0.002, min(0.02, 2.0 / roll_sens))
    return ControlGains(
        bank_per_hdg_error=BANK_PER_HDG_ERROR,
        bank_inner_kp=kp,
        bank_inner_kd=kd,
    )


class SimpleFixedWingController(Controller):
    def __init__(
        self,
        heading_pid: PID,
        altitude_pid: PID,
        airspeed_pid: PID,
        cruise_throttle: float,
        gains: ControlGains | None = None,
    ) -> None:
        # heading_pid kept for API compat — inner loop is cascaded, not a PID
        self.heading_pid = heading_pid
        self.altitude_pid = altitude_pid
        self.airspeed_pid = airspeed_pid
        self.cruise_throttle = cruise_throttle
        self.gains = gains or ControlGains()
        self._prev_bank_deg = 0.0
        self._prev_hdg_deg: float | None = None
        # Leaky integrator for the throttle_for_alt alt hold. P-only left
        # a standing alt error whenever holding altitude needed more (or
        # less) than the baseline throttle. The leak (time constant ~50 s
        # at 20 Hz) self-limits windup; cleared whenever the mode is off.
        self._alt_thr_integral = 0.0
        # Last commanded throttle, for the slew limiter ("the engine is
        # not a switch"). Starts at idle.
        self._prev_throttle = 0.0
        # Coupling-mode tracker: the classic PIDs sit UNUSED while the
        # coupled laws fly, their integrals frozen mid-thought. Re-entering
        # classic mode with a stale integral produced full-stick surprises
        # (DECELERATE's idle zoom, BASE_LEG's full-power climb). Reset on
        # every flip.
        self._prev_coupled: bool | None = None
        # VS-law smoothing state (see the vs_target branch).
        self._vs_target_smooth: float | None = None
        self._prev_vs: float | None = None
        self._vs_rate_filt = 0.0
        # Attitude-cascade state (VS branch): trim reference + rate filter.
        self._theta_ref_deg: float | None = None
        self._prev_pitch_deg: float | None = None
        self._pitch_rate_filt = 0.0
        self._vs_filt: float | None = None
        self._theta_cmd_prev: float | None = None
        # Last commanded pitch, for the stick slew limiter.
        self._prev_pitch = 0.0
        # The ONE closed-loop throttle: a slow-walk integrator, full
        # 0–100% range, seeded from the last commanded throttle when a
        # closed-loop phase begins (continuity). None while the ribbon
        # commands throttle explicitly.
        self._thr_walk: float | None = None
        # Config feed-forward state: flaps/gear are drag events WE
        # schedule — spool the walk against them the instant they deploy
        # instead of letting speed sag and chasing it (12 kt sag observed).
        self._ff_flap_prev: float | None = None
        self._ff_gear_prev: bool | None = None

    def compute(self, telemetry: Telemetry, targets: Targets, dt: float) -> Actuators:
        hdg_error = _wrap_deg(targets.heading_deg - telemetry.heading_deg)
        alt_error = targets.altitude_ft - telemetry.altitude_ft
        # A keyframe may legitimately carry NO speed target (None =
        # "no speed regulation", e.g. emergent-speed cruise). No target
        # means no error — without this guard the subtraction crashes
        # the control loop mid-flight.
        spd_error = ((targets.airspeed_kts - telemetry.airspeed_kts)
                     if targets.airspeed_kts is not None else 0.0)

        # Coupling-mode flip → wipe the stale integrals of whichever
        # loops sat unused (they hold minutes-old wound-up state).
        if (self._prev_coupled is not None
                and targets.throttle_for_alt != self._prev_coupled):
            self.altitude_pid.reset()
            self.airspeed_pid.reset()
        self._prev_coupled = targets.throttle_for_alt

        # ── Heading → Bank → Aileron ─────────────────────────────────
        g = self.gains
        target_bank_deg = hdg_error * g.bank_per_hdg_error

        # Commander-issued roll clamp (optional). roll_limit is in [-1,+1]
        # actuator units; we map 1.0 → 90° of commanded bank.
        if targets.roll_limit is not None:
            max_bank = abs(targets.roll_limit) * 90.0
            target_bank_deg = _clamp(target_bank_deg, max_bank)

        bank_error = target_bank_deg - telemetry.roll_deg
        bank_rate = (telemetry.roll_deg - self._prev_bank_deg) / dt if dt > 0 else 0.0
        self._prev_bank_deg = telemetry.roll_deg

        roll_cmd = g.bank_inner_kp * bank_error - g.bank_inner_kd * bank_rate
        roll_cmd = _clamp(roll_cmd, 1.0)  # hardware truth

        # ── Pitch + throttle: which loop drives what depends on the
        # coupling mode.
        #
        #   Default (classic):    alt PID → pitch, speed PID → throttle
        #   throttle_for_alt:     alt PID → throttle, speed PID → pitch
        #                         (alt-priority "power for altitude"
        #                          coupling — used in cruise/descent so
        #                          the throttle defends alt instead of
        #                          chasing speed past target alt).
        #
        # Both modes use the same PID instances but feed errors to
        # different actuators. When swapping (throttle_for_alt=True),
        # the speed PID's integral can be poisoned by climb-phase
        # windup, so we use a PROPORTIONAL-ONLY response for the
        # swapped paths (gains picked to match the original PID's
        # full-strength response at typical errors).
        if targets.vs_target_fpm is not None and not math.isnan(telemetry.vs_fpm):
            # Sink-rate tracking via an ATTITUDE CASCADE. Stick position
            # is physically a pitch RATE: between stick and vertical
            # speed sit TWO integrations plus 1–2 s of aero lag, and
            # proportional control through a double integrator is a
            # textbook oscillator at ANY gain (low gain = slow porpoise,
            # high gain = full-scale railing — both flown this week).
            # Every real autopilot closes an attitude loop first:
            #   INNER: stick holds pitch ATTITUDE (one integration —
            #          stiff, self-damping via pitch-rate feedback)
            #   OUTER: VS error nudges the attitude TARGET a few
            #          degrees + a slow trim integrator finds the
            #          attitude that holds the slope.
            vs_now_fpm = telemetry.vs_fpm
            # Target slew (asymmetric: arrests ≤600 fpm/s so they can't
            # saturate into balloon zooms; steepening 1500 fpm/s).
            if self._vs_target_smooth is None:
                self._vs_target_smooth = targets.vs_target_fpm
            else:
                up_step = 600.0 * dt
                down_step = 1500.0 * dt
                self._vs_target_smooth = max(
                    self._vs_target_smooth - down_step,
                    min(self._vs_target_smooth + up_step,
                        targets.vs_target_fpm))
            # HOLD the attitude via a TRIM INTEGRAL. The trim ref
            # accumulates the exact steady attitude the airplane needs
            # and just keeps holding it — that IS "keep the current
            # inputs, never let go." The old ±50 fpm deadband ("close
            # enough = touch nothing") was the catch-and-release: near
            # target it stopped correcting, the plane drifted, the loop
            # woke up and grabbed again. Removed. A tiny 10 fpm noise
            # floor is all that survives. Integral gain roughly doubled
            # so the trim converges instead of creeping forever.
            if self._vs_filt is None:
                self._vs_filt = vs_now_fpm
            else:
                a_vs = min(1.0, dt / 0.4)
                self._vs_filt += a_vs * (vs_now_fpm - self._vs_filt)
            vs_err = self._vs_target_smooth - self._vs_filt  # fpm, + = pull
            if abs(vs_err) < 10.0:      # noise floor only, not a hold gap
                vs_err = 0.0

            # OUTER: attitude target = trim integral + proportional lead.
            # The integral holds; the P term gives it immediate authority.
            if self._theta_ref_deg is None:
                self._theta_ref_deg = telemetry.pitch_deg
            self._theta_ref_deg += vs_err * 0.0025 * dt  # 300 fpm → 0.75°/s
            self._theta_ref_deg = max(-10.0, min(12.0, self._theta_ref_deg))
            theta_raw = self._theta_ref_deg + vs_err * 0.0015  # 500 fpm → 0.75°
            theta_raw = max(-10.0, min(14.0, theta_raw))
            # Rate-limit the commanded attitude itself: micro-steps only.
            if self._theta_cmd_prev is None:
                self._theta_cmd_prev = telemetry.pitch_deg
            step = 1.5 * dt
            theta_cmd = max(self._theta_cmd_prev - step,
                            min(self._theta_cmd_prev + step, theta_raw))
            self._theta_cmd_prev = theta_cmd

            # INNER: attitude hold — SOFT spring, STRONG timely damper.
            # Forensics (flight b71fa83f→013738 descent): the loop rang
            # at ~1 Hz with the stick alternating sign every 0.6 s. The
            # stick low-pass (τ 0.15) + heavy rate filtering added
            # 50-90° of phase lag INSIDE the loop — the damping term
            # arrived late enough to DRIVE the oscillation instead of
            # opposing it. Damping must arrive on time: light filter
            # (0.5 blend), kd nearly tripled, kp halved.
            if self._prev_pitch_deg is not None and dt > 0:
                raw_rate = (telemetry.pitch_deg - self._prev_pitch_deg) / dt
                self._pitch_rate_filt = (0.5 * raw_rate
                                         + 0.5 * self._pitch_rate_filt)
            self._prev_pitch_deg = telemetry.pitch_deg
            pitch_cmd = (0.05 * (theta_cmd - telemetry.pitch_deg)
                         - 0.055 * self._pitch_rate_filt)
        elif targets.throttle_for_alt:
            # Speed → Pitch (sign-inverted) + vertical-speed damping.
            #   spd_err > 0 (too slow) → pitch_cmd < 0 (nose-down → gain speed)
            #   spd_err < 0 (too fast) → pitch_cmd > 0 (nose-up → bleed speed)
            # The VS term is the phugoid killer: P-only speed→pitch plus
            # P-only alt→throttle ring energy back and forth (flight
            # 20260706_133421 swung ±300 ft / ±15 kts in cruise, out of
            # phase — constant total energy sloshing). Damping the
            # exchange RATE (climbing fast → ease the nose down) removes
            # the oscillation without touching the setpoints.
            SPEED_TO_PITCH_KP = 0.015   # 10 kts → 0.15 pitch (~9°)
            # VS damping is the phugoid killer, and it was tuned soft for
            # the SF50 (0.00008). The heavier/faster King Air phugoided
            # +-260 ft / +-25 kt through it. Tripled to 0.00025 (1000 fpm
            # -> 0.25 pitch opposing): climbing hard now gets a firm
            # nose-down BEFORE the swing builds, sinking a firm nose-up.
            VS_TO_PITCH_DAMP = 0.00025
            vs = telemetry.vs_fpm if not math.isnan(telemetry.vs_fpm) else 0.0
            pitch_cmd = (-spd_error * SPEED_TO_PITCH_KP
                         - vs * VS_TO_PITCH_DAMP)
            # Never DIVE for speed while at/below the target altitude —
            # buying KE with PE we don't have is the throttle's job.
            # (TRANSITION entered 20 kts slow and 50 ft low; the pitch
            # law dove −0.31, sank further, then zoomed +130 ft over.
            # Nose-down for speed is legitimate only with alt to spare.)
            if alt_error > -20.0:
                pitch_cmd = max(pitch_cmd, -0.05)
            # …and never CLIMB to bleed speed. Pulling up converts the
            # surplus into altitude that must be dumped again minutes
            # later (INBOUND traded 30 kts for +300 ft on the loop
            # flight). Bleeding happens level or descending, full stop.
            if spd_error < -5.0 and vs > 100.0:
                pitch_cmd = min(pitch_cmd, 0.04)
            # Leaving VS-tracking mode: clear its smoothing state.
            self._vs_target_smooth = None
            self._prev_vs = None
            self._vs_rate_filt = 0.0
            self._theta_ref_deg = None
            self._vs_filt = None
            self._theta_cmd_prev = None
        else:
            pitch_cmd = self.altitude_pid.update(
                alt_error, dt, measurement=telemetry.altitude_ft,
            )
            self._vs_target_smooth = None
            self._prev_vs = None
            self._vs_rate_filt = 0.0
            self._theta_ref_deg = None
            self._vs_filt = None
            self._theta_cmd_prev = None

        # Commander-issued pitch clamp. Nose-up cap is always honored; nose-down
        # cap is opt-in.
        if targets.pitch_limit is not None:
            pitch_cmd = min(pitch_cmd, abs(targets.pitch_limit))
        if targets.pitch_down_limit is not None:
            pitch_cmd = max(pitch_cmd, -abs(targets.pitch_down_limit))
        # Minimal output smoothing (τ≈0.04 s): spike removal ONLY.
        # The previous τ 0.15 sat INSIDE the attitude loop and its
        # phase lag turned the damping term into a driver (~1 Hz ring,
        # stick alternating every 0.6 s). Smoothness comes from proper
        # rate damping arriving ON TIME, never from filtering the loop.
        if dt > 0:
            alpha = min(1.0, dt / 0.04)
            pitch_cmd = self._prev_pitch + alpha * (pitch_cmd - self._prev_pitch)
        pitch_cmd = _clamp(pitch_cmd, 1.0)  # hardware truth
        self._prev_pitch = pitch_cmd

        # ── Throttle ─────────────────────────────────────────────────
        if targets.throttle is not None:
            # Explicit throttle from ribbon (e.g. CLIMB full, FLARE idle)
            self._alt_thr_integral = 0.0
            self._thr_walk = None   # next closed-loop phase re-seeds
            if targets.flap_ratio is not None:
                self._ff_flap_prev = targets.flap_ratio
            if targets.gear_down is not None:
                self._ff_gear_prev = bool(targets.gear_down)
            throttle_cmd = targets.throttle
        elif targets.throttle_for_alt:
            # Alt → Throttle: GENTLE proportional + a slow TRIM INTEGRAL
            # that settles onto the ONE steady power the altitude needs
            # and holds it. (User doctrine: "a few dozen feet low should
            # add a few percent, not lunge to 95%; slowly reach one
            # constant permanent power.") The old P gain (0.003) dumped
            # +0.30 for a 100 ft error — that lunge was the up-and-down.
            #   alt_err > 0 (below target) → throttle UP  (gently)
            #   alt_err < 0 (above target) → throttle DOWN (gently)
            # Tunable gains (per-airframe via ControlGains / the yaml).
            ALT_TO_THROTTLE_KP = self.gains.alt_throttle_kp
            ALT_TO_THROTTLE_KI = self.gains.alt_throttle_ki
            base = (targets.throttle_base
                    if targets.throttle_base is not None
                    else self.cruise_throttle)
            # The trim integral is now the PRIMARY term. Its leak is very
            # slow (~0.9998/tick ≈ 200 s) so it HOLDS the steady power
            # instead of decaying and forcing the error to keep feeding
            # it — that decay-and-rebuild was part of the hunting.
            self._alt_thr_integral += alt_error * dt * ALT_TO_THROTTLE_KI
            self._alt_thr_integral *= 0.9998
            self._alt_thr_integral = max(-0.35, min(0.35, self._alt_thr_integral))
            # VS damping: climbing through the target → back power off
            # EARLY, before the alt error flips. Damped vs the COMMANDED
            # sink when one exists, so a normal descent isn't fought.
            ALT_TO_THROTTLE_VS_DAMP = self.gains.alt_throttle_kd  # per fpm
            vs_ref = targets.vs_target_fpm if targets.vs_target_fpm is not None else 0.0
            vs_now = telemetry.vs_fpm if not math.isnan(telemetry.vs_fpm) else vs_ref
            vs_thr = vs_now - vs_ref
            # Gentle P, tightly clamped: it provides only the immediate
            # nudge; the integral does the settling. No lunging.
            _pc = self.gains.alt_throttle_p_clamp
            p_term = max(-_pc, min(_pc * 0.8, alt_error * ALT_TO_THROTTLE_KP))
            throttle_cmd = (base + p_term
                            + self._alt_thr_integral
                            - vs_thr * ALT_TO_THROTTLE_VS_DAMP)
        else:
            self._alt_thr_integral = 0.0
            # ONE continuous throttle law (user doctrine): the power is
            # free from 0 to 100% in EVERY closed-loop phase, and it only
            # ever WALKS — bit by bit — toward whatever the plane needs,
            # settling there. No fixed bases, no caps, no per-phase
            # arithmetic. The integrator seeds from the last command so
            # each phase continues the last.
            #   speed target present (glideslope/approach) → walk on the
            #     speed error (power holds landing speed);
            #   no speed target (cruise — speed emergent)  → walk on the
            #     altitude error, damped by vertical speed so the power
            #     eases off as the climb develops instead of overshooting.
            if self._thr_walk is None:
                self._thr_walk = self._prev_throttle
            # Disturbance feed-forward: the commander KNOWS when it adds
            # drag. Full flaps ≈ +0.22 of power to hold speed, gear ≈
            # +0.08 — applied the tick they deploy, so the walk starts
            # from roughly the right band and only has to trim, instead
            # of letting the speed sag 12 kts and surging after it.
            if targets.flap_ratio is not None:
                if (self._ff_flap_prev is not None
                        and targets.flap_ratio > self._ff_flap_prev):
                    self._thr_walk = min(1.0, self._thr_walk + 0.22 * (
                        targets.flap_ratio - self._ff_flap_prev))
                self._ff_flap_prev = targets.flap_ratio
            if targets.gear_down is not None:
                if self._ff_gear_prev is False and targets.gear_down:
                    self._thr_walk = min(1.0, self._thr_walk + 0.08)
                self._ff_gear_prev = bool(targets.gear_down)
            if targets.airspeed_kts is not None:
                drive = spd_error * 0.004          # /s per knot
            else:
                vs_now = (telemetry.vs_fpm
                          if not math.isnan(telemetry.vs_fpm) else 0.0)
                drive = alt_error * 0.0004 - vs_now * 0.00012
            self._thr_walk += drive * dt
            self._thr_walk = max(0.0, min(1.0, self._thr_walk))
            throttle_cmd = self._thr_walk
        # The engine is not a switch. Closed-loop throttle (both coupled
        # and classic modes) slews at most 0.5/s — full sweep in 2 s.
        # Explicit ribbon throttle (takeoff full power, flare idle) is
        # exempt: those are commander orders, instant by design.
        if targets.throttle is None and dt > 0:
            max_step = 0.5 * dt
            throttle_cmd = max(self._prev_throttle - max_step,
                               min(self._prev_throttle + max_step, throttle_cmd))

        # Commander-issued throttle ceiling (cruise gentle-accel). Applied
        # AFTER the slew so the cap is hard; the stall floor below still
        # overrides for genuine low-and-slow danger.
        if targets.throttle_max is not None:
            throttle_cmd = min(throttle_cmd, targets.throttle_max)

        # ── STALL FLOOR — LOW and slow only ──────────────────────────
        # Low and slow is the one corner of the energy matrix where
        # throttle is the ONLY fix (flight 20260706_135230 mushed to
        # 68 kts at idle). But HIGH and slow is a SPLIT problem, not a
        # total-energy problem: the fix is pitch DOWN (trade the spare
        # altitude for the missing speed — free), never power. Flight
        # e9398f14 surged to 0.77 throttle at 160 AGL while ABOVE the
        # slope because this floor was altitude-blind — pumping energy
        # into a plane trying to land. Gate: only force power when at
        # or below the target line (alt_error ≥ −50 ft). Above it, the
        # nose owns the recovery. No slew when it fires: stall recovery
        # is the one case where the engine IS a switch.
        # The alt-gate (only power when at/below the line) holds EXCEPT
        # on short final: below 600 ft AGL, slow gets power regardless
        # of the slope. Flight 946a68cb flew final at 70 kts because it
        # was above the line — at that speed with full flaps the
        # elevator ran out of authority (stick pinned +0.25, nose still
        # falling), the plane dove to -1300 fpm, the commit gate rightly
        # refused, and it bounced. Near the ground, airspeed IS the
        # flare; the doctrine yields to physics there.
        low_final = (not math.isnan(telemetry.agl_m)
                     and telemetry.agl_m * 3.28084 < 600.0)
        if (targets.stall_floor_kts is not None
                and not math.isnan(telemetry.airspeed_kts)
                and telemetry.airspeed_kts < targets.stall_floor_kts
                and (alt_error >= -50.0 or low_final)):
            deficit_kts = targets.stall_floor_kts - telemetry.airspeed_kts
            throttle_cmd = max(throttle_cmd, min(1.0, deficit_kts * 0.1))

        throttle_cmd = max(0.0, min(1.0, throttle_cmd))  # hardware truth
        self._prev_throttle = throttle_cmd

        # ── Yaw (ribbon-driven yaw-hold only; no standalone yaw PID) ─
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

        brake_ratio = targets.brake_ratio if targets.brake_ratio is not None else 0.0
        brake_ratio = max(0.0, min(1.0, brake_ratio))  # hardware truth
        gear_down = targets.gear_down if targets.gear_down is not None else True
        flap_ratio = targets.flap_ratio if targets.flap_ratio is not None else 0.0
        flap_ratio = max(0.0, min(1.0, flap_ratio))  # hardware truth

        return Actuators(
            throttle=throttle_cmd,
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=yaw_cmd,
            brake_ratio=brake_ratio,
            gear_down=gear_down,
            flap_ratio=flap_ratio,
        )
