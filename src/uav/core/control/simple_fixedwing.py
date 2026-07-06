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


@dataclass
class ControlGains:
    bank_per_hdg_error: float = BANK_PER_HDG_ERROR
    bank_inner_kp: float = BANK_INNER_KP
    bank_inner_kd: float = BANK_INNER_KD


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

    def compute(self, telemetry: Telemetry, targets: Targets, dt: float) -> Actuators:
        hdg_error = _wrap_deg(targets.heading_deg - telemetry.heading_deg)
        alt_error = targets.altitude_ft - telemetry.altitude_ft
        spd_error = targets.airspeed_kts - telemetry.airspeed_kts

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
            # Sink-rate tracking (flare). The commander orders a vertical
            # speed; pitch arrests the difference. P-only: 500 fpm of
            # error → 0.15 pitch. Altitude is irrelevant in the flare —
            # what breaks a landing is vertical speed at the pavement.
            VS_TO_PITCH_KP = 0.0003
            pitch_cmd = (targets.vs_target_fpm - telemetry.vs_fpm) * VS_TO_PITCH_KP
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
            VS_TO_PITCH_DAMP = 0.00008  # 1000 fpm → 0.08 pitch opposing
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
        else:
            pitch_cmd = self.altitude_pid.update(
                alt_error, dt, measurement=telemetry.altitude_ft,
            )

        # Commander-issued pitch clamp. Nose-up cap is always honored; nose-down
        # cap is opt-in.
        if targets.pitch_limit is not None:
            pitch_cmd = min(pitch_cmd, abs(targets.pitch_limit))
        if targets.pitch_down_limit is not None:
            pitch_cmd = max(pitch_cmd, -abs(targets.pitch_down_limit))
        pitch_cmd = _clamp(pitch_cmd, 1.0)  # hardware truth

        # ── Throttle ─────────────────────────────────────────────────
        if targets.throttle is not None:
            # Explicit throttle from ribbon (e.g. CLIMB full, FLARE idle)
            self._alt_thr_integral = 0.0
            throttle_cmd = targets.throttle
        elif targets.throttle_for_alt:
            # Alt → Throttle (P-only)
            #   alt_err > 0 (below target) → throttle UP
            #   alt_err < 0 (above target) → throttle DOWN
            # Kp = 3 × the altitude_pid Kp (~0.003 per ft of alt error)
            # so 100 ft below target = 0.3 throttle delta from baseline,
            # 300 ft below = saturated to full. Within the throttle's
            # 0–1 range this gives crisp recovery without integral
            # windup carrying over from earlier phases.
            ALT_TO_THROTTLE_KP = 0.003
            ALT_TO_THROTTLE_KI = 0.0001   # 100 ft error → +0.01 throttle/s
            # Baseline: commander-supplied (near idle on glideslope
            # phases, where gravity provides the energy) or the default
            # cruise setting for level flight.
            base = (targets.throttle_base
                    if targets.throttle_base is not None
                    else self.cruise_throttle)
            # Leaky integral trims the standing error P-only leaves when
            # level flight needs more/less than baseline throttle.
            self._alt_thr_integral += alt_error * dt * ALT_TO_THROTTLE_KI
            self._alt_thr_integral *= 0.999  # leak — self-limiting
            # Clamp ±0.30: ±0.15 couldn't span the gap between the
            # baseline and this airframe's true level-flight thrust, so
            # cruise parked ~90 ft off target (flight 20260706_135230).
            self._alt_thr_integral = max(-0.30, min(0.30, self._alt_thr_integral))
            # VS damping: climbing through the target → cut power EARLY,
            # before the alt error flips sign. Rate feedback = the D term
            # the P-only law was missing (see phugoid note above).
            ALT_TO_THROTTLE_VS_DAMP = 0.00015  # 1000 fpm → 0.15 throttle
            vs_thr = telemetry.vs_fpm if not math.isnan(telemetry.vs_fpm) else 0.0
            throttle_cmd = (base + alt_error * ALT_TO_THROTTLE_KP
                            + self._alt_thr_integral
                            - vs_thr * ALT_TO_THROTTLE_VS_DAMP)
        else:
            self._alt_thr_integral = 0.0
            throttle_cmd = self.cruise_throttle + self.airspeed_pid.update(
                spd_error, dt, measurement=telemetry.airspeed_kts,
            )
        # The engine is not a switch. Closed-loop throttle (both coupled
        # and classic modes) slews at most 0.5/s — full sweep in 2 s.
        # Explicit ribbon throttle (takeoff full power, flare idle) is
        # exempt: those are commander orders, instant by design.
        if targets.throttle is None and dt > 0:
            max_step = 0.5 * dt
            throttle_cmd = max(self._prev_throttle - max_step,
                               min(self._prev_throttle + max_step, throttle_cmd))

        # ── STALL FLOOR — overrides everything, including the slew ───
        # Low and slow is the one corner of the energy matrix where
        # throttle is the ONLY fix. Flight 20260706_135230: gear drag at
        # idle above the slope bled 113 → 68 kts while the alt-priority
        # law held throttle at zero — the coupling starved the plane to
        # defend an altitude CEILING. Below the floor, power ramps in
        # proportionally (floor-5 kts → half, floor-10 → full) no matter
        # what the altitude error says. No slew: stall recovery is the
        # one case where the engine IS a switch.
        if (targets.stall_floor_kts is not None
                and not math.isnan(telemetry.airspeed_kts)
                and telemetry.airspeed_kts < targets.stall_floor_kts):
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
