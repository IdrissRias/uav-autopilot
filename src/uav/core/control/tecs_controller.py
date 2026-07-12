"""
TECSController — one energy law for the whole flight envelope.

Longitudinal control is a single Total-Energy law (throttle → total energy,
elevator → energy balance) applied to EVERY energy phase: climb, cruise,
descent, approach, decelerate. It chases two RESULTS handed over by the ribbon,
target altitude and target airspeed, and drives the two inputs (throttle,
elevator) freely to null the energy errors. There is no per-phase control
policy, no step-and-check, no throttle slew, no elevator trim-hold, no pitch
policy cap. Inputs are bounded only by physics (0..1 throttle, ±1 surface) and
by three RESULT-based envelope floors that bite only at the edges: stall,
terrain sink, and overspeed. (Idriss doctrine, 2026-07-12: "result controlled,
not input controlled — we chase results and adjust inputs.")

Three phases are genuinely NOT energy tracking and stay explicit: ground/
takeoff (throttle commanded, nose eases up to rotate), flare (idle + gentle
nose-up), and rollout (derotate). Those are entered by the ribbon handing an
explicit throttle; every throttle-None phase is the one energy law. Lateral
(heading→bank→aileron) and yaw ground-steering are the proven cascades, kept
verbatim.

ALTITUDE PRIORITY via an asymmetric band: never below target, up to +BAND_UP
above is fine. The plane rides at-or-above the (possibly descending) target and
never dives through it. The band tapers to zero near the ground so the flare
lands on the numbers.

Drop-in for SimpleFixedWingController: compute(telemetry, targets, dt).
"""
from __future__ import annotations

import math

from uav.core.control.base import Controller
from uav.core.control.pid import PID
from uav.core.control.simple_fixedwing import ControlGains, _wrap_deg, _clamp
from uav.core.control.tecs import TECS, TECSParams
from uav.sim.types import Telemetry, Actuators, Targets

FT_TO_M = 0.3048
MS_TO_KT = 1.94384
FPM_TO_MS = 0.00508   # ft/min → m/s

# Flare = hold ONE gentle nose-up attitude and let her settle, idle power.
FLARE_PITCH_DEG = 5.0

# Altitude band (altitude priority). Never below target; tolerate up to this
# far above. The band is what removes the dive-through-the-line overshoot: the
# plane only has to reach "within BAND_UP above", never an exact line, so there
# is no aggressive dive to overshoot.
ALT_BAND_UP_FT = 200.0
BAND_TAPER_FROM_FT = 30.0   # band shrinks to 0 by this AGL (land on the numbers)
BAND_TAPER_SLOPE = 0.5      # ft of band per ft of AGL above the taper floor

# ── DESCENT / APPROACH: attitude-hold + power-for-altitude (Idriss doctrine,
# 2026-07-12). The nose does NOT dive to chase the glideslope. A plane on a 3°
# path sits nose-UP (flight path = pitch - AoA), so we HOLD a gentle scheduled
# nose-up attitude, make THROTTLE the altitude servo (pull power / idle to sink,
# add to hold), let FLAPS be the drag brake, and let speed be emergent. The nose
# eases toward level only as a last resort (already at idle and still above the
# band). This kills the dive-and-PIO by construction: pitch barely moves.
# First-pass numbers, to be tuned from flight logs.
GS_THETA_UP_HI = 1.5      # held nose-up attitude (deg), high/fast on descent
GS_THETA_UP_LO = 4.5      # held nose-up attitude near the flare (slower, more AoA)
GS_THETA_MIN_HI = -2.5    # lowest the nose may ease to, high up (last-resort sink)
GS_THETA_MIN_LO = 0.0     # near the ground the nose may never point down
GS_SCHED_HI_AGL = 800.0   # schedule endpoints (ft AGL): full nose-up-schedule
GS_SCHED_LO_AGL = 80.0    #   blends between these
GS_THR_TRIM = 0.22        # nominal power that holds the path inside the band
GS_THR_KP = 0.004         # throttle per ft of altitude error (power = altitude)
GS_PITCH_ASSIST = 0.006   # deg nose-down per ft ABOVE the band (last resort)

# AoA envelope — defend ANGLE OF ATTACK at all times (the real stall variable;
# stall is an AoA event, not a speed event). Above this, unload (nose down toward
# the limit) and add power. Replaces the crude speed-based stall floor.
ALPHA_MAX_DEG = 12.0      # margin below the ~15-16° critical AoA

# Energy-balance weighting for the pitch loop (TECS spdweight). The pitch loop
# splits into two weights that sum to 2: speed weight = SPDWEIGHT, altitude
# weight = 2 - SPDWEIGHT. Lower SPDWEIGHT = the ELEVATOR defends ALTITUDE harder
# (throttle then carries speed). Idriss doctrine is altitude-first ("never below
# target"), so weight it hard toward altitude:
#   0.25 -> altitude weight 1.75, speed weight 0.25  (7:1 toward altitude)
# The stall floor still protects the low-speed edge, so starving the pitch loop
# of speed authority is safe.
SPDWEIGHT = 0.25

# Envelope floors (result-based safety). Stall floor is a CATASTROPHIC backstop
# ONLY. The energy loop actively chases the ribbon's target speed (a safe
# cruise/approach speed), so in normal flight the plane never approaches stall
# and the floor would only FIGHT the loop — which is exactly what the old
# v_land-based guard did, slamming full throttle through every on-speed approach
# and shoving the plane up above the glideslope. So it fires only below an
# absolute speed far under any real stall (flap-stall ~90 kt): a true last-ditch
# net that never acts in normal flight. (Idriss, 2026-07-12: "the plane would
# never let itself stall, it's on a control loop.")
STALL_FLOOR_ABS_KTS = 30.0   # fire only below this — catastrophic backstop
TERRAIN_SINK_PER_FT = 8.0    # max allowed sink (fpm) per ft AGL (tight near ground)
TERRAIN_SINK_MIN_FPM = 100.0 # never tighter than this
GLOBAL_SINK_MAX_FPM = 1500.0 # absolute sink ceiling at any height — the floor
                             # engages early (not only near the ground) so a bad
                             # target can't build an unarrestable dive up high

# Throttle spool rate. The engine cannot (and should not) jump full-range in a
# tick — a real turboprop spools over seconds. The command WALKS toward whatever
# the energy law / floors ask for at this rate, which both models the physics and
# kills the throttle slamming 0<->1 (power-band doctrine: "slow power walks, never
# jockey"). 0.25/s ≈ 0.01 per 25 Hz tick — full range in ~4 s. dt-capped so a
# stalled tick can't sneak a big jump through.
THROTTLE_SLEW_PER_S = 0.25

# Pitch attitude inner loop (soft spring + rate damper + slow self-trim integral
# with anti-windup). Detuned for X-Plane (2026-07-12): spring cut (KP 0.05->0.03)
# so the elevator eases toward the demanded attitude instead of slamming to full
# on a big error; damper (KD) kept, which raises the damping ratio and calms the
# PIO. The pitch-rate signal is filtered harder below so KD stops chattering on
# X-Plane's noisy 25 Hz rate.
PITCH_KP = 0.03
PITCH_KI = 0.01
PITCH_KD = 0.055
PITCH_I_LIMIT = 0.5


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
        # Pitch attitude inner loop state.
        self._prev_pitch_deg: float | None = None
        self._pitch_rate_filt = 0.0
        self._pitch_integ = 0.0
        # lateral state
        self._prev_bank_deg = 0.0
        self._prev_hdg_deg: float | None = None
        # yaw ground-steering step-and-check state
        self._yaw_cmd_held = 0.0
        self._yaw_wait = 0.0
        # longitudinal state
        self._prev_V_ms: float | None = None
        self._prev_throttle = 0.0

    # ── helpers ──────────────────────────────────────────────────────
    def _airspeed_ms(self, tel: Telemetry) -> float:
        v = tel.airspeed_kts
        if math.isnan(v):
            v = 0.0
        return max(v, 0.0) / MS_TO_KT

    def _band_up_ft(self, agl_ft: float) -> float:
        """The +altitude band width, tapering to 0 near the ground so the flare
        lands on target."""
        if math.isnan(agl_ft):
            return ALT_BAND_UP_FT
        return max(0.0, min(ALT_BAND_UP_FT,
                            (agl_ft - BAND_TAPER_FROM_FT) * BAND_TAPER_SLOPE))

    def _alt_band_hdmd_ft(self, target_ft: float, h_ft: float,
                          agl_ft: float) -> float:
        """Asymmetric altitude band → the height demand fed to TECS (cruise/climb).
        Below target → demand target (never below); inside the band → demand h
        (no push); above the band → demand the top of the band (ease down)."""
        band = self._band_up_ft(agl_ft)
        if h_ft < target_ft:
            return target_ft
        if h_ft <= target_ft + band:
            return h_ft
        return target_ft + band

    def _sched_frac(self, agl_ft: float) -> float:
        """0 high up → 1 near the ground: blends the descent attitude schedule."""
        if math.isnan(agl_ft) or agl_ft >= GS_SCHED_HI_AGL:
            return 0.0
        if agl_ft <= GS_SCHED_LO_AGL:
            return 1.0
        return (GS_SCHED_HI_AGL - agl_ft) / (GS_SCHED_HI_AGL - GS_SCHED_LO_AGL)

    def _glideslope_law(self, tel: Telemetry, targets: Targets,
                        agl_ft: float) -> tuple[float, float]:
        """DESCENT / APPROACH: hold a gentle nose-up attitude, power for altitude,
        flaps for drag, speed emergent. Returns (throttle_target, pitch_deg).
        The throttle target is walked by the outer spool-rate limiter."""
        alt = (tel.altitude_ft if not math.isnan(tel.altitude_ft)
               else targets.altitude_ft)
        target = targets.altitude_ft
        band = self._band_up_ft(agl_ft)
        # altitude error against the band: + below target (add power / climb back),
        # 0 inside the band, - above the band (idle / sink).
        if alt < target:
            e = target - alt
        elif alt <= target + band:
            e = 0.0
        else:
            e = (target + band) - alt   # negative
        # THROTTLE is the altitude servo. Power = altitude.
        thr = GS_THR_TRIM + GS_THR_KP * e
        # PITCH holds a scheduled gentle nose-up attitude; it eases toward the
        # (bounded) floor ONLY when above the band, and never dives.
        frac = self._sched_frac(agl_ft)
        theta_up = GS_THETA_UP_HI + (GS_THETA_UP_LO - GS_THETA_UP_HI) * frac
        theta_min = GS_THETA_MIN_HI + (GS_THETA_MIN_LO - GS_THETA_MIN_HI) * frac
        theta = theta_up
        if e < 0.0:
            theta = max(theta_min, theta_up + e * GS_PITCH_ASSIST)
        return max(0.0, min(1.0, thr)), theta

    def _envelope_floors(self, thr: float, pitch_deg: float, V_kts: float,
                         vs_fpm: float, agl_ft: float, vmax_kts: float | None,
                         alpha_deg: float, pitch_now_deg: float
                         ) -> tuple[float, float]:
        """Result-based safety, applied in priority order (stall wins, so it is
        applied LAST). These only ever act at the edges of the envelope."""
        if math.isnan(V_kts):
            return thr, pitch_deg
        sink = -vs_fpm if not math.isnan(vs_fpm) else 0.0   # +ve = descending

        # Overspeed (lowest priority): near the never-exceed / flap limit, do
        # not push the nose down further and cut power. Trades speed for height.
        if vmax_kts is not None and V_kts > vmax_kts:
            pitch_deg = max(pitch_deg, 1.0)
            thr = 0.0

        # Terrain / dive floor ("regard for the ground"): sink is capped by the
        # tighter of an AGL-proportional limit (tight near ground) and a global
        # ceiling (engages early at any height). When exceeded, command nose-up
        # proportional to the overshoot. The inner loop already slams full
        # nose-up elevator on a big demand, so the cap is really about engaging
        # EARLY enough to arrest before the ground, not about the number.
        agl_cap = (max(TERRAIN_SINK_MIN_FPM, agl_ft * TERRAIN_SINK_PER_FT)
                   if not math.isnan(agl_ft) else GLOBAL_SINK_MAX_FPM)
        max_sink = min(agl_cap, GLOBAL_SINK_MAX_FPM)
        # Soft engagement: start easing the nose up at 75% of the cap so the
        # arrest anticipates the limit instead of overshooting it (reacting to
        # actual sink lags; a hard trip point rings past the ceiling).
        soft = 0.75 * max_sink
        if sink > soft:
            pitch_deg = max(pitch_deg, min(14.0, (sink - soft) * 0.03))

        # STALL = ANGLE OF ATTACK, not speed (highest priority, applied last).
        # Defend critical AoA at all times: above the limit, unload by commanding
        # the nose down just enough to bring AoA back to the limit, and add full
        # power. Works in any maneuver, not just 1g. A very-low speed backstop
        # remains only for when AoA is unavailable (NaN).
        if not math.isnan(alpha_deg):
            if alpha_deg > ALPHA_MAX_DEG:
                pitch_deg = min(pitch_deg,
                                pitch_now_deg - (alpha_deg - ALPHA_MAX_DEG))
                thr = 1.0
        elif V_kts < STALL_FLOOR_ABS_KTS:
            pitch_deg = min(pitch_deg, 0.0)
            thr = 1.0

        return max(0.0, min(1.0, thr)), pitch_deg

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

        agl_ft = (telemetry.agl_m * 3.28084
                  if not math.isnan(telemetry.agl_m) else float('nan'))

        pitch_dmd_deg = 0.0
        energy_phase = False

        if targets.throttle is not None and targets.vs_target_fpm is not None:
            # FLARE / ROLLOUT: explicit idle, hold ONE steady flare attitude
            # (the inner loop eases the nose to it). She decelerates and settles
            # onto the mains instead of hunting.
            throttle_cmd = targets.throttle
            pitch_dmd_deg = FLARE_PITCH_DEG
            self.tecs.reset()

        elif targets.throttle is not None:
            # GROUND / TAKEOFF: honour the throttle; ease the nose toward a
            # gentle climb attitude scaled by how far below target we are, so
            # she rotates at Vr. Bounded so it can't yank.
            throttle_cmd = targets.throttle
            if targets.altitude_ft is not None:
                below = (targets.altitude_ft - telemetry.altitude_ft)
                pitch_dmd_deg = max(0.0, min(8.0, below * 0.004))
            self.tecs.reset()

        elif targets.altitude_ft is not None and getattr(targets, "on_glideslope", False):
            # ── DESCENT / APPROACH: attitude-hold + power-for-altitude.
            # Hold a gentle nose-up attitude, throttle is the altitude servo,
            # flaps are the drag brake, speed is emergent. The nose does not dive.
            energy_phase = True
            throttle_cmd, pitch_dmd_deg = self._glideslope_law(
                telemetry, targets, agl_ft)
            self.tecs.reset()   # keep TECS integrators fresh for the next cruise

        elif targets.altitude_ft is not None and targets.airspeed_kts is not None:
            # ── TECS ENERGY LAW (climb / cruise / decelerate).
            # Chase the altitude BAND and the target speed with throttle (total
            # energy) and elevator (balance). Descent/approach are handled above
            # by the attitude-hold law instead, since TECS's energy-balance pitch
            # dives to catch the slope and PIO'd on the real short-period.
            energy_phase = True
            target_ft = targets.altitude_ft
            h_dmd_ft = self._alt_band_hdmd_ft(target_ft, telemetry.altitude_ft
                                              if not math.isnan(telemetry.altitude_ft)
                                              else target_ft, agl_ft)
            V_dmd = targets.airspeed_kts / MS_TO_KT
            throttle_cmd, pitch_dmd_deg = self.tecs.update(
                h, V, hdot, vdot, h_dmd_ft * FT_TO_M, V_dmd, dt, spdweight=SPDWEIGHT)

        else:
            # No demands (shouldn't happen) — hold last throttle, level attitude.
            throttle_cmd = self._prev_throttle
            pitch_dmd_deg = 2.0

        # ── ENVELOPE FLOORS (result-based safety; energy phases only) ────
        if energy_phase:
            pitch_now = (telemetry.pitch_deg
                         if not math.isnan(telemetry.pitch_deg) else 0.0)
            throttle_cmd, pitch_dmd_deg = self._envelope_floors(
                throttle_cmd, pitch_dmd_deg, telemetry.airspeed_kts,
                telemetry.vs_fpm, agl_ft, getattr(targets, "v_max_kts", None),
                telemetry.alpha_deg, pitch_now)

        throttle_cmd = max(0.0, min(1.0, throttle_cmd))
        # Spool-rate walk (energy phases only; ground/takeoff/flare are explicit
        # immediate commands and pass through). The engine can't slam; power
        # walks toward the demand at THROTTLE_SLEW_PER_S. dt-capped so a stalled
        # tick can't jump.
        if energy_phase and dt > 0:
            step = THROTTLE_SLEW_PER_S * min(dt, 0.1)
            throttle_cmd = max(self._prev_throttle - step,
                               min(self._prev_throttle + step, throttle_cmd))
        self._prev_throttle = throttle_cmd

        # ── PITCH ATTITUDE INNER LOOP (PI+D, self-trimming, anti-windup) ──
        # Soft spring + rate damper (X-Plane-proven) plus a slow trim integral
        # that holds the demanded attitude without the ad-hoc "never let go"
        # hack. Anti-windup: stop integrating while the surface is saturated.
        pdeg = telemetry.pitch_deg if not math.isnan(telemetry.pitch_deg) else 0.0
        if self._prev_pitch_deg is not None and dt > 0:
            raw = (pdeg - self._prev_pitch_deg) / dt
            # Heavier filter (0.3/0.7) so the KD damper doesn't chatter on
            # X-Plane's noisy 25 Hz pitch-rate.
            self._pitch_rate_filt = 0.3 * raw + 0.7 * self._pitch_rate_filt
        self._prev_pitch_deg = pdeg
        err = pitch_dmd_deg - pdeg
        self._pitch_integ += PITCH_KI * err * dt
        self._pitch_integ = max(-PITCH_I_LIMIT, min(PITCH_I_LIMIT, self._pitch_integ))
        pitch_cmd = PITCH_KP * err + self._pitch_integ - PITCH_KD * self._pitch_rate_filt
        if pitch_cmd > 1.0 or pitch_cmd < -1.0:
            self._pitch_integ -= PITCH_KI * err * dt   # unwind while saturated
        pitch_cmd = _clamp(pitch_cmd, 1.0)

        # ── YAW (ribbon-driven ground-steering): STEP-AND-CHECK, SPEED-SCALED
        # Kept verbatim — this is takeoff-roll / rollout nosewheel steering, a
        # different animal from the longitudinal law. The rudder is aerodynamic,
        # so the SAME deflection turns less as speed drops; the step is scaled
        # by (V/Vref)^2, floored/capped so it can't blow up near a stop.
        YAW_REF_KTS = 60.0
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
                step = YAW_STEP / q_ratio
                if hdg_error > YAW_DEADBAND_DEG:
                    self._yaw_cmd_held = min(yaw_limit, self._yaw_cmd_held + step)
                elif hdg_error < -YAW_DEADBAND_DEG:
                    self._yaw_cmd_held = max(-yaw_limit, self._yaw_cmd_held - step)
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
