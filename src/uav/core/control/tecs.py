"""
TECS — Total Energy Control System (longitudinal control).

This is how real fixed-wing autopilots (ArduPilot, PX4) control altitude and
airspeed together, and it replaces every hand-written per-phase pitch/throttle
law we used to carry. The insight: pitch and throttle EACH affect BOTH altitude
and speed, so assigning one control to one output oscillates. Reframe in ENERGY
and the coupling vanishes:

    THROTTLE controls TOTAL energy   (height + speed together)
    ELEVATOR controls the BALANCE    (how energy splits between height & speed)

"Too high and too slow" is then not a paradox — total energy is fine, the split
is wrong, so pitch alone fixes it (nose down: trade the spare height for the
missing speed), no power change. That is exactly the corner we kept crashing on.

Units are SI internally (metres, m/s, radians); the integration layer converts
from feet/knots at the boundary. Reference: ArduPilot AP_TECS.cpp.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

G = 9.80665  # m/s^2


@dataclass
class TECSParams:
    # Energy-rate limits (the airframe's real performance envelope).
    clmb_max: float = 10.0      # max climb rate at full throttle, m/s (~2000 fpm)
    sink_max: float = 5.0       # max descent rate, m/s
    sink_min: float = 1.5       # min descent rate (idle glide), m/s

    # Outer error → demanded-rate gains (how hard we chase the setpoint).
    kh: float = 0.30            # 1/s : height error → climb-rate demand
    kv: float = 0.30            # 1/s : speed error → accel demand
    ax_max: float = 2.0         # m/s^2 : accel demand clamp

    # THROTTLE loop (total specific energy).
    thr_cruise: float = 0.45    # trim power that holds level cruise
    thr_min: float = 0.0
    thr_max: float = 1.0
    kff_thr: float = 0.030      # feedforward: throttle per (m^2/s^3) of demanded energy rate
    kp_thr: float = 0.010       # proportional on total-energy-rate error (gentled for real-plane noise)
    ki_thr: float = 0.0012      # integral on total-energy error (trims the offset)
    thr_integ_limit: float = 0.30

    # PITCH loop (energy balance). Output is a pitch-ANGLE demand.
    kp_pitch: float = 0.55      # on balance-energy error
    kd_pitch: float = 0.28      # on balance-energy-rate error (damping)
    ki_pitch: float = 0.030     # integral on balance error
    pitch_integ_limit: float = 0.20   # rad
    pitch_max_deg: float = 15.0
    pitch_min_deg: float = -12.0

    # Speed/height weighting for the PITCH loop: 0 = pitch defends height only,
    # 2 = pitch defends speed only, 1 = balanced. On approach we bias toward
    # speed (the nose flies airspeed); in cruise, balanced.
    spdweight: float = 1.0


class TECS:
    """One controller for every phase. Feed it (altitude, speed) demands; it
    returns a throttle (0..1) and a pitch-angle demand (degrees) for the
    attitude inner loop to fly."""

    def __init__(self, params: TECSParams | None = None) -> None:
        self.p = params or TECSParams()
        self._thr_integ = 0.0
        self._pitch_integ = 0.0
        # Filtered rates (measured rates are noisy; the loop needs them clean).
        self._hdot_filt = 0.0
        self._vdot_filt = 0.0

    def reset(self) -> None:
        self._thr_integ = 0.0
        self._pitch_integ = 0.0
        self._hdot_filt = 0.0
        self._vdot_filt = 0.0

    def update(
        self,
        h: float, V: float, hdot: float, vdot: float,
        h_dmd: float, V_dmd: float, dt: float,
        *, spdweight: float | None = None, thr_cruise: float | None = None,
    ) -> tuple[float, float]:
        """All SI. Returns (throttle 0..1, pitch_demand_degrees)."""
        p = self.p
        if dt <= 0:
            return p.thr_cruise, 0.0
        w_ske = p.spdweight if spdweight is None else spdweight   # speed weight
        w_ske = max(0.0, min(2.0, w_ske))
        w_spe = 2.0 - w_ske                                       # height weight
        thr_trim = p.thr_cruise if thr_cruise is None else thr_cruise
        V = max(V, 1.0)  # guard the divisions

        # Heavier filtering of the measured rates (0.9 s): on the real plane
        # the finite-difference acceleration is noisy, and it feeds the
        # total-energy-RATE term that drives throttle — unfiltered it made
        # the power bang full↔idle. Filter hard; the feedforward + integral
        # carry the response, the P term just trims.
        a = min(1.0, dt / 0.9)
        self._hdot_filt += a * (hdot - self._hdot_filt)
        self._vdot_filt += a * (vdot - self._vdot_filt)
        hdot_f, vdot_f = self._hdot_filt, self._vdot_filt

        # ── Outer: setpoint error → demanded RATES (bounded to the envelope)
        hdot_dem = max(-p.sink_max, min(p.clmb_max, (h_dmd - h) * p.kh))
        vdot_dem = max(-p.ax_max, min(p.ax_max, (V_dmd - V) * p.kv))

        # ── Specific energies and their rates (mass drops out)
        # potential: g*h  | kinetic: V^2/2   (units m^2/s^2)
        SPE_err = G * (h_dmd - h)
        SKE_err = 0.5 * (V_dmd * V_dmd - V * V)
        SPEdot_dem = G * hdot_dem
        SKEdot_dem = V * vdot_dem
        SPEdot = G * hdot_f
        SKEdot = V * vdot_f

        # ── THROTTLE ← TOTAL energy (potential + kinetic)
        STE_err = SPE_err + SKE_err                 # total energy error (position)
        STEdot_dem = SPEdot_dem + SKEdot_dem        # demanded total energy rate
        STEdot = SPEdot + SKEdot                    # actual
        STEdot_err = STEdot_dem - STEdot
        self._thr_integ += STE_err * p.ki_thr * dt
        self._thr_integ = max(-p.thr_integ_limit,
                              min(p.thr_integ_limit, self._thr_integ))
        throttle = (thr_trim
                    + STEdot_dem * p.kff_thr        # feedforward the demand
                    + STEdot_err * p.kp_thr         # correct the error
                    + self._thr_integ)
        # Anti-windup: only integrate while not saturated.
        if throttle > p.thr_max or throttle < p.thr_min:
            self._thr_integ -= STE_err * p.ki_thr * dt
        throttle = max(p.thr_min, min(p.thr_max, throttle))

        # ── PITCH ← energy BALANCE (weighted split of height vs speed)
        SEB_err = SPE_err * w_spe - SKE_err * w_ske
        SEBdot_dem = SPEdot_dem * w_spe - SKEdot_dem * w_ske
        SEBdot = SPEdot * w_spe - SKEdot * w_ske
        SEBdot_err = SEBdot_dem - SEBdot
        self._pitch_integ += SEB_err * p.ki_pitch * dt
        self._pitch_integ = max(-p.pitch_integ_limit,
                                min(p.pitch_integ_limit, self._pitch_integ))
        # Energy-balance (m^2/s^2 & m^2/s^3) → a flight-path/pitch angle by
        # normalising with (g*V): d(specific energy)/d(angle) ≈ g*V.
        pitch_rad = (SEB_err * p.kp_pitch
                     + SEBdot_err * p.kd_pitch
                     + self._pitch_integ) / (G * V)
        pitch_deg = math.degrees(pitch_rad)
        pitch_deg = max(p.pitch_min_deg, min(p.pitch_max_deg, pitch_deg))
        return throttle, pitch_deg
