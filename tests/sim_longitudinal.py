"""
Offline longitudinal aircraft sim — the test bench we should have had all along.

A point-mass King Air with real-ish aerodynamics (lift/drag polar, thrust,
pitch dynamics). We fly TECS + the pitch attitude loop through it and watch
altitude and airspeed respond IN A FILE. Nothing here touches X-Plane. This is
where a control change is allowed to oscillate or crash — never on the plane.

Run:  PYTHONPATH=src python3 tests/sim_longitudinal.py
"""
from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from uav.core.control.tecs import TECS, TECSParams, G          # noqa: E402
from uav.core.control.attitude import default_pitch_axis        # noqa: E402

MS_TO_KT = 1.94384
FT_TO_M = 0.3048


class KingAir:
    """Point-mass longitudinal model. State: h, V, gamma (flight-path angle),
    theta (pitch attitude), q (pitch rate). Controls: throttle 0..1, elevator."""

    # Airframe constants — ANCHORED to the real X-Plane C90B.acf:
    #   mass 10,100 lb MTOW (use ~9,500 lb typical), wing 279 ft², 2×550 shp,
    #   Vs 100 / Vso 90 kt. Drag polar calibrated so cruise ~90 m/s (175 kt)
    #   sits near half power and full power climbs ~2000 fpm — matches the
    #   real aircraft. Final trim comes from actual flight telemetry.
    m = 4310.0          # kg (~9,500 lb flight weight)
    S = 25.9            # m^2 (279 ft² published wing area)
    rho = 1.05          # kg/m^3 (few-thousand-ft density)
    CD0 = 0.028         # parasite drag
    k = 0.050           # induced drag factor
    CL_alpha = 5.5      # per rad
    CL0 = 0.25          # lift at zero alpha
    # Turboprop thrust ≈ shaft-power / speed (NOT constant): strong at low
    # speed (climb), tapering with speed. Capped at a static-thrust limit.
    P_max = 820000.0    # W (2 × 550 shp)
    eta_prop = 0.80     # propeller efficiency
    T_static = 16000.0  # N low-speed static-thrust cap
    # Pitch dynamics: elevator → pitch angular accel, damped.
    elev_power = 6.0    # rad/s^2 per unit elevator at ref q-bar
    q_damp = 2.2        # pitch-rate damping (1/s)
    ref_q = 0.5 * 1.05 * 90.0 ** 2   # reference dynamic pressure for scaling

    def __init__(self, h=1524.0, V=90.0):
        self.h = h          # m
        self.V = V          # m/s
        self.gamma = 0.0    # rad (flight path)
        self.theta = 0.03   # rad (pitch attitude ~1.7°)
        self.q = 0.0        # rad/s
        self._prev_V = V

    def step(self, throttle: float, elevator: float, dt: float):
        g = G
        alpha = self.theta - self.gamma
        qbar = 0.5 * self.rho * self.V * self.V
        CL = self.CL0 + self.CL_alpha * alpha
        CD = self.CD0 + self.k * CL * CL
        L = qbar * self.S * CL
        D = qbar * self.S * CD
        # Turboprop: thrust = power/speed, capped at static thrust.
        T = min(self.T_static,
                throttle * self.P_max * self.eta_prop / max(self.V, 25.0))

        # Translational dynamics
        Vdot = (T - D) / self.m - g * math.sin(self.gamma)
        gammadot = (L - self.m * g * math.cos(self.gamma)) / (self.m * max(self.V, 1.0))
        hdot = self.V * math.sin(self.gamma)

        # Pitch rotational dynamics (elevator authority scales with q-bar)
        qdot = (self.elev_power * (qbar / self.ref_q) * elevator
                - self.q_damp * self.q)

        # Integrate
        self.V += Vdot * dt
        self.gamma += gammadot * dt
        self.h += hdot * dt
        self.q += qdot * dt
        self.theta += self.q * dt
        self.V = max(self.V, 20.0)   # never model below a crawl
        return hdot, Vdot


def run(scenario_name, phases, params=None, T=120.0, dt=0.05, h0=1524.0, V0=90.0):
    """phases: list of (start_time_s, h_dmd_m, V_dmd_ms). Returns trace."""
    ac = KingAir(h=h0, V=V0)
    tecs = TECS(params or TECSParams())
    pitch = default_pitch_axis()
    prev_V = ac.V
    trace = []
    h_dmd, V_dmd = h0, V0
    for i in range(int(T / dt)):
        t = i * dt
        for (ts, hd, vd) in phases:
            if t >= ts:
                h_dmd, V_dmd = hd, vd
        # Estimate rates (as the real system would, from telemetry deltas)
        hdot = ac.V * math.sin(ac.gamma)
        vdot = (ac.V - prev_V) / dt
        prev_V = ac.V
        # Controllers
        throttle, pitch_dmd_deg = tecs.update(
            ac.h, ac.V, hdot, vdot, h_dmd, V_dmd, dt)
        elevator = pitch.update(pitch_dmd_deg, math.degrees(ac.theta),
                                ac.V * MS_TO_KT, dt)
        ac.step(throttle, elevator, dt)
        trace.append(dict(t=t, h=ac.h, V=ac.V, hdot=hdot,
                          theta=math.degrees(ac.theta), thr=throttle,
                          pitch_dmd=pitch_dmd_deg, elev=elevator,
                          h_dmd=h_dmd, V_dmd=V_dmd))
    _report(scenario_name, trace)
    return trace


def _report(name, tr):
    print(f"\n=== {name} ===")
    print(f"{'t':>5}{'alt_ft':>8}{'spd_kt':>8}{'θ°':>6}{'thr':>6}{'pd°':>6}"
          f"{'alt_err':>8}{'spd_err':>8}")
    n = len(tr)
    for r in tr[::max(1, n // 16)]:
        print(f"{r['t']:5.0f}{r['h']/FT_TO_M:8.0f}{r['V']*MS_TO_KT:8.1f}"
              f"{r['theta']:6.1f}{r['thr']:6.2f}{r['pitch_dmd']:6.1f}"
              f"{(r['h_dmd']-r['h'])/FT_TO_M:+8.0f}{(r['V_dmd']-r['V'])*MS_TO_KT:+8.1f}")
    # settle metrics over the last 20 s
    tail = [r for r in tr if r['t'] > tr[-1]['t'] - 20]
    alt_err = [abs(r['h_dmd'] - r['h']) / FT_TO_M for r in tail]
    spd_err = [abs(r['V_dmd'] - r['V']) * MS_TO_KT for r in tail]
    # oscillation proxy: sign changes of (alt - alt_dmd) over the last 40 s
    late = [r['h'] - r['h_dmd'] for r in tr if r['t'] > tr[-1]['t'] - 40]
    osc = sum(1 for a, b in zip(late, late[1:]) if (a > 0) != (b > 0))
    print(f"  settle: |alt_err|≈{max(alt_err):.0f} ft  |spd_err|≈{max(spd_err):.1f} kt"
          f"  alt-crossings(40s)={osc}")


if __name__ == "__main__":
    ft = lambda x: x * FT_TO_M
    kt = lambda x: x / MS_TO_KT
    # 1. Hold cruise: level 5000 ft, 175 kt — does it just sit there?
    run("HOLD cruise 5000ft / 175kt",
        [(0, ft(5000), kt(175))], h0=ft(5000), V0=kt(175))
    # 2. Climb 1000 ft, hold speed.
    run("CLIMB 5000→6000 ft @ 175kt",
        [(0, ft(5000), kt(175)), (5, ft(6000), kt(175))], h0=ft(5000), V0=kt(175))
    # 3. Descend 1000 ft.
    run("DESCEND 5000→4000 ft @ 150kt",
        [(0, ft(5000), kt(175)), (5, ft(4000), kt(150))], h0=ft(5000), V0=kt(175))
    # 4. Slow down at constant altitude (the "too high too slow" corner test).
    run("DECEL 175→120 kt @ 5000ft",
        [(0, ft(5000), kt(175)), (5, ft(5000), kt(120))], h0=ft(5000), V0=kt(175))
