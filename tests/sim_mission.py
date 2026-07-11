"""
Offline INTEGRATED-controller mission test.

Runs the real TECSController (TECS + attitude inner loop + surface clamps)
through the point-mass King Air across a full vertical mission — climb, cruise,
descent, and the decelerate-to-approach corner — and checks it stays smooth.
This is the "watch a whole flight succeed in a file" bench, so a bad change
crashes here, never on the plane.

Run:  PYTHONPATH=src python3 tests/sim_mission.py
"""
from __future__ import annotations
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from sim_longitudinal import KingAir, FT_TO_M, MS_TO_KT           # noqa: E402
from uav.core.control.tecs_controller import TECSController        # noqa: E402
from uav.sim.types import Telemetry, Targets                       # noqa: E402


def _tgt(alt_ft, spd_kt, pitch_limit=0.35, pitch_down_limit=0.35,
         throttle=None, vs=None):
    return Targets(
        heading_deg=0.0, altitude_ft=alt_ft, airspeed_kts=spd_kt,
        throttle=throttle, roll_limit=0.4, pitch_limit=pitch_limit,
        pitch_down_limit=pitch_down_limit, vs_target_fpm=vs, gear_down=True,
        flap_ratio=0.0)


def run_mission():
    dt = 0.05
    ac = KingAir(h=460.0, V=62.0)     # ~1500 ft, 120 kt, just airborne
    ctl = TECSController(cruise_throttle=0.45)
    # schedule of TARGET (altitude_ft, speed_kt); the emitted demand ramps
    # toward these at the engine's continuity rates (alt ≤12 ft/s ≈720 fpm,
    # like a glideslope; speed ≤2.5 kt/s), so the demand never steps.
    sched = [
        (0,   4500, 120),   # climb to cruise
        (60,  4500, 175),   # accelerate to cruise speed
        (100, 4500, 175),   # hold cruise
        (130, 1600, 150),   # descend (demand ramps down like a slope)
        (200, 1600, 110),   # decelerate to approach speed
    ]
    T = 260.0
    trace = []
    tgt_alt, tgt_spd = sched[0][1], sched[0][2]
    alt_ft = ac.h / FT_TO_M
    spd_kt = ac.V * MS_TO_KT
    for i in range(int(T / dt)):
        t = i * dt
        for (ts, a, s) in sched:
            if t >= ts:
                tgt_alt, tgt_spd = a, s
        # rate-limit the emitted demand (the real ribbon's continuity ramps):
        # altitude ≤12 ft/s (≈720 fpm, glideslope-like), speed ≤2.5 kt/s.
        alt_ft += max(-12.0 * dt, min(12.0 * dt, tgt_alt - alt_ft))
        spd_kt += max(-2.5 * dt, min(2.5 * dt, tgt_spd - spd_kt))
        hdot_fpm = ac.V * math.sin(ac.gamma) / FT_TO_M * 60.0
        tel = Telemetry(
            airspeed_kts=ac.V * MS_TO_KT, altitude_ft=ac.h / FT_TO_M,
            pitch_deg=math.degrees(ac.theta), roll_deg=0.0, heading_deg=0.0,
            timestamp=t, vs_fpm=hdot_fpm, agl_m=ac.h - 400.0, groundspeed_kts=ac.V * MS_TO_KT)
        act = ctl.compute(tel, _tgt(alt_ft, spd_kt), dt)
        ac.step(act.throttle, act.pitch, dt)
        trace.append(dict(t=t, alt=ac.h / FT_TO_M, spd=ac.V * MS_TO_KT,
                          theta=math.degrees(ac.theta), thr=act.throttle,
                          elev=act.pitch, alt_dmd=alt_ft, spd_dmd=spd_kt))

    print(f"{'t':>5}{'phase':>10}{'alt':>7}{'a_dmd':>7}{'spd':>7}{'s_dmd':>7}"
          f"{'θ°':>6}{'thr':>6}{'elev':>6}")
    labels = {0: 'CLIMB', 55: 'LEVEL', 95: 'CRUISE', 120: 'DESCENT', 175: 'DECEL'}
    for r in trace[::int(4.0 / dt)]:
        ph = ''
        for ts, lab in labels.items():
            if r['t'] >= ts:
                ph = lab
        print(f"{r['t']:5.0f}{ph:>10}{r['alt']:7.0f}{r['alt_dmd']:7.0f}"
              f"{r['spd']:7.0f}{r['spd_dmd']:7.0f}{r['theta']:6.1f}"
              f"{r['thr']:6.2f}{r['elev']:+6.2f}")

    # smoothness: elevator reversals (jitter) + settle errors per phase tail
    def phase_tail(lo, hi):
        seg = [r for r in trace if lo <= r['t'] < hi]
        return seg[-int(8 / dt):] if len(seg) > int(8 / dt) else seg
    print("\nsettle (last 8 s of each phase):")
    for (ts, a, s), nxt in zip(sched, [x[0] for x in sched[1:]] + [T]):
        tail = phase_tail(ts, nxt)
        if not tail:
            continue
        ae = max(abs(r['alt_dmd'] - r['alt']) for r in tail)
        se = max(abs(r['spd_dmd'] - r['spd']) for r in tail)
        # elevator jitter: mean |Δelev| per tick
        jit = sum(abs(b['elev'] - a2['elev']) for a2, b in zip(tail, tail[1:])) / max(1, len(tail) - 1)
        print(f"  {labels.get(ts,''):8}  alt_err≤{ae:5.0f} ft  spd_err≤{se:4.0f} kt  "
              f"elev_jitter={jit:.4f} {'SMOOTH' if jit < 0.02 else 'JITTERY'}")


if __name__ == "__main__":
    run_mission()
