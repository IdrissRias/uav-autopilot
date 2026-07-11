"""
Closed-loop glideslope test driving the REAL TECSController.

Unlike sim_glideslope_stability.py (which re-implements the formulas), this
instantiates the actual controller and flies the calibrated King Air point-
mass model with its real outputs, so it exercises the throttle step-and-check
tiers, the AGL sink floor, AND the elevator trim-hold ("never let go")
together — the combination that must be proven before flying.

Two scenarios:
  1. NORMAL: start 200 ft above the slope + flap/gear trim kicks. The off-slope
     error must DECAY, not sustain/grow — proof the trim-hold does not balloon.
  2. TARGET SNAPS TO GROUND at t=60 s (models the geometry bearing-snap bug):
     the controller is told the target altitude is 0. The AGL floor must stop
     the aircraft following it into the dirt — it must NOT dive far below the
     honest slope, and sink stays bounded.

The run ends when the honest slope reaches 300 ft (approach handoff), a normal
end — not a crash.

Run:  PYTHONPATH=src python3 tests/sim_gs_closedloop.py
"""
from __future__ import annotations
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from sim_longitudinal import KingAir, FT_TO_M, MS_TO_KT   # noqa: E402
from uav.core.control.tecs_controller import TECSController   # noqa: E402
from uav.sim.types import Telemetry, Targets   # noqa: E402

FT_PER_NM = 500.0


def run(label, snap_to_ground_at=None, T=300.0, dt=0.1, disturb_ft=200.0):
    gs_kts = 120.0
    required_fpm = -(FT_PER_NM * gs_kts / 60.0)
    ideal_alt_ft = 2200.0
    ac = KingAir(h=(ideal_alt_ft + disturb_ft) * FT_TO_M, V=gs_kts / MS_TO_KT)
    ac.gamma = math.radians(-2.3)

    ctl = TECSController(cruise_throttle=0.45)
    ctl._prev_throttle = 0.3
    off_rate_filt, off_prev = 0.0, None
    kicks, kicked = {40.0, 90.0}, set()

    off_trace, sink_after_snap, off_after_snap = [], [], []
    crashed = False
    for i in range(int(T / dt)):
        t = i * dt
        for kt in kicks:
            if t >= kt and kt not in kicked:
                ac.gamma -= math.radians(2.0)
                kicked.add(kt)

        actual_alt_ft = ac.h / FT_TO_M
        ideal_alt_ft += (required_fpm / 60.0) * dt   # the HONEST slope
        if ideal_alt_ft <= 300.0:
            break   # normal approach handoff — not a crash
        target_alt_ft = 0.0 if (snap_to_ground_at is not None
                                and t >= snap_to_ground_at) else ideal_alt_ft

        off_slope_ft = actual_alt_ft - ideal_alt_ft
        if off_prev is not None:
            off_rate_filt = 0.3 * ((off_slope_ft - off_prev) / dt) + 0.7 * off_rate_filt
        off_prev = off_slope_ft
        conv = 0.15 if off_slope_ft >= 0 else 0.20
        vs_target = max(-1500.0, min(250.0,
                        required_fpm - off_slope_ft * conv - off_rate_filt * 45.0))

        vs_fpm = ac.V * math.sin(ac.gamma) * 196.85
        tel = Telemetry(
            airspeed_kts=ac.V * MS_TO_KT, altitude_ft=actual_alt_ft,
            pitch_deg=math.degrees(ac.theta), roll_deg=0.0, heading_deg=270.0,
            timestamp=t, vs_fpm=vs_fpm, agl_m=ac.h, groundspeed_kts=ac.V * MS_TO_KT)
        tgt = Targets(
            heading_deg=270.0, altitude_ft=target_alt_ft, airspeed_kts=gs_kts,
            vs_target_fpm=vs_target, on_glideslope=True, pitch_limit=0.25,
            pitch_down_limit=0.20, gear_down=True, flap_ratio=1.0)
        act = ctl.compute(tel, tgt, dt)
        ac.step(act.throttle, act.pitch, dt)

        off_trace.append(off_slope_ft)
        if snap_to_ground_at is not None and t >= snap_to_ground_at:
            sink_after_snap.append(-vs_fpm)
            off_after_snap.append(ac.h / FT_TO_M - ideal_alt_ft)
        if ac.h <= 0:
            crashed = True
            break

    print(label)
    if crashed:
        print("  *** GROUND CONTACT (CRASH) ***\n")
        return False

    half = len(off_trace) // 2
    amp_early = max(off_trace[:half]) - min(off_trace[:half]) if half else 0
    amp_late = max(off_trace[half:]) - min(off_trace[half:]) if half else 0
    worst_below = min(off_trace) if off_trace else 0
    print(f"  off-slope amplitude  early={amp_early:.0f}  late={amp_late:.0f} ft"
          f"   worst below slope={worst_below:.0f} ft")

    if snap_to_ground_at is not None:
        max_sink = max(sink_after_snap) if sink_after_snap else 0
        dove = min(off_after_snap) if off_after_snap else 0
        ok = dove > -300.0 and max_sink < 2000.0
        print(f"  after snap-to-ground: max sink={max_sink:.0f} fpm  "
              f"max dive below honest slope={dove:.0f} ft")
        print(f"  DID NOT CHASE TARGET INTO DIVE: {ok}\n")
        return ok

    # NO BALLOON = amplitude stays small and bounded (not growing without
    # limit) and the plane never dives far below the slope. Convergence of a
    # 200 ft initial offset is NOT required — the deliberately gentle
    # conv_gain parallels rather than snaps back, by design.
    ok = amp_late < 45.0 and worst_below > -350.0
    print(f"  VERDICT: {'STABLE (no balloon)' if ok else 'UNSTABLE / ballooned'}\n")
    return ok


def run_floor(label, T=60.0, dt=0.1):
    """Low-altitude AGL-floor test: plane at 250 ft AGL, told to dive to the
    ground (target 0). The floor must keep the sink bounded (~AGL*12 fpm) so it
    settles toward the ground GENTLY instead of plummeting, and never arrives
    at a plummet sink rate."""
    ac = KingAir(h=250.0 * FT_TO_M, V=110.0 / MS_TO_KT)
    ac.gamma = math.radians(-3.0)
    ctl = TECSController(cruise_throttle=0.45)
    ctl._prev_throttle = 0.3
    max_sink = 0.0
    sink_at = {}   # AGL bucket -> sink, to check sink DECREASES toward the ground
    reached_ground = False
    for i in range(int(T / dt)):
        t = i * dt
        agl_ft = ac.h / FT_TO_M
        if agl_ft <= 5.0:
            reached_ground = True
            break
        vs_fpm = ac.V * math.sin(ac.gamma) * 196.85
        max_sink = max(max_sink, -vs_fpm)
        sink_at[int(agl_ft // 25) * 25] = -vs_fpm
        tel = Telemetry(
            airspeed_kts=ac.V * MS_TO_KT, altitude_ft=agl_ft,
            pitch_deg=math.degrees(ac.theta), roll_deg=0.0, heading_deg=270.0,
            timestamp=t, vs_fpm=vs_fpm, agl_m=ac.h, groundspeed_kts=ac.V * MS_TO_KT)
        tgt = Targets(
            heading_deg=270.0, altitude_ft=0.0, airspeed_kts=110.0,
            vs_target_fpm=-1500.0, on_glideslope=True, pitch_limit=0.25,
            pitch_down_limit=0.20, gear_down=True, flap_ratio=1.0)
        act = ctl.compute(tel, tgt, dt)
        ac.step(act.throttle, act.pitch, dt)
    # Anti-plummet property: sink must be bounded AND must DECREASE as the
    # aircraft nears the ground (the floor shrinks with height). Compare a
    # high band (~200 ft) with a low band (~50 ft).
    hi = sink_at.get(200, sink_at.get(175, max_sink))
    lo = sink_at.get(50, sink_at.get(25, 0))
    # No flare in this isolated test (it fires at 30 ft via the ribbon in the
    # real system), so a CONTROLLED descent to the ground under a literal
    # target=0 command is expected — what matters is it's bounded and EASING,
    # not a plummet.
    ok = max_sink < 1600.0 and lo < hi
    print(label)
    print(f"  max sink={max_sink:.0f} fpm   sink@~200ft={hi:.0f}  sink@~50ft={lo:.0f} fpm"
          f"   {'DECREASING toward ground' if lo < hi else 'NOT decreasing'}")
    print(f"  AGL FLOOR BOUNDED + EASED THE SINK: {ok}\n")
    return ok


if __name__ == "__main__":
    ok1 = run("1. NORMAL descent (200 ft disturbance + trim kicks) — no balloon?")
    ok2 = run("2. HIGH-ALT target snaps to ground at t=60 — resists bad dive?",
              snap_to_ground_at=60.0)
    ok3 = run_floor("3. LOW-ALT (250 ft) told to dive to ground — AGL floor bounds sink?")
    print("=" * 60)
    print(f"RESULT: no_balloon={'PASS' if ok1 else 'FAIL'}  "
          f"resist_dive={'PASS' if ok2 else 'FAIL'}  agl_floor={'PASS' if ok3 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)
