"""
Closed-loop glideslope STABILITY test — the thing that should have existed
before any glideslope gain was ever tuned by eyeball.

The flight 162820 disaster (8 sign-crossings, +-150-194 ft swings, the WHOLE
descent) was invisible to sparse-sampled telemetry printouts and to short
single-tick bench checks. It only shows up if you run the REAL closed loop
(pitch off-slope convergence + throttle altitude walk + real aircraft
dynamics) for long enough to see multiple oscillation periods and measure
whether the amplitude DECAYS or not.

This replicates the real formulas from flight_engine.py (off_slope -> vs_target)
and tecs_controller.py (vs_target -> pitch trim; alt_err -> throttle walk) and
drives the calibrated King Air point-mass model with them, with an initial
200 ft disturbance, for 200 s of simulated flight — long enough for 5+ cycles
of the ~20-30 s phugoid period this project has fought all day.

Run:  PYTHONPATH=src python3 tests/sim_glideslope_stability.py
"""
from __future__ import annotations
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from sim_longitudinal import KingAir, FT_TO_M, MS_TO_KT   # noqa: E402
from uav.core.control.attitude import default_pitch_axis   # noqa: E402

FT_PER_NM = 500.0   # corrected _GLIDE_FT_PER_NM


def run(conv_gain_hi, conv_gain_lo, closure_damp, theta_trim_gain,
        thr_gain, label, T=220.0, dt=0.1, disturb_ft=200.0,
        repeated_kicks=False):
    ac = KingAir(h=850.0 * FT_TO_M, V=150.0 / MS_TO_KT)
    ac.gamma = math.radians(-4.7)  # start roughly on the required slope angle
    pitch_axis = default_pitch_axis()

    ideal_alt_ft = 850.0 + disturb_ft   # geometric line starts here; plane is BELOW it
    # (equivalently: think of "ideal_alt_ft - actual" as off_slope sign convention
    #  matching flight_engine: off_slope_ft = actual - ideal, + = above)
    gs_kts = 150.0
    required_fpm = -(FT_PER_NM * gs_kts / 60.0)

    off_slope_prev = None
    off_rate_filt = 0.0
    theta_ref = math.degrees(ac.theta)
    thr = 0.5
    prev_theta = None
    rate_filt = 0.0

    trace = []
    kick_times = {30.0, 70.0, 110.0} if repeated_kicks else set()
    kicked = set()
    for i in range(int(T / dt)):
        t = i * dt
        # DISCRETE RECONFIGURATION KICK: flap/gear deployment abruptly
        # changes trim (drag/lift), like a real DESCENT->DESCENT_FLAP-
        # >DESCENT_GEAR transition. Model it as a flight-path-angle jolt
        # the plane has to re-trim through, exactly the kind of repeated
        # excitation a single clean disturbance test can't reveal.
        for kt in kick_times:
            if t >= kt and kt not in kicked:
                ac.gamma -= math.radians(2.0)  # sudden extra 2 deg nose-down
                kicked.add(kt)
        actual_alt_ft = ac.h / FT_TO_M
        ideal_alt_ft += (required_fpm / 60.0) * dt   # the line keeps descending
        off_slope_ft = actual_alt_ft - ideal_alt_ft  # + above, - below

        if off_slope_prev is not None:
            raw_rate = (off_slope_ft - off_slope_prev) / dt
            off_rate_filt = 0.3 * raw_rate + 0.7 * off_rate_filt
        off_slope_prev = off_slope_ft

        conv_gain = conv_gain_hi if off_slope_ft >= 0.0 else conv_gain_lo
        vs_raw = required_fpm - off_slope_ft * conv_gain - off_rate_filt * closure_damp
        vs_target = max(-1500.0, min(250.0, vs_raw))

        vs_now = ac.V * math.sin(ac.gamma) * 196.85  # m/s -> fpm
        vs_err = vs_target - vs_now
        theta_ref += vs_err * theta_trim_gain * dt
        theta_ref = max(-8.0, min(6.0, theta_ref))
        pitch_dmd = max(-10.0, min(8.0, theta_ref + vs_err * (theta_trim_gain * 0.44)))

        pdeg = math.degrees(ac.theta)
        if prev_theta is not None:
            raw = (pdeg - prev_theta) / dt
            rate_filt = 0.5 * raw + 0.5 * rate_filt
        prev_theta = pdeg
        elevator = 0.05 * (pitch_dmd - pdeg) - 0.055 * rate_filt
        # REAL constraint the earlier version of this test missed: every
        # DESCENT/APPROACH keyframe carries pitch_limit=0.25 — nose-UP
        # elevator authority is capped at 25%, not the full +-1.0 this test
        # used before. A plane needing more than that to climb back to the
        # line simply can't get it, and the trim integral (theta_ref) keeps
        # winding up while saturated — textbook windup-driven overshoot.
        elevator = min(elevator, 0.25)
        elevator = max(-1.0, min(1.0, elevator))

        alt_err_ft = ideal_alt_ft - actual_alt_ft   # + = below target (throttle convention)
        thr += alt_err_ft * thr_gain * dt
        thr = max(0.0, min(1.0, thr))

        ac.step(thr, elevator, dt)
        trace.append(off_slope_ft)

    # Peak-to-peak amplitude in each 40s window; must DECAY, not hold/grow.
    win = int(40 / dt)
    windows = [trace[k:k + win] for k in range(0, len(trace) - win, win)]
    amps = [max(w) - min(w) for w in windows]
    crossings = sum(1 for a, b in zip(trace, trace[1:]) if (a > 0) != (b > 0))
    print(f"{label}")
    print(f"  peak-to-peak amplitude per 40s window: " + " -> ".join(f"{a:.0f}" for a in amps))
    print(f"  sign crossings over {T:.0f}s: {crossings}")
    decaying = len(amps) >= 2 and amps[-1] < amps[0] * 0.6
    verdict = "DAMPED (converging)" if decaying else "NOT DAMPED (sustained/growing oscillation)"
    print(f"  VERDICT: {verdict}\n")
    return amps, crossings


if __name__ == "__main__":
    # 1. REPRODUCE the crash: the gains actually shipped in flight 162820.
    run(conv_gain_hi=1.5, conv_gain_lo=2.5, closure_damp=30.0,
        theta_trim_gain=0.0018, thr_gain=0.0008,
        label="CURRENT (shipped) gains — should reproduce the oscillation")

    # 2. Candidate fix: pitch mostly flies the baseline slope, throttle owns
    #    the position correction (per Idriss doctrine: altitude is throttle's
    #    job). Pitch's off-slope authority cut way down; damping increased.
    run(conv_gain_hi=0.15, conv_gain_lo=0.20, closure_damp=45.0,
        theta_trim_gain=0.0008, thr_gain=0.0008,
        label="CANDIDATE FIX — pitch mostly holds baseline slope, throttle owns correction")

    print("=" * 70)
    print("REPEATED-KICK STRESS TEST (models flap/gear deployment trim jolts)")
    print("=" * 70 + "\n")

    run(conv_gain_hi=1.5, conv_gain_lo=2.5, closure_damp=30.0,
        theta_trim_gain=0.0018, thr_gain=0.0008, disturb_ft=0.0,
        label="CURRENT gains, repeated kicks — this is the real-flight scenario",
        repeated_kicks=True)

    run(conv_gain_hi=0.15, conv_gain_lo=0.20, closure_damp=45.0,
        theta_trim_gain=0.0008, thr_gain=0.0008, disturb_ft=0.0,
        label="CANDIDATE FIX, repeated kicks",
        repeated_kicks=True)
