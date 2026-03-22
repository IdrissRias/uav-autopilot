"""Let a trained model fly the plane.

Usage:
    python -m uav.rl.skills.fly_model models/level_flight_bc.npz --duration 60
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from uav.rl.xplane_rl_adapter import XPlaneRLAdapter
from uav.rl.numpy_policy import NumpyMLPPolicy
from uav.sim.types import Actuators


def _wrap(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def fly(args):
    adapter = XPlaneRLAdapter(
        xplane_ip=args.xplane_ip,
        xplane_port=args.xplane_port,
        local_port=0,
        freq_hz=int(args.loop_hz),
    )
    time.sleep(1.0)

    policy = NumpyMLPPolicy(args.model)
    print(f"[FLY] Loaded model: {args.model} ({len(policy.layers)} layers)")

    # Capture current state as targets
    t = adapter.read_telemetry()
    target_alt = t.altitude_ft
    target_hdg = t.heading_deg
    target_spd = t.airspeed_kts
    print(f"[FLY] Targets: alt={target_alt:.0f}ft hdg={target_hdg:.0f}° spd={target_spd:.0f}kts")
    print(f"[FLY] AI is flying for {args.duration}s. Ctrl+C to stop.")

    start = time.time()
    tick = 0

    try:
        while time.time() - start < args.duration:
            loop_start = time.time()

            t = adapter.read_telemetry()
            if not t.is_valid():
                time.sleep(0.05)
                continue

            # Build observation (same as training)
            obs = np.clip(np.array([
                t.pitch_deg / 20.0,
                t.roll_deg / 30.0,
                _wrap(t.heading_deg - target_hdg) / 30.0,
                (t.altitude_ft - target_alt) / 200.0,
                0.0,  # speed error zeroed — level flight ignores speed
                t.pitch_deg / 10.0,
            ], dtype=np.float32), -3.0, 3.0)

            # Get action from AI (pitch + roll only)
            raw_action = policy.predict(obs)
            act = np.clip(raw_action, -1.0, 1.0)

            # Apply to plane — throttle fixed, AI controls pitch + roll
            actuators = Actuators(
                throttle=0.5,
                pitch=float(act[0]) * 0.5,
                roll=float(act[1]) * 0.5,
                yaw=0.0,
                gear_down=False,
                flap_ratio=0.0,
            )
            adapter.write_actuators(actuators)

            tick += 1
            if tick % 50 == 0:
                elapsed = time.time() - start
                agl_ft = t.agl_m * 3.28084 if not math.isnan(t.agl_m) else 0
                print(f"  [{elapsed:.0f}s] alt={t.altitude_ft:.0f} hdg={t.heading_deg:.0f} spd={t.airspeed_kts:.0f} "
                      f"agl={agl_ft:.0f} pitch_cmd={act[0]:.2f} roll_cmd={act[1]:.2f} thr=0.50")

            # Rate limit
            dt = time.time() - loop_start
            sleep_time = (1.0 / args.loop_hz) - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[FLY] Stopped by user.")

    print("[FLY] AI flight ended. You have control.")


def main():
    p = argparse.ArgumentParser(description="Let AI fly the plane")
    p.add_argument("model", nargs="?", default="models/level_flight_latest.npz",
                   help="Path to .npz model weights (default: latest checkpoint)")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--loop-hz", type=float, default=10.0)
    p.add_argument("--xplane-ip", default="127.0.0.1")
    p.add_argument("--xplane-port", type=int, default=49000)
    args = p.parse_args()
    fly(args)


if __name__ == "__main__":
    main()
