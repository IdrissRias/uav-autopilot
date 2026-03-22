"""Record human flight demonstrations for imitation learning.

You fly the plane. This script records:
  - Observations (what the agent would see)
  - Actions (what YOU are doing — read from X-Plane control surfaces)

Usage:
    python -m uav.rl.skills.record_demo --output demos/level_flight.npz --duration 120

Then train with:
    python -m uav.rl.skills.train_from_demo demos/level_flight.npz
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from uav.rl.xplane_rl_adapter import XPlaneRLAdapter


def _wrap(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def record(args):
    adapter = XPlaneRLAdapter(
        xplane_ip=args.xplane_ip,
        xplane_port=args.xplane_port,
        local_port=0,
        freq_hz=int(args.loop_hz),
    )
    time.sleep(1.0)

    # Also subscribe to control surface datarefs so we can read YOUR inputs
    # These are what the pilot is commanding (joystick/keyboard)
    import struct
    control_drefs = {
        200: "sim/cockpit2/controls/yoke_pitch_ratio",
        201: "sim/cockpit2/controls/yoke_roll_ratio",
        202: "sim/cockpit2/engine/actuators/throttle_ratio_all",
    }
    for idx, dref in control_drefs.items():
        payload = struct.pack("<5sii", b"RREF\0", int(args.loop_hz), idx)
        dr = dref.encode("ascii") + b"\0"
        dr = dr.ljust(400, b"\0")
        adapter.sock.sendto(payload + dr, adapter.xplane_addr)

    # Give subscriptions time to activate
    time.sleep(0.5)

    # Capture initial state as targets
    t = adapter.read_telemetry()
    target_alt = t.altitude_ft
    target_hdg = t.heading_deg
    target_spd = t.airspeed_kts

    print(f"[RECORD] Targets captured: alt={target_alt:.0f}ft hdg={target_hdg:.0f}° spd={target_spd:.0f}kts")
    print(f"[RECORD] Recording for {args.duration}s at {args.loop_hz}Hz...")
    print(f"[RECORD] Fly the plane stable. Press Ctrl+C to stop early.")

    observations = []
    actions = []
    start = time.time()
    tick = 0

    try:
        while time.time() - start < args.duration:
            loop_start = time.time()

            # Read telemetry
            t = adapter.read_telemetry()
            if not t.is_valid():
                time.sleep(0.05)
                continue

            # Read control inputs from X-Plane (what YOU are doing)
            # These come back via RREF with indices 200-202
            raw_data = {}
            try:
                while True:
                    data, _ = adapter.sock.recvfrom(1024)
                    if data[:4] == b"RREF":
                        off = 5
                        while off + 8 <= len(data):
                            idx, val = struct.unpack("<if", data[off:off + 8])
                            raw_data[idx] = val
                            # Also update telemetry cache
                            mapped = adapter._IDX_MAP.get(idx, idx)
                            adapter._last_values[mapped] = val
                            off += 8
            except Exception:
                pass

            pitch_cmd = raw_data.get(200, 0.0)
            roll_cmd = raw_data.get(201, 0.0)
            throttle_cmd = raw_data.get(202, 0.3)

            # Build observation (same as level_flight env)
            obs = np.array([
                t.pitch_deg / 20.0,
                t.roll_deg / 30.0,
                _wrap(t.heading_deg - target_hdg) / 30.0,
                (t.altitude_ft - target_alt) / 200.0,
                (t.airspeed_kts - target_spd) / 20.0,
                t.pitch_deg / 10.0,
            ], dtype=np.float32)
            obs = np.clip(obs, -3.0, 3.0)

            # Action: what the pilot is commanding
            # Map to [-1, 1] to match env action space
            # pitch_cmd and roll_cmd are already [-1, 1] from X-Plane
            # throttle: [0, 1] → [-1, 1]
            action = np.array([
                float(pitch_cmd) / 0.5,      # undo the 0.5 scale in env
                float(roll_cmd) / 0.5,        # undo the 0.5 scale in env
                float(throttle_cmd) * 2.0 - 1.0,  # [0,1] → [-1,1]
            ], dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)

            observations.append(obs)
            actions.append(action)
            tick += 1

            if tick % 50 == 0:
                elapsed = time.time() - start
                print(f"  [{elapsed:.0f}s] alt={t.altitude_ft:.0f} hdg={t.heading_deg:.0f} spd={t.airspeed_kts:.0f} "
                      f"pitch_cmd={pitch_cmd:.2f} roll_cmd={roll_cmd:.2f} thr={throttle_cmd:.2f} "
                      f"({len(observations)} samples)")

            # Rate limit
            dt = time.time() - loop_start
            sleep_time = (1.0 / args.loop_hz) - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[RECORD] Stopped by user.")

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)

    print(f"[RECORD] Collected {len(observations)} samples")
    print(f"[RECORD] Obs shape: {observations.shape}, Action shape: {actions.shape}")

    # Save
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, observations=observations, actions=actions,
             target_alt=target_alt, target_hdg=target_hdg, target_spd=target_spd)
    print(f"[RECORD] Saved to {args.output}")


def main():
    p = argparse.ArgumentParser(description="Record human flight demo")
    p.add_argument("--output", default="demos/level_flight.npz")
    p.add_argument("--duration", type=float, default=120.0, help="Recording duration in seconds")
    p.add_argument("--loop-hz", type=float, default=10.0)
    p.add_argument("--xplane-ip", default="127.0.0.1")
    p.add_argument("--xplane-port", type=int, default=49000)
    args = p.parse_args()
    record(args)


if __name__ == "__main__":
    main()
