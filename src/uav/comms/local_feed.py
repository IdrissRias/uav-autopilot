"""Local UDP target feed.

The autopilot's TARGETS (target altitude, target speed) live only in the daemon,
not in X-Plane. This broadcasts them as a tiny JSON packet to 127.0.0.1 so a
LOCAL dashboard (peregrine_telemetry) can show current-vs-target deltas without
any cloud dependency — it already reads X-Plane telemetry over UDP; this is the
same idea for the one thing X-Plane doesn't know: what the autopilot is aiming
for. Fire-and-forget: a failed send never disturbs the control loop.
"""
from __future__ import annotations

import json
import socket

LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 49007   # peregrine_telemetry binds this to receive the target feed

_sock: socket.socket | None = None


def _get_sock() -> socket.socket:
    global _sock
    if _sock is None:
        _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return _sock


def publish_targets(target_alt_ft, target_speed_kts, phase: str) -> None:
    """Send one target frame to the local dashboard. Never raises."""
    try:
        payload = json.dumps({
            "tgt_alt_ft": (None if target_alt_ft is None else float(target_alt_ft)),
            "tgt_spd_kts": (None if target_speed_kts is None else float(target_speed_kts)),
            "phase": phase,
        }).encode("utf-8")
        _get_sock().sendto(payload, (LOCAL_HOST, LOCAL_PORT))
    except Exception:
        pass   # fire-and-forget
