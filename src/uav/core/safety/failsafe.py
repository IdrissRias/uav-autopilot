from __future__ import annotations

import time

from uav.sim.types import Telemetry


def is_telemetry_stale(telemetry: Telemetry, timeout_s: float) -> bool:
    if telemetry.timestamp <= 0.0:
        return True
    return (time.time() - telemetry.timestamp) > timeout_s
