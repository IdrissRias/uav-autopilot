"""
Hardware-only actuator clamps.

NOT a safety policy — just the physical envelope of the actuators themselves:
  * surfaces  ∈ [-1, +1]  (full deflection in either direction)
  * throttle  ∈ [0, 1]    (can't go negative, can't exceed full)
  * brake     ∈ [0, 1]    (same)

If the commander says "full deflection," the soldier gives full deflection.
No configurable reductions. No policy. If you think you need one, put it in
the ribbon — that's the commander.
"""
from __future__ import annotations

from uav.sim.types import Actuators


class SafetyLimits:
    """Placeholder for API compat. Accepts any kwargs, ignores them."""

    def __init__(self, **_kwargs) -> None:
        pass

    def clamp(self, act: Actuators) -> Actuators:
        return Actuators(
            throttle=max(0.0, min(1.0, act.throttle)),
            roll=max(-1.0, min(1.0, act.roll)),
            pitch=max(-1.0, min(1.0, act.pitch)),
            yaw=max(-1.0, min(1.0, act.yaw)),
            brake_ratio=max(0.0, min(1.0, act.brake_ratio)),
            gear_down=act.gear_down,
            flap_ratio=max(0.0, min(1.0, act.flap_ratio)),
        )


def abort_actuators() -> Actuators:
    """Neutral actuators for stale-telemetry failsafe."""
    return Actuators(throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0,
                     brake_ratio=1.0, gear_down=True)
