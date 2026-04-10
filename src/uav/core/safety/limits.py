from __future__ import annotations

from dataclasses import dataclass

from uav.sim.types import Actuators


@dataclass
class SafetyLimits:
    throttle_min: float
    throttle_max: float
    max_roll: float
    max_pitch: float
    max_yaw: float
    brake_min: float = 0.0
    brake_max: float = 1.0

    def clamp(self, act: Actuators) -> Actuators:
        throttle = min(max(act.throttle, self.throttle_min), self.throttle_max)
        roll = min(max(act.roll, -self.max_roll), self.max_roll)
        pitch = min(max(act.pitch, -self.max_pitch), self.max_pitch)
        yaw = min(max(act.yaw, -self.max_yaw), self.max_yaw)
        brake = min(max(act.brake_ratio, self.brake_min), self.brake_max)
        return Actuators(throttle=throttle, roll=roll, pitch=pitch, yaw=yaw, brake_ratio=brake, gear_down=act.gear_down, flap_ratio=act.flap_ratio)


def abort_actuators() -> Actuators:
    return Actuators(throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0, brake_ratio=1.0, gear_down=True)
