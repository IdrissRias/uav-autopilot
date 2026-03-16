from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


@dataclass
class Telemetry:
    airspeed_kts: float
    altitude_ft: float
    pitch_deg: float
    roll_deg: float
    heading_deg: float
    timestamp: float
    lat_deg: float = math.nan
    lon_deg: float = math.nan
    agl_m: float = math.nan

    def is_valid(self) -> bool:
        # Valid attitude/airspeed/alt/heading telemetry (position optional).
        # agl_m < -2.0 means the aircraft is genuinely underground/crashed — treat as invalid.
        # -0.5 was too tight: terrain mesh rounding after a POSI teleport gives -0.3 to -1.5m
        # transiently even when the plane is fine.
        if not math.isnan(self.agl_m) and self.agl_m < -2.0:
            return False
        return not (
            math.isnan(self.airspeed_kts)
            or math.isnan(self.altitude_ft)
            or math.isnan(self.pitch_deg)
            or math.isnan(self.roll_deg)
            or math.isnan(self.heading_deg)
        )

    def has_position(self) -> bool:
        return not (math.isnan(self.lat_deg) or math.isnan(self.lon_deg))


@dataclass
class Actuators:
    throttle: float
    roll: float
    pitch: float
    yaw: float
    brake_ratio: float = 0.0
    gear_down: bool = True


@dataclass
class Targets:
    heading_deg: float | None
    altitude_ft: float | None
    airspeed_kts: float | None
    climb_rate_fpm: float | None = None
    throttle: float | None = None
    brake_ratio: float | None = None
    gear_down: bool | None = None
    roll_limit: float | None = None
    pitch_limit: float | None = None
    yaw_hold: bool | None = None
    yaw_kp: float | None = None
    yaw_ki: float | None = None
    yaw_limit: float | None = None
    yaw_full_deg: float | None = None
    pitch_protect_kts: float | None = None
    pitch_protect_gain: float | None = None


class Mode(str, Enum):
    GROUND = "GROUND"
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    APPROACH = "APPROACH"
    LAND = "LAND"
    ABORT = "ABORT"
