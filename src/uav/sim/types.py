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
    vs_fpm: float = math.nan       # vertical speed (ft/min) — positive = climbing
    groundspeed_kts: float = math.nan

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
    flap_ratio: float = 0.0   # 0.0 = up, 1.0 = full flaps


@dataclass
class Targets:
    """Commander's orders to the soldier. All fields are nullable — if the ribbon
    doesn't command it, the soldier doesn't try to control it.

    roll_limit/pitch_limit are commander-supplied clamps (optional). Everything
    else is a setpoint. No hidden policy — if it isn't here, it isn't enforced."""
    heading_deg: float | None
    altitude_ft: float | None
    airspeed_kts: float | None
    throttle: float | None = None
    brake_ratio: float | None = None
    gear_down: bool | None = None
    flap_ratio: float | None = None   # 0.0 = up, 1.0 = full; None = don't change
    roll_limit: float | None = None   # ribbon-issued order; None = full authority
    pitch_limit: float | None = None  # ribbon-issued order; None = full authority (nose-up cap)
    pitch_down_limit: float | None = None  # opt-in nose-DOWN cap (only during glideslope
    # phases — preserves stall-recovery authority everywhere else)
    yaw_hold: bool | None = None
    yaw_kp: float | None = None
    yaw_limit: float | None = None


class Mode(str, Enum):
    GROUND = "GROUND"
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    APPROACH = "APPROACH"
    LAND = "LAND"
    ABORT = "ABORT"
