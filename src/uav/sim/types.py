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
    # Throttle-for-altitude coupling. When True, the controller drives
    # throttle off the alt error (alt-priority: power for altitude)
    # and pitch off the speed error (attitude for airspeed). Default
    # False = classic decoupled (alt→pitch, speed→throttle). Enabled
    # on cruise/descent phases so the engine modulates to defend alt
    # rather than chasing speed past the target.
    throttle_for_alt: bool = False
    # Baseline throttle for the throttle_for_alt P-law. None = the
    # controller's cruise_throttle. Descent phases set this near idle so
    # an on-slope plane (alt_error ≈ 0) isn't carrying 55%+ power down
    # a steep slope.
    throttle_base: float | None = None
    # Sink-rate command (ft/min, negative = descending). When set, pitch
    # tracks this INSTEAD of altitude/speed — used by the flare, where
    # what matters is arresting sink, not holding an altitude.
    vs_target_fpm: float | None = None
    # Stall floor (kts). Below this airspeed the controller FORCES
    # throttle up regardless of the alt-priority coupling — the
    # low-and-slow corner of the energy matrix, where power is the only
    # fix. Flight 20260706_135230 mushed to 68 kts at idle because the
    # plane was above the slope and the alt law refused power. None on
    # phases where slow is by design (flare/rollout).
    stall_floor_kts: float | None = None
    # Bleed mode: the plane is above the glideslope dumping surplus
    # energy. Pitch HOLDS a nose-up drag attitude (induced drag is the
    # brake) instead of relaxing to zero when speed reaches target, and
    # is capped so the bleed never turns into a climb — pulling into a
    # zoom just re-borrows the energy (flight 914edfc6: 189 kt dive
    # recovery pulled through level into a zoom that stalled at 20 kts).
    bleed_mode: bool = False


class Mode(str, Enum):
    GROUND = "GROUND"
    TAKEOFF = "TAKEOFF"
    CLIMB = "CLIMB"
    CRUISE = "CRUISE"
    APPROACH = "APPROACH"
    LAND = "LAND"
    ABORT = "ABORT"
