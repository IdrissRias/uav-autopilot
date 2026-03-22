"""Skill definitions for interactive training mode.

Each skill defines its observation/action spaces, how to build observations
from telemetry, how to map actions to actuators, and how to read pilot inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from uav.sim.types import Actuators, Telemetry


class Phase(str, Enum):
    GROUND = "ground"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    TURN = "turn"
    CRUISE = "cruise"
    DESCEND = "descend"
    LAND = "land"


def _wrap(a: float) -> float:
    """Wrap angle to [-180, 180]."""
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


@dataclass
class SkillDef:
    """Definition of a single flight skill."""
    name: str
    phase: Phase
    obs_dim: int
    act_dim: int
    act_names: list[str]
    prompt: str                                  # instruction shown to user
    min_duration_s: float = 10.0                 # minimum recording time

    # Functions set after construction (avoids circular refs)
    obs_builder: Callable | None = field(default=None, repr=False)
    act_to_actuators: Callable | None = field(default=None, repr=False)
    pilot_to_action: Callable | None = field(default=None, repr=False)
    completion_check: Callable | None = field(default=None, repr=False)

    # Default actuator values for channels NOT controlled by this skill
    throttle_default: float = 0.5
    gear_default: bool = False
    flap_default: float = 0.0
    yaw_default: float = 0.0


# ─── Observation builders ────────────────────────────────────────────

def _obs_takeoff(t: Telemetry, targets: dict, extras: dict) -> np.ndarray:
    agl_ft = t.agl_m * 3.28084 if not math.isnan(t.agl_m) else 0.0
    climb_rate = extras.get("climb_rate_fpm", 0.0)
    throttle_pos = extras.get("throttle_pos", 0.0)
    return np.clip(np.array([
        t.pitch_deg / 20.0,
        t.roll_deg / 30.0,
        t.airspeed_kts / 200.0,
        agl_ft / 100.0,
        _wrap(t.heading_deg - targets.get("hdg", t.heading_deg)) / 30.0,
        throttle_pos,
        climb_rate / 2000.0,
    ], dtype=np.float32), -3.0, 3.0)


def _obs_climb(t: Telemetry, targets: dict, extras: dict) -> np.ndarray:
    climb_rate = extras.get("climb_rate_fpm", 0.0)
    throttle_pos = extras.get("throttle_pos", 0.0)
    return np.clip(np.array([
        t.pitch_deg / 20.0,
        t.roll_deg / 30.0,
        _wrap(t.heading_deg - targets.get("hdg", t.heading_deg)) / 30.0,
        (t.altitude_ft - targets.get("alt", t.altitude_ft)) / 500.0,
        t.airspeed_kts / 200.0,
        climb_rate / 2000.0,
        throttle_pos,
    ], dtype=np.float32), -3.0, 3.0)


def _obs_turn(t: Telemetry, targets: dict, extras: dict) -> np.ndarray:
    hdg_rate = extras.get("hdg_rate_dps", 0.0)
    return np.clip(np.array([
        t.pitch_deg / 20.0,
        t.roll_deg / 30.0,
        _wrap(t.heading_deg - targets.get("hdg", t.heading_deg)) / 30.0,
        (t.altitude_ft - targets.get("alt", t.altitude_ft)) / 200.0,
        t.airspeed_kts / 200.0,
        t.roll_deg / 30.0,     # bank angle (redundant but explicit for turn)
        hdg_rate / 5.0,
    ], dtype=np.float32), -3.0, 3.0)


def _obs_cruise(t: Telemetry, targets: dict, extras: dict) -> np.ndarray:
    return np.clip(np.array([
        t.pitch_deg / 20.0,
        t.roll_deg / 30.0,
        _wrap(t.heading_deg - targets.get("hdg", t.heading_deg)) / 30.0,
        (t.altitude_ft - targets.get("alt", t.altitude_ft)) / 200.0,
        (t.airspeed_kts - targets.get("spd", t.airspeed_kts)) / 20.0,
        t.pitch_deg / 10.0,
    ], dtype=np.float32), -3.0, 3.0)


def _obs_descend(t: Telemetry, targets: dict, extras: dict) -> np.ndarray:
    climb_rate = extras.get("climb_rate_fpm", 0.0)
    throttle_pos = extras.get("throttle_pos", 0.0)
    return np.clip(np.array([
        t.pitch_deg / 20.0,
        t.roll_deg / 30.0,
        _wrap(t.heading_deg - targets.get("hdg", t.heading_deg)) / 30.0,
        (t.altitude_ft - targets.get("alt", t.altitude_ft)) / 500.0,
        t.airspeed_kts / 200.0,
        climb_rate / 2000.0,
        throttle_pos,
    ], dtype=np.float32), -3.0, 3.0)


def _obs_land(t: Telemetry, targets: dict, extras: dict) -> np.ndarray:
    agl_ft = t.agl_m * 3.28084 if not math.isnan(t.agl_m) else 0.0
    climb_rate = extras.get("climb_rate_fpm", 0.0)
    throttle_pos = extras.get("throttle_pos", 0.0)
    return np.clip(np.array([
        t.pitch_deg / 20.0,
        t.roll_deg / 30.0,
        _wrap(t.heading_deg - targets.get("hdg", t.heading_deg)) / 30.0,
        agl_ft / 200.0,
        t.airspeed_kts / 200.0,
        climb_rate / 1000.0,
        t.pitch_deg / 10.0,
        throttle_pos,
    ], dtype=np.float32), -3.0, 3.0)


# ─── Action → Actuators mappers ─────────────────────────────────────

def _act_pitch_roll(action: np.ndarray, skill: SkillDef) -> Actuators:
    """2-action: [pitch, roll]."""
    return Actuators(
        throttle=skill.throttle_default,
        pitch=float(np.clip(action[0], -1, 1)) * 0.5,
        roll=float(np.clip(action[1], -1, 1)) * 0.5,
        yaw=skill.yaw_default,
        gear_down=skill.gear_default,
        flap_ratio=skill.flap_default,
    )


def _act_pitch_roll_throttle(action: np.ndarray, skill: SkillDef) -> Actuators:
    """3-action: [pitch, roll, throttle]."""
    throttle = float(np.clip(action[2], -1, 1) + 1.0) / 2.0  # [-1,1] → [0,1]
    return Actuators(
        throttle=throttle,
        pitch=float(np.clip(action[0], -1, 1)) * 0.5,
        roll=float(np.clip(action[1], -1, 1)) * 0.5,
        yaw=skill.yaw_default,
        gear_down=skill.gear_default,
        flap_ratio=skill.flap_default,
    )


def _act_land(action: np.ndarray, skill: SkillDef) -> Actuators:
    """4-action: [pitch, roll, throttle, flaps]."""
    throttle = float(np.clip(action[2], -1, 1) + 1.0) / 2.0
    flaps = float(np.clip(action[3], -1, 1) + 1.0) / 2.0
    return Actuators(
        throttle=throttle,
        pitch=float(np.clip(action[0], -1, 1)) * 0.5,
        roll=float(np.clip(action[1], -1, 1)) * 0.5,
        yaw=skill.yaw_default,
        gear_down=True,
        flap_ratio=flaps,
    )


# ─── Pilot input → action mappers ───────────────────────────────────

def _pilot_pitch_roll(raw: dict) -> np.ndarray:
    """Read pilot pitch + roll inputs, map to [-1, 1] action."""
    pitch = float(raw.get(200, 0.0)) / 0.5   # undo 0.5 env scaling
    roll = float(raw.get(201, 0.0)) / 0.5
    return np.clip(np.array([pitch, roll], dtype=np.float32), -1.0, 1.0)


def _pilot_pitch_roll_throttle(raw: dict) -> np.ndarray:
    """Read pilot pitch + roll + throttle."""
    pitch = float(raw.get(200, 0.0)) / 0.5
    roll = float(raw.get(201, 0.0)) / 0.5
    throttle = float(raw.get(202, 0.3)) * 2.0 - 1.0  # [0,1] → [-1,1]
    return np.clip(np.array([pitch, roll, throttle], dtype=np.float32), -1.0, 1.0)


def _pilot_land(raw: dict) -> np.ndarray:
    """Read pilot pitch + roll + throttle + flaps."""
    pitch = float(raw.get(200, 0.0)) / 0.5
    roll = float(raw.get(201, 0.0)) / 0.5
    throttle = float(raw.get(202, 0.3)) * 2.0 - 1.0
    flaps = float(raw.get(204, 0.0)) * 2.0 - 1.0  # [0,1] → [-1,1]
    return np.clip(np.array([pitch, roll, throttle, flaps], dtype=np.float32), -1.0, 1.0)


# ─── Completion checks ──────────────────────────────────────────────

def _check_takeoff(t: Telemetry, targets: dict, extras: dict) -> bool:
    agl_ft = t.agl_m * 3.28084 if not math.isnan(t.agl_m) else 0.0
    return agl_ft > 100.0 and extras.get("climb_rate_fpm", 0.0) > 200.0


def _check_climb(t: Telemetry, targets: dict, extras: dict) -> bool:
    target_alt = targets.get("alt", 3000.0)
    return t.altitude_ft >= target_alt - 100.0


def _check_turn(t: Telemetry, targets: dict, extras: dict) -> bool:
    target_hdg = targets.get("hdg", t.heading_deg)
    return abs(_wrap(t.heading_deg - target_hdg)) < 10.0


def _check_cruise(t: Telemetry, targets: dict, extras: dict) -> bool:
    return False  # cruise ends by time or user pressing Enter


def _check_descend(t: Telemetry, targets: dict, extras: dict) -> bool:
    target_alt = targets.get("alt", 1500.0)
    return t.altitude_ft <= target_alt + 100.0


def _check_land(t: Telemetry, targets: dict, extras: dict) -> bool:
    agl_ft = t.agl_m * 3.28084 if not math.isnan(t.agl_m) else 500.0
    return agl_ft < 5.0 and t.airspeed_kts < 40.0


# ─── Registry ────────────────────────────────────────────────────────

def build_skill_registry() -> list[SkillDef]:
    """Build the ordered list of all trainable skills."""
    skills = [
        SkillDef(
            name="takeoff", phase=Phase.TAKEOFF, obs_dim=7, act_dim=3,
            act_names=["pitch", "roll", "throttle"],
            prompt="Take off! Push throttle to full, rotate, and climb to 1000ft AGL.",
            min_duration_s=15.0,
            throttle_default=1.0, gear_default=True, flap_default=0.0,
            obs_builder=_obs_takeoff,
            act_to_actuators=_act_pitch_roll_throttle,
            pilot_to_action=_pilot_pitch_roll_throttle,
            completion_check=_check_takeoff,
        ),
        SkillDef(
            name="climb", phase=Phase.CLIMB, obs_dim=7, act_dim=3,
            act_names=["pitch", "roll", "throttle"],
            prompt="Climb to {alt}ft. Keep wings level and heading steady.",
            min_duration_s=15.0,
            throttle_default=0.8, gear_default=False, flap_default=0.0,
            obs_builder=_obs_climb,
            act_to_actuators=_act_pitch_roll_throttle,
            pilot_to_action=_pilot_pitch_roll_throttle,
            completion_check=_check_climb,
        ),
        SkillDef(
            name="turn", phase=Phase.TURN, obs_dim=7, act_dim=2,
            act_names=["pitch", "roll"],
            prompt="Turn to heading {hdg}°. Smooth, coordinated turn.",
            min_duration_s=10.0,
            throttle_default=0.5, gear_default=False, flap_default=0.0,
            obs_builder=_obs_turn,
            act_to_actuators=_act_pitch_roll,
            pilot_to_action=_pilot_pitch_roll,
            completion_check=_check_turn,
        ),
        SkillDef(
            name="cruise", phase=Phase.CRUISE, obs_dim=6, act_dim=2,
            act_names=["pitch", "roll"],
            prompt="Hold straight and level. Keep altitude and heading steady for 30 seconds.",
            min_duration_s=20.0,
            throttle_default=0.5, gear_default=False, flap_default=0.0,
            obs_builder=_obs_cruise,
            act_to_actuators=_act_pitch_roll,
            pilot_to_action=_pilot_pitch_roll,
            completion_check=_check_cruise,
        ),
        SkillDef(
            name="descend", phase=Phase.DESCEND, obs_dim=7, act_dim=3,
            act_names=["pitch", "roll", "throttle"],
            prompt="Descend to {alt}ft. Reduce power, hold heading.",
            min_duration_s=15.0,
            throttle_default=0.3, gear_default=False, flap_default=0.0,
            obs_builder=_obs_descend,
            act_to_actuators=_act_pitch_roll_throttle,
            pilot_to_action=_pilot_pitch_roll_throttle,
            completion_check=_check_descend,
        ),
        SkillDef(
            name="land", phase=Phase.LAND, obs_dim=8, act_dim=4,
            act_names=["pitch", "roll", "throttle", "flaps"],
            prompt="Land the plane. Gear down, flaps as needed, smooth touchdown.",
            min_duration_s=15.0,
            throttle_default=0.2, gear_default=True, flap_default=0.5,
            obs_builder=_obs_land,
            act_to_actuators=_act_land,
            pilot_to_action=_pilot_land,
            completion_check=_check_land,
        ),
    ]
    return skills


# Extra datarefs needed beyond the standard 100-107 telemetry
CONTROL_DREFS = {
    200: "sim/cockpit2/controls/yoke_pitch_ratio",
    201: "sim/cockpit2/controls/yoke_roll_ratio",
    202: "sim/cockpit2/engine/actuators/throttle_ratio_all",
    203: "sim/cockpit2/controls/gear_handle_down",
    204: "sim/flightmodel/controls/flaprqst",
    205: "sim/flightmodel/position/vh_ind_fpm",
}
