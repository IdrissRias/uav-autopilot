"""AI inference per skill with smooth blending.

Handles the transition from human control to AI control (and vice versa)
using exponential blending to avoid jerky inputs.
"""
from __future__ import annotations

import math
import time

import numpy as np

from uav.rl.numpy_policy import NumpyMLPPolicy
from uav.rl.xplane_rl_adapter import XPlaneRLAdapter
from uav.sim.types import Actuators, Telemetry
from uav.rl.skills.skill_registry import SkillDef


class SkillPilot:
    """Flies one skill using a trained NumpyMLPPolicy with smooth blending."""

    BLEND_TAU = 0.3  # exponential blend time constant (seconds)

    def __init__(self, adapter: XPlaneRLAdapter, skill: SkillDef, weights_path: str):
        self.adapter = adapter
        self.skill = skill
        self.policy = NumpyMLPPolicy(weights_path)
        self._blend_start: float | None = None
        self._blend_alpha: float = 1.0  # 1.0 = full AI
        self._last_human_act: Actuators | None = None

    def start_takeover(self, last_human_actuators: Actuators | None = None) -> None:
        """Begin smooth transition from human to AI control."""
        self._last_human_act = last_human_actuators
        if last_human_actuators is not None:
            self._blend_start = time.time()
            self._blend_alpha = 0.0
        else:
            self._blend_alpha = 1.0
            self._blend_start = None

    def tick(self, telemetry: Telemetry, targets: dict, extras: dict) -> Actuators:
        """Run one inference step. Returns blended actuators."""
        # Build observation
        obs = self.skill.obs_builder(telemetry, targets, extras)

        # Run policy
        raw_action = self.policy.predict(obs)
        action = np.clip(raw_action, -1.0, 1.0)

        # Map to actuators
        ai_act = self.skill.act_to_actuators(action, self.skill)

        # Update blend alpha
        if self._blend_start is not None:
            elapsed = time.time() - self._blend_start
            self._blend_alpha = min(1.0, 1.0 - math.exp(-elapsed / self.BLEND_TAU))
            if self._blend_alpha > 0.99:
                self._blend_alpha = 1.0
                self._blend_start = None

        # Blend if transitioning
        if self._blend_alpha < 1.0 and self._last_human_act is not None:
            return _lerp_actuators(self._last_human_act, ai_act, self._blend_alpha)

        return ai_act

    @property
    def is_blending(self) -> bool:
        return self._blend_start is not None


def _lerp_actuators(a: Actuators, b: Actuators, alpha: float) -> Actuators:
    """Linearly interpolate between two Actuators."""
    def _lerp(x: float, y: float) -> float:
        return x + alpha * (y - x)

    return Actuators(
        throttle=_lerp(a.throttle, b.throttle),
        pitch=_lerp(a.pitch, b.pitch),
        roll=_lerp(a.roll, b.roll),
        yaw=_lerp(a.yaw, b.yaw),
        brake_ratio=_lerp(a.brake_ratio, b.brake_ratio),
        gear_down=b.gear_down if alpha > 0.5 else a.gear_down,
        flap_ratio=_lerp(a.flap_ratio, b.flap_ratio),
    )
