"""Skill 1: Hold level flight — raw actuator control, no teleport.

The agent IS the brain. It outputs raw pitch, roll, and throttle commands.
No PID. No teleport. You fly the plane to altitude, then the agent takes over.

Observation (6):
    [0] pitch / 20
    [1] roll / 30
    [2] heading_error / 30
    [3] altitude_error / 200
    [4] speed_error / 20
    [5] vertical_speed / 10    (pitch as proxy for climb/descent)

Action (3):
    [0] pitch     [-1, 1] → elevator
    [1] roll      [-1, 1] → aileron
    [2] throttle  [-1, 1] → [0.0, 1.0]
"""
from __future__ import annotations

import math
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav.rl.xplane_rl_adapter import XPlaneRLAdapter
from uav.sim.types import Actuators

OBS_DIM = 6
ACT_DIM = 2  # pitch + roll only — throttle is fixed


def _wrap(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


class LevelFlightEnv(gym.Env):
    """Learn to fly level with raw actuator commands. No teleport — starts from current state."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        adapter: XPlaneRLAdapter,
        target_alt_ft: float = 2500.0,
        target_hdg: float = 90.0,
        target_speed_kts: float = 100.0,
        sim_speed: float = 2.0,
        loop_hz: float = 10.0,
        episode_steps: int = 600,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.target_alt_ft = target_alt_ft
        self.target_hdg = target_hdg
        self.target_speed_kts = target_speed_kts
        self.sim_speed = sim_speed
        self.loop_hz = loop_hz
        self.episode_steps = episode_steps

        self.observation_space = spaces.Box(
            low=-3.0, high=3.0, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32,
        )

        self._step_count = 0
        self._prev_action: np.ndarray | None = None
        self._prev_time = 0.0
        self._total_episodes = 0
        self._episode_rewards: list[float] = []
        self._ep_reward = 0.0  # accumulates reward for current episode
        self._tolerance_level = 0  # 0=loose, 1=medium, 2=tight
        self._consecutive_good = 0  # consecutive episodes above threshold

        # Captured on first reset — whatever state the plane is in
        self._initial_alt = target_alt_ft
        self._initial_hdg = target_hdg
        self._initial_spd = target_speed_kts

    def reset(
        self, *, seed: int | None = None, options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self.adapter._send_dref(self.sim_speed, "sim/time/sim_speed")

        # No teleport — just keep flying. Read current state as the new target.
        # Teleport is unreliable in X-Plane 12, so we never reset position.
        # Each "episode" is just a new reward cycle from wherever the plane is.
        time.sleep(0.1)
        t = self._wait_valid_telemetry()

        # On first reset, capture the plane's current state as targets
        if self._total_episodes == 0:
            self._initial_alt = t.altitude_ft
            self._initial_hdg = t.heading_deg
            self._initial_spd = t.airspeed_kts
            self.target_alt_ft = t.altitude_ft
            self.target_hdg = t.heading_deg
            self.target_speed_kts = t.airspeed_kts
            print(f"[RL] Captured targets: alt={t.altitude_ft:.0f}ft hdg={t.heading_deg:.0f}° spd={t.airspeed_kts:.0f}kts")

        # Keep the same targets — the goal is always to hold the original alt/hdg

        # Track performance and advance tolerance only when earning it
        if self._total_episodes > 0 and self._step_count > 0:
            avg_reward_per_step = self._ep_reward / self._step_count
            self._episode_rewards.append(avg_reward_per_step)

            # Advance tolerance when agent sustains good per-step reward
            # Thresholds: loose→medium needs avg 0.4/step for 10 eps
            #             medium→tight needs avg 0.5/step for 15 eps
            thresholds = [0.4, 0.5]
            required_streak = [10, 15]

            if self._tolerance_level < 2:
                target = thresholds[self._tolerance_level]
                needed = required_streak[self._tolerance_level]
                if avg_reward_per_step >= target:
                    self._consecutive_good += 1
                else:
                    self._consecutive_good = 0

                if self._consecutive_good >= needed:
                    self._tolerance_level += 1
                    self._consecutive_good = 0
                    tol_names = ["loose", "medium", "tight"]
                    print(f"[RL] ★ ADVANCING to {tol_names[self._tolerance_level]} tolerance! (earned it)")

        self._total_episodes += 1
        self._step_count = 0
        self._prev_action = None
        self._prev_time = time.time()
        self._ep_reward = 0.0

        tol_names = ["loose", "medium", "tight"]
        print(f"[RL] Episode {self._total_episodes} — tolerance: {tol_names[self._tolerance_level]}")

        return self._build_obs(t), {"phase": "level_flight"}

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Rate limit
        now = time.time()
        dt_target = 1.0 / self.loop_hz
        elapsed = now - self._prev_time
        if elapsed < dt_target:
            time.sleep(dt_target - elapsed)
        self._prev_time = time.time()

        # Only pitch + roll — throttle is fixed at 50%
        # This lets the AI focus on what matters: keeping wings level and altitude
        act = np.clip(action, -1.0, 1.0)
        actuators = Actuators(
            throttle=0.5,                             # fixed — not the AI's problem
            pitch=float(act[0]) * 0.5,               # scale down for gentler control
            roll=float(act[1]) * 0.5,                 # scale down for gentler control
            yaw=0.0,
            gear_down=False,
            flap_ratio=0.0,
        )
        self.adapter.write_actuators(actuators)

        # Read telemetry
        t = self.adapter.read_telemetry()
        self._step_count += 1

        agl_ft = t.agl_m * 3.28084 if not math.isnan(t.agl_m) else 500.0

        # Danger detection — heavy penalty but DON'T terminate
        # We never terminate episodes because teleport doesn't work reliably.
        # Instead, the agent must learn to recover from bad states.
        in_danger = (
            not t.is_valid()
            or agl_ft < 100.0
            or abs(t.roll_deg) > 60.0      # too much bank
            or abs(t.pitch_deg) > 40.0      # too steep
        )
        crashed = False  # never terminate — keep flying
        timed_out = self._step_count >= self.episode_steps

        # ── Reward (progressively tightens as agent improves) ──
        hdg_err = abs(_wrap(t.heading_deg - self.target_hdg))
        alt_err = abs(t.altitude_ft - self.target_alt_ft)
        spd_err = abs(t.airspeed_kts - self.target_speed_kts)

        # Tightening: tolerances shrink only when agent earns it (performance-based)
        if self._tolerance_level == 0:
            hdg_tol, alt_tol, spd_tol, bank_tol = 30.0, 200.0, 30.0, 20.0
        elif self._tolerance_level == 1:
            hdg_tol, alt_tol, spd_tol, bank_tol = 15.0, 100.0, 15.0, 10.0
        else:
            hdg_tol, alt_tol, spd_tol, bank_tol = 5.0, 50.0, 8.0, 5.0

        r_alive = 0.1
        r_hdg = max(0, 1.0 - hdg_err / hdg_tol) * 0.3
        r_alt = max(0, 1.0 - alt_err / alt_tol) * 0.4
        r_spd = max(0, 1.0 - spd_err / spd_tol) * 0.2
        r_wings = max(0, 1.0 - abs(t.roll_deg) / bank_tol) * 0.1

        # Smoothness
        if self._prev_action is not None:
            delta = np.abs(action - self._prev_action).mean()
        else:
            delta = 0.0
        r_smooth = -0.05 * float(delta)

        r_crash = -2.0 if in_danger else 0.0  # penalty for dangerous state, but don't terminate

        total = r_alive + r_hdg + r_alt + r_spd + r_wings + r_smooth + r_crash
        self._ep_reward += total
        self._prev_action = action.copy()

        obs = self._build_obs(t)
        info = {
            "hdg_err": hdg_err, "alt_err": alt_err,
            "spd_err": spd_err, "bank": abs(t.roll_deg),
            "step": self._step_count, "reward": total,
        }

        return obs, total, crashed, timed_out, info

    def close(self) -> None:
        try:
            self.adapter._send_dref(1.0, "sim/time/sim_speed")
        except Exception:
            pass

    def _build_obs(self, t) -> np.ndarray:
        return np.clip(np.array([
            t.pitch_deg / 20.0,
            t.roll_deg / 30.0,
            _wrap(t.heading_deg - self.target_hdg) / 30.0,
            (t.altitude_ft - self.target_alt_ft) / 200.0,
            (t.airspeed_kts - self.target_speed_kts) / 20.0,
            t.pitch_deg / 10.0,
        ], dtype=np.float32), -3.0, 3.0)

    def _wait_valid_telemetry(self, timeout: float = 3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            t = self.adapter.read_telemetry()
            if t.is_valid():
                return t
            time.sleep(0.05)
        return self.adapter.read_telemetry()
