"""Per-skill demo recording.

Records human pilot inputs alongside telemetry observations for a specific
skill. Uses SkillDef to determine the observation/action format.
"""
from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

from uav.rl.xplane_rl_adapter import XPlaneRLAdapter
from uav.sim.types import Telemetry
from uav.rl.skills.skill_registry import SkillDef, CONTROL_DREFS


class DemoRecorder:
    """Records human flight data for a single skill."""

    def __init__(self, adapter: XPlaneRLAdapter, skill: SkillDef, loop_hz: float = 10.0):
        self.adapter = adapter
        self.skill = skill
        self.loop_hz = loop_hz

        self._observations: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._recording = False
        self._start_time = 0.0

    @property
    def n_samples(self) -> int:
        return len(self._observations)

    @property
    def duration(self) -> float:
        if not self._recording:
            return 0.0
        return time.time() - self._start_time

    def subscribe_control_drefs(self) -> None:
        """Subscribe to pilot input datarefs (200-205)."""
        for idx, dref in CONTROL_DREFS.items():
            payload = struct.pack("<5sii", b"RREF\0", int(self.loop_hz), idx)
            dr = dref.encode("ascii") + b"\0"
            dr = dr.ljust(400, b"\0")
            self.adapter.sock.sendto(payload + dr, self.adapter.xplane_addr)
        time.sleep(0.3)

    def start(self) -> None:
        """Begin recording."""
        self._observations = []
        self._actions = []
        self._recording = True
        self._start_time = time.time()

    def stop(self) -> tuple[np.ndarray, np.ndarray]:
        """Stop recording and return collected data."""
        self._recording = False
        if not self._observations:
            return np.zeros((0, self.skill.obs_dim), dtype=np.float32), \
                   np.zeros((0, self.skill.act_dim), dtype=np.float32)
        return (
            np.array(self._observations, dtype=np.float32),
            np.array(self._actions, dtype=np.float32),
        )

    def tick(self, telemetry: Telemetry, targets: dict, extras: dict) -> bool:
        """Record one frame. Call at loop_hz rate.

        Returns True if a sample was recorded.
        """
        if not self._recording:
            return False
        if not telemetry.is_valid():
            return False

        # Read raw pilot inputs from UDP buffer
        raw_drefs = self._read_control_drefs()

        # Add throttle position to extras for obs builders that need it
        extras["throttle_pos"] = float(raw_drefs.get(202, 0.3))

        # Build observation and action using skill-specific functions
        obs = self.skill.obs_builder(telemetry, targets, extras)
        action = self.skill.pilot_to_action(raw_drefs)

        self._observations.append(obs)
        self._actions.append(action)
        return True

    def save(self, path: str | Path, targets: dict) -> None:
        """Save recorded data to .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        obs, acts = self.stop() if self._recording else (
            np.array(self._observations, dtype=np.float32) if self._observations
            else np.zeros((0, self.skill.obs_dim), dtype=np.float32),
            np.array(self._actions, dtype=np.float32) if self._actions
            else np.zeros((0, self.skill.act_dim), dtype=np.float32),
        )

        np.savez(
            str(path),
            observations=obs,
            actions=acts,
            skill_name=self.skill.name,
            **{f"target_{k}": v for k, v in targets.items()},
        )

    def _read_control_drefs(self) -> dict[int, float]:
        """Read pending RREF packets and extract control dref values."""
        raw = {}
        try:
            while True:
                data, _ = self.adapter.sock.recvfrom(1024)
                if not data.startswith(b"RREF"):
                    continue
                off = 5
                while off + 8 <= len(data):
                    idx, val = struct.unpack_from("<if", data, off)
                    raw[idx] = val
                    # Also update telemetry cache for mapped indices
                    mapped = self.adapter._IDX_MAP.get(idx, idx)
                    if mapped != idx:
                        self.adapter._last_values[mapped] = val
                    off += 8
        except Exception:
            pass
        return raw
