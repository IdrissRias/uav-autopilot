from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from uav.sim.types import Telemetry, Actuators, Targets


class Recorder:
    def __init__(self, log_dir: str = "logs") -> None:
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(log_dir, f"run-{ts}.csv")
        self._file = open(self.path, "w", newline="")
        self._writer = None

    def _init_writer(self, targets: Optional[Targets]) -> None:
        fieldnames = [
            "time",
            "mode",
            "telemetry_airspeed_kts",
            "telemetry_altitude_ft",
            "telemetry_pitch_deg",
            "telemetry_roll_deg",
            "telemetry_heading_deg",
            "telemetry_lat_deg",
            "telemetry_lon_deg",
            "act_throttle",
            "act_roll",
            "act_pitch",
            "act_yaw",
            "act_brake_ratio",
            "target_heading_deg",
            "target_altitude_ft",
            "target_airspeed_kts",
            "target_throttle",
            "target_brake_ratio",
            # ── L1 track-follower telemetry (populated once FlightEngine
            # arms the follower on an aim_at keyframe). Lets us grade
            # "religious following" offline: sustained low |cross_track_nm|
            # = plane hugs the ribbon line; segment_idx monotonic advance
            # = no regressions / U-turns past the end.
            "cross_track_nm",
            "along_track_nm",
            "segment_idx",
            "segment_progress",
            "ribbon_length_nm",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def record(
        self,
        mode: str,
        telemetry: Telemetry,
        targets: Optional[Targets],
        actuators: Actuators,
        ribbon: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._writer is None:
            self._init_writer(targets)

        target_heading = targets.heading_deg if targets else None
        target_alt = targets.altitude_ft if targets else None
        target_spd = targets.airspeed_kts if targets else None
        target_thr = targets.throttle if targets else None
        target_brake = targets.brake_ratio if targets else None

        # Ribbon follower stats — None before follower arms or on
        # keyframes with non-aim_at heading modes (dep/dest runway, hold).
        r = ribbon or {}

        row = {
            "time": time.time(),
            "mode": mode,
            "telemetry_airspeed_kts": telemetry.airspeed_kts,
            "telemetry_altitude_ft": telemetry.altitude_ft,
            "telemetry_pitch_deg": telemetry.pitch_deg,
            "telemetry_roll_deg": telemetry.roll_deg,
            "telemetry_heading_deg": telemetry.heading_deg,
            "telemetry_lat_deg": telemetry.lat_deg,
            "telemetry_lon_deg": telemetry.lon_deg,
            "act_throttle": actuators.throttle,
            "act_roll": actuators.roll,
            "act_pitch": actuators.pitch,
            "act_yaw": actuators.yaw,
            "act_brake_ratio": actuators.brake_ratio,
            "target_heading_deg": target_heading,
            "target_altitude_ft": target_alt,
            "target_airspeed_kts": target_spd,
            "target_throttle": target_thr,
            "target_brake_ratio": target_brake,
            "cross_track_nm": r.get("cross_track_nm"),
            "along_track_nm": r.get("along_track_nm"),
            "segment_idx": r.get("segment_idx"),
            "segment_progress": r.get("segment_progress"),
            "ribbon_length_nm": r.get("ribbon_length_nm"),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
