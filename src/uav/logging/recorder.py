from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict
from typing import Optional

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
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def record(
        self,
        mode: str,
        telemetry: Telemetry,
        targets: Optional[Targets],
        actuators: Actuators,
    ) -> None:
        if self._writer is None:
            self._init_writer(targets)

        target_heading = targets.heading_deg if targets else None
        target_alt = targets.altitude_ft if targets else None
        target_spd = targets.airspeed_kts if targets else None
        target_thr = targets.throttle if targets else None
        target_brake = targets.brake_ratio if targets else None

        row = {
            "time": time.time(),
            "mode": mode,
            "telemetry_airspeed_kts": telemetry.airspeed_kts,
            "telemetry_altitude_ft": telemetry.altitude_ft,
            "telemetry_pitch_deg": telemetry.pitch_deg,
            "telemetry_roll_deg": telemetry.roll_deg,
            "telemetry_heading_deg": telemetry.heading_deg,
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
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
