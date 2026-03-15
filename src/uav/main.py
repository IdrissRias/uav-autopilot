from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import yaml
from importlib import resources

from uav.core.autopilot import Autopilot
from uav.core.control.pid import PID
from uav.core.control.fixedwing_controller import FixedWingController
from uav.core.guidance.fixedwing_guidance import FixedWingGuidance
from uav.core.mode_manager import ModeManager
from uav.core.safety.limits import SafetyLimits, abort_actuators
from uav.logging.recorder import Recorder
from uav.sim.xplane_udp import XPlaneUDP
from uav.nav.select_destination import pick_nearest_airport


def _load_yaml_from_path(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_yaml_resource(rel_path: str) -> Dict[str, Any]:
    # rel_path is within the uav package, e.g. "config/default.yaml"
    p = resources.files("uav").joinpath(rel_path)
    return yaml.safe_load(p.read_text())


def load_airframe(path_or_id: str) -> Dict[str, Any]:
    # Accept either an explicit path or an airframe id like "c172".
    if path_or_id.endswith(".yaml") or "/" in path_or_id or "\\" in path_or_id:
        return _load_yaml_from_path(path_or_id)
    return _load_yaml_resource(f"config/airframes/{path_or_id}.yaml")


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return _load_yaml_resource("config/default.yaml")
    return _load_yaml_from_path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="UAV Autopilot (X-Plane UDP)")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML (defaults to bundled uav/config/default.yaml)",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="If >0, stop after this many seconds (useful for automated test loops)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    airframe_id = cfg.get("airframe", {}).get("id", "c172")
    airframe = load_airframe(airframe_id)

    ports = cfg["ports"]
    loop_rate_hz = float(cfg["loop"]["rate_hz"])

    adapter = XPlaneUDP(
        xplane_ip=ports["xplane_ip"],
        xplane_port=int(ports["xplane_port"]),
        local_port=int(ports["local_port"]),
        freq_hz=int(loop_rate_hz),
    )

    controller_cfg = cfg["controller"]
    throttle_cfg = airframe["throttle"]
    heading_pid = PID(
        controller_cfg["heading_kp"],
        controller_cfg["heading_ki"],
        controller_cfg["heading_kd"],
    )
    altitude_pid = PID(
        controller_cfg["altitude_kp"],
        controller_cfg["altitude_ki"],
        controller_cfg["altitude_kd"],
    )
    airspeed_pid = PID(
        controller_cfg["airspeed_kp"],
        controller_cfg["airspeed_ki"],
        controller_cfg["airspeed_kd"],
    )

    controller = FixedWingController(
        heading_pid=heading_pid,
        altitude_pid=altitude_pid,
        airspeed_pid=airspeed_pid,
        cruise_throttle=throttle_cfg["cruise"],
        pitch_rate_limit_per_s=float(controller_cfg.get("pitch_rate_limit_per_s", 0.6)),
    )

    safety_cfg = cfg["safety"]
    safety = SafetyLimits(
        throttle_min=safety_cfg["throttle_min"],
        throttle_max=safety_cfg["throttle_max"],
        max_roll=airframe["limits"]["max_roll_cmd"],
        max_pitch=airframe["limits"]["max_pitch_cmd"],
        max_yaw=airframe["limits"]["max_yaw_cmd"],
        brake_min=safety_cfg.get("brake_min", 0.0),
        brake_max=safety_cfg.get("brake_max", 1.0),
    )

    ctx = {
        "controller": controller_cfg,
        "targets": cfg["targets"],
        "mode": cfg["mode"],
        "takeoff": cfg.get("takeoff", {}),
        "climb": cfg.get("climb", {}),
        "airframe": airframe,
        "nav": cfg.get("nav", {}),
        "destination": None,
    }
    mode_cfg = cfg["mode"]
    mode_manager = ModeManager(
        start_mode=mode_cfg["start_mode"],
        ctx=ctx,
        auto_start=bool(mode_cfg.get("auto_start", True)),
        has_destination=bool(mode_cfg.get("has_destination", False)),
    )

    guidance = FixedWingGuidance()
    recorder = Recorder()

    autopilot = Autopilot(
        adapter=adapter,
        controller=controller,
        guidance=guidance,
        mode_manager=mode_manager,
        safety=safety,
        recorder=recorder,
        loop_rate_hz=loop_rate_hz,
        telemetry_timeout_s=safety_cfg["telemetry_timeout_s"],
    )

    print(f"Starting autopilot (local_port={ports['local_port']} -> bound {getattr(adapter,'local_port',None)}). Press Ctrl+C to stop.")
    try:
        autopilot.run(duration_s=float(args.duration_s) if args.duration_s else 0.0)
    except KeyboardInterrupt:
        print("Shutting down...")
        autopilot.stop()
        adapter.write_actuators(abort_actuators())
        time.sleep(0.1)
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
