from __future__ import annotations

import argparse
import logging
import time
from typing import Any, Dict

import yaml
from importlib import resources

from uav.core.autopilot import Autopilot
from uav.core.control.pid import PID
from uav.core.control.fixedwing_controller import FixedWingController
from uav.core.guidance.fixedwing_guidance import FixedWingGuidance
from uav.core.reactive_director import ReactiveFlightDirector
from uav.core.safety.limits import SafetyLimits, abort_actuators
from uav.logging.recorder import Recorder
from uav.sim.xplane_udp import XPlaneUDP
from uav.nav.select_destination import pick_nearest_airport

log = logging.getLogger(__name__)


def _load_yaml_from_path(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_yaml_resource(rel_path: str) -> Dict[str, Any]:
    p = resources.files("uav").joinpath(rel_path)
    return yaml.safe_load(p.read_text())


def load_airframe(path_or_id: str) -> Dict[str, Any]:
    if path_or_id.endswith(".yaml") or "/" in path_or_id or "\\" in path_or_id:
        return _load_yaml_from_path(path_or_id)
    return _load_yaml_resource(f"config/airframes/{path_or_id}.yaml")


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return _load_yaml_resource("config/default.yaml")
    return _load_yaml_from_path(path)


def _load_envelope(icao_type: str, airframe: Dict[str, Any]) -> Dict[str, Any]:
    """Load aircraft envelope from Supabase, overlay onto airframe YAML.

    If Supabase is unreachable, the airframe YAML acts as fallback.
    If Supabase has learned values, they override the YAML seeds.

    Returns the airframe dict with speeds_kts updated from the envelope.
    """
    try:
        from uav.db.aircraft_loader import load_aircraft
        envelope = load_aircraft(icao_type)

        # Override YAML speeds with database values (learned or seed)
        airframe["speeds_kts"]["v_stall"] = envelope.v_stall_clean
        airframe["speeds_kts"]["v_rotate"] = envelope.v_rotate
        airframe["speeds_kts"]["v_climb"] = envelope.v_best_climb
        airframe["speeds_kts"]["v_cruise"] = envelope.v_cruise
        airframe["speeds_kts"]["v_approach"] = envelope.v_approach
        airframe["speeds_kts"]["v_land"] = envelope.v_land
        airframe["speeds_kts"]["v_never_exceed"] = envelope.v_never_exceed

        # Override rates
        airframe["rates_fpm"]["climb"] = envelope.best_climb_fpm

        # Override throttle cruise setting
        airframe["throttle"]["cruise"] = envelope.cruise_throttle

        # Store the full envelope object for reference
        airframe["_envelope"] = envelope
        airframe["_envelope_source"] = envelope.source
        airframe["_envelope_confidence"] = envelope.confidence

        print(f"[PEREGRINE] Envelope loaded from Supabase ({envelope.source}, "
              f"{envelope.confidence:.0%} confidence)")

    except Exception as e:
        print(f"[PEREGRINE] Supabase unavailable, using YAML fallback: {e}")
        airframe["_envelope"] = None
        airframe["_envelope_source"] = "yaml_fallback"
        airframe["_envelope_confidence"] = 0.0

    return airframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Peregrine Autopilot (X-Plane UDP)")
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

    # ── Load airframe: YAML file first, then overlay Supabase envelope ──
    icao_type = cfg.get("airframe", {}).get("icao_type", "SF50")
    airframe_id = icao_type.lower()  # yaml file is named sf50.yaml
    airframe = load_airframe(airframe_id)
    airframe = _load_envelope(icao_type, airframe)

    ports = cfg["ports"]
    loop_rate_hz = float(cfg["loop"]["rate_hz"])

    adapter = XPlaneUDP(
        xplane_ip=ports["xplane_ip"],
        xplane_port=int(ports["xplane_port"]),
        local_port=int(ports["local_port"]),
        freq_hz=int(loop_rate_hz),
    )

    # ── PID controllers from envelope (database) or YAML fallback ──
    envelope = airframe.get("_envelope")
    if envelope and hasattr(envelope, "pid_heading"):
        pid_h = envelope.pid_heading
        pid_a = envelope.pid_altitude
        pid_s = envelope.pid_airspeed
        heading_pid = PID(pid_h.get("kp", 0.003), pid_h.get("ki", 0.0), pid_h.get("kd", 0.002))
        altitude_pid = PID(pid_a.get("kp", 0.005), pid_a.get("ki", 0.0), pid_a.get("kd", 0.002))
        airspeed_pid = PID(pid_s.get("kp", 0.015), pid_s.get("ki", 0.0), pid_s.get("kd", 0.004))
        print(f"[PEREGRINE] PID gains loaded from database")
    else:
        safety_cfg = cfg.get("safety", {})
        heading_pid = PID(
            float(safety_cfg.get("heading_kp", 0.003)),
            float(safety_cfg.get("heading_ki", 0.0)),
            float(safety_cfg.get("heading_kd", 0.002)),
        )
        altitude_pid = PID(
            float(safety_cfg.get("altitude_kp", 0.005)),
            float(safety_cfg.get("altitude_ki", 0.0)),
            float(safety_cfg.get("altitude_kd", 0.002)),
        )
        airspeed_pid = PID(
            float(safety_cfg.get("airspeed_kp", 0.015)),
            float(safety_cfg.get("airspeed_ki", 0.0)),
            float(safety_cfg.get("airspeed_kd", 0.004)),
        )
        print(f"[PEREGRINE] PID gains from YAML fallback")

    throttle_cfg = airframe["throttle"]
    controller = FixedWingController(
        heading_pid=heading_pid,
        altitude_pid=altitude_pid,
        airspeed_pid=airspeed_pid,
        cruise_throttle=throttle_cfg["cruise"],
        pitch_rate_limit_per_s=float(cfg.get("safety", {}).get("pitch_rate_limit_per_s", 0.6)),
    )

    safety_cfg = cfg.get("safety", {})
    safety = SafetyLimits(
        throttle_min=float(safety_cfg.get("throttle_min", 0.0)),
        throttle_max=float(safety_cfg.get("throttle_max", 1.0)),
        max_roll=airframe["limits"]["max_roll_cmd"],
        max_pitch=airframe["limits"]["max_pitch_cmd"],
        max_yaw=airframe["limits"]["max_yaw_cmd"],
        brake_min=float(safety_cfg.get("brake_min", 0.0)),
        brake_max=float(safety_cfg.get("brake_max", 1.0)),
    )

    ctx = {
        "controller": cfg.get("safety", {}),
        "targets": cfg.get("targets", {"target_alt_ft": 10000.0, "target_hdg_deg": 90.0, "target_airspeed_kts": 200.0}),
        "mode": cfg.get("mode", {}),
        "takeoff": cfg.get("takeoff", {}),
        "climb": cfg.get("climb", {}),
        "airframe": airframe,
        "nav": cfg.get("nav", {}),
        "destination": None,
    }
    mode_manager = ReactiveFlightDirector(ctx=ctx)

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
        telemetry_timeout_s=float(safety_cfg.get("telemetry_timeout_s", 2.0)),
        icao_type=icao_type,
    )

    print(f"Starting Peregrine (local_port={ports['local_port']} -> bound {getattr(adapter,'local_port',None)}). Press Ctrl+C to stop.")
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
