from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from typing import Any, Dict

import yaml
from importlib import resources

from uav.core.autopilot import Autopilot
from uav.core.control.pid import PID
from uav.core.control.fixedwing_controller import FixedWingController
from uav.core.control.simple_fixedwing import derive_gains
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


def _apply_envelope(envelope, airframe: Dict[str, Any]) -> None:
    """Apply an AircraftEnvelope's values onto the airframe dict."""
    airframe["speeds_kts"]["v_stall"] = envelope.v_stall_clean
    airframe["speeds_kts"]["v_rotate"] = envelope.v_rotate
    airframe["speeds_kts"]["v_climb"] = envelope.v_best_climb
    airframe["speeds_kts"]["v_cruise"] = envelope.v_cruise
    airframe["speeds_kts"]["v_approach"] = envelope.v_approach
    airframe["speeds_kts"]["v_land"] = envelope.v_land
    airframe["speeds_kts"]["v_never_exceed"] = envelope.v_never_exceed
    airframe["rates_fpm"]["climb"] = envelope.best_climb_fpm
    airframe["throttle"]["cruise"] = envelope.cruise_throttle
    airframe["_envelope"] = envelope
    airframe["_envelope_source"] = envelope.source
    airframe["_envelope_confidence"] = envelope.confidence


def _load_envelope(icao_type: str, airframe: Dict[str, Any]) -> Dict[str, Any]:
    """Load aircraft envelope from local SQLite database.

    SQLite is the primary data source (zero latency).
    Background sync keeps it in agreement with Supabase.
    Falls back to YAML if SQLite load fails (shouldn't happen).
    """
    try:
        from uav.db.aircraft_loader import load_aircraft
        envelope = load_aircraft(icao_type)
        _apply_envelope(envelope, airframe)
        return airframe
    except Exception as e:
        print(f"[PEREGRINE] SQLite load failed ({e}), using YAML fallback")
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
    parser.add_argument(
        "--wait-for-fly",
        action="store_true",
        default=True,
        help="Wait for 'fly' command from app before starting takeoff",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Enter interactive training mode — you fly, the AI learns from you",
    )
    parser.add_argument(
        "--train-skip-to",
        type=str,
        default=None,
        help="Skip to a specific skill in training mode (takeoff/climb/turn/cruise/descend/land)",
    )
    parser.add_argument(
        "--train-chain",
        action="store_true",
        default=False,
        help="Skip training, run full AI chain flight with existing models",
    )
    parser.add_argument(
        "--director",
        choices=["pid", "ribbon", "rl", "blended"],
        default="ribbon",
        help="Flight director: ribbon (default, V2 path follower), pid (V1 reactive), rl (full RL), blended (PID+RL per phase)",
    )
    parser.add_argument(
        "--rl-model",
        default=None,
        help="Path to .npz RL model weights (required for --director rl or blended)",
    )
    parser.add_argument(
        "--rl-phases",
        default="APPROACH,LAND",
        help="Comma-separated phases for RL in blended mode (default: APPROACH,LAND)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Initialize local SQLite database + background Supabase sync ──
    # Run in a background thread so it never blocks startup
    import threading
    def _init_db_and_sync():
        try:
            from uav.db import local_db
            from uav.db.sync import full_sync, start_background_sync
            local_db.get_connection()
            print(f"[PEREGRINE] Local database: {local_db.DB_PATH}")
            print(f"[PEREGRINE] DB stats: {local_db.get_stats()}")
            sync_result = full_sync()
            if sync_result.get("supabase_available"):
                print(f"[PEREGRINE] Supabase sync OK — pull: {sync_result['pull']}, push: {sync_result['push']}")
                start_background_sync(interval_s=30.0)
            else:
                print(f"[PEREGRINE] Supabase offline — local SQLite only")
        except Exception as e:
            print(f"[PEREGRINE] DB/sync init failed: {e}")
    db_thread = threading.Thread(target=_init_db_and_sync, daemon=True)
    db_thread.start()
    db_thread.join(timeout=10.0)  # wait max 10s, then continue regardless
    if db_thread.is_alive():
        print("[PEREGRINE] DB sync still running in background — continuing startup")

    # ── Start Supabase Broadcast (live telemetry + commands) ──
    # This runs in a background thread — connects to Supabase Realtime.
    # The autopilot object will be created below, so we use a closure
    # to forward commands once it exists.
    _autopilot_ref = [None]  # mutable ref for closure

    def _on_command(data: dict) -> None:
        ap = _autopilot_ref[0]
        if ap:
            ap.handle_command(data)

    from uav.comms import broadcast as _broadcast
    try:
        _broadcast.start(command_callback=_on_command)
    except Exception as e:
        print(f"[PEREGRINE] Broadcast: failed to start ({e})")

    # ── Load airframe: YAML file first, then overlay SQLite envelope ──
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

    # ── Training mode: bypass autopilot, run interactive skill trainer ──
    if args.train or args.train_chain:
        from uav.rl.xplane_rl_adapter import XPlaneRLAdapter
        from uav.rl.skills.training_mode import TrainingSession

        # Use RL adapter (separate RREF indices, no conflict)
        train_adapter = XPlaneRLAdapter(
            xplane_ip=ports["xplane_ip"],
            xplane_port=int(ports["xplane_port"]),
            local_port=0,  # auto-pick port
            freq_hz=int(loop_rate_hz),
        )
        time.sleep(0.5)

        print("[PEREGRINE] Entering training mode...")
        session = TrainingSession(adapter=train_adapter, loop_hz=loop_rate_hz)
        session.run(skip_to=args.train_skip_to, chain_only=args.train_chain)
        return

    # ── PID controllers from envelope (database) or YAML fallback ──
    envelope = airframe.get("_envelope")
    if envelope and hasattr(envelope, "pid_heading"):
        pid_h = envelope.pid_heading
        pid_a = envelope.pid_altitude
        pid_s = envelope.pid_airspeed
        heading_pid = PID(pid_h.get("kp", 0.004), pid_h.get("ki", 0.0001), pid_h.get("kd", 0.008))
        # Altitude PID: pitch has FULL authority over altitude.
        # kp=0.001: 100ft error → 0.10 pitch — firm, responsive
        # ki=0.00008: moderate integral to eliminate persistent offset
        # kd=0.006: STRONG damping — prevents overshoot, smooths convergence
        altitude_pid = PID(0.001, 0.00008, 0.006)
        airspeed_pid = PID(pid_s.get("kp", 0.015), pid_s.get("ki", 0.0), pid_s.get("kd", 0.004))
        print(f"[PEREGRINE] PID gains loaded from database")
    else:
        safety_cfg = cfg.get("safety", {})
        heading_pid = PID(
            float(safety_cfg.get("heading_kp", 0.004)),
            float(safety_cfg.get("heading_ki", 0.0001)),
            float(safety_cfg.get("heading_kd", 0.008)),
        )
        altitude_pid = PID(
            float(safety_cfg.get("altitude_kp", 0.0008)),
            float(safety_cfg.get("altitude_ki", 0.00005)),
            float(safety_cfg.get("altitude_kd", 0.005)),
        )
        airspeed_pid = PID(
            float(safety_cfg.get("airspeed_kp", 0.015)),
            float(safety_cfg.get("airspeed_ki", 0.0)),
            float(safety_cfg.get("airspeed_kd", 0.004)),
        )
        print(f"[PEREGRINE] PID gains from YAML fallback")

    # ── Derive control gains from calibration (or use defaults) ──
    control_gains = derive_gains(envelope) if envelope else None
    if control_gains and envelope and getattr(envelope, 'calibration_confidence', 0) >= 0.3:
        print(f"[PEREGRINE] Control gains derived from calibration "
              f"(confidence={envelope.calibration_confidence:.0%})")
    else:
        print("[PEREGRINE] Using default control gains (uncalibrated)")

    throttle_cfg = airframe["throttle"]
    controller = FixedWingController(
        heading_pid=heading_pid,
        altitude_pid=altitude_pid,
        airspeed_pid=airspeed_pid,
        cruise_throttle=throttle_cfg["cruise"],
        pitch_rate_limit_per_s=float(cfg.get("safety", {}).get("pitch_rate_limit_per_s", 0.6)),
        gains=control_gains,
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
    # ── Select flight director ──
    if args.director == "rl":
        if not args.rl_model:
            parser.error("--rl-model is required when --director=rl")
        from uav.rl.rl_director import RLFlightDirector
        mode_manager = RLFlightDirector(ctx=ctx, model_path=args.rl_model)
        print(f"[PEREGRINE] Director: RL ({args.rl_model})")
    elif args.director == "blended":
        if not args.rl_model:
            parser.error("--rl-model is required when --director=blended")
        from uav.rl.blended_director import BlendedFlightDirector
        rl_phases = set(args.rl_phases.split(","))
        mode_manager = BlendedFlightDirector(
            ctx=ctx, rl_model_path=args.rl_model, rl_phases=rl_phases,
        )
        print(f"[PEREGRINE] Director: Blended (RL phases: {rl_phases})")
    elif args.director == "ribbon":
        from uav.core.flight_engine import FlightEngine
        mode_manager = FlightEngine(ctx=ctx)
        print("[PEREGRINE] Director: Ribbon (V2 path follower)")
    else:
        mode_manager = ReactiveFlightDirector(ctx=ctx)
        print("[PEREGRINE] Director: PID")

    guidance = FixedWingGuidance()
    recorder = Recorder()

    # ── Look up aircraft ID from Supabase (for heartbeat publishing) ──
    aircraft_id = None
    try:
        from supabase import create_client
        from dotenv import load_dotenv as _ld
        _ld()
        _sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_ANON_KEY"])
        rows = _sb.table("aircraft").select("id").eq("icao_type", icao_type).limit(1).execute()
        if rows.data:
            aircraft_id = rows.data[0]["id"]
            print(f"[PEREGRINE] Aircraft ID: {aircraft_id}")
        else:
            print(f"[PEREGRINE] No aircraft found for {icao_type} — heartbeat disabled")
    except Exception as e:
        print(f"[PEREGRINE] Aircraft lookup failed: {e}")

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
        wait_for_fly_command=args.wait_for_fly,
        aircraft_id=aircraft_id,
    )

    # Wire up safety envelope for V2 (always active, protects all directors)
    from uav.core.safety_envelope import enforce_envelope
    autopilot._safety_envelope = enforce_envelope
    autopilot._v_stall = float(airframe.get("speeds_kts", {}).get("v_stall", 77.0))
    autopilot._v_ne = float(airframe.get("speeds_kts", {}).get("v_never_exceed", 250.0))

    _autopilot_ref[0] = autopilot  # wire up broadcast command handler

    # ── Signal handler: save flight data on kill/Ctrl+C ──
    def _shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[PEREGRINE] Received {sig_name} — saving flight data...")
        autopilot.stop()
        # Finalize active flight so it's not left as in_progress
        try:
            telemetry = adapter.read_telemetry()
            autopilot._finalize_active_flight(telemetry, status="aborted", reason=f"Process killed ({sig_name})")
        except Exception as e:
            print(f"[PEREGRINE] Flight finalization on signal failed: {e}")
        adapter.write_actuators(abort_actuators())
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    print(f"Starting Peregrine (local_port={ports['local_port']} -> bound {getattr(adapter,'local_port',None)}). Press Ctrl+C to stop.")
    try:
        autopilot.run(duration_s=float(args.duration_s) if args.duration_s else 0.0)
    except (KeyboardInterrupt, SystemExit):
        print("Shutting down...")
        autopilot.stop()
        # Finalize active flight if not already done by signal handler
        try:
            telemetry = adapter.read_telemetry()
            autopilot._finalize_active_flight(telemetry, status="aborted", reason="Autopilot stopped")
        except Exception as e:
            print(f"[PEREGRINE] Flight finalization failed: {e}")
        adapter.write_actuators(abort_actuators())
        time.sleep(0.1)
    finally:
        recorder.close()
        # Final sync push + close DB
        from uav.db.sync import push_pending, stop_background_sync
        try:
            stop_background_sync()
            push_result = push_pending()
            if push_result.get("pushed", 0) > 0:
                print(f"[PEREGRINE] Final sync: pushed {push_result['pushed']} items")
        except Exception:
            pass
        from uav.db import local_db as _ldb
        _ldb.close()


if __name__ == "__main__":
    main()
