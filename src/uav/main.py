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
from uav.core.control.simple_fixedwing import SimpleFixedWingController, derive_gains
from uav.core.guidance import PassthroughGuidance
from uav.core.flight_engine import FlightEngine
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
        # integral_limit=50000 ft·s: lets integrator actually accumulate
        # on sustained errors (default 1.0 saturated in one tick at 200ft
        # alt error, leaving ki contribution effectively zero).
        altitude_pid = PID(0.001, 0.00008, 0.006, integral_limit=50000.0)
        # Airspeed PID: kp=0.05 so a ±10kt error swings throttle by ±0.5 —
        # enough to drive throttle to idle (from 0.55 cruise base) with a
        # modest -10kt error, and to full thrust with a +10kt error.
        # Previous kp=0.025 was too soft: plane -15kt below target AND
        # throttle stuck at 0.24 (flight 20260419_173924, DESCENT keyframes),
        # because the PID settled at equilibrium rather than winning against
        # the cruise-throttle feedforward. The user's architectural
        # expectation: throttle tracks target speed, going up or down to
        # whatever it takes — NOT pinned at idle during descent, NOT
        # settling mid-range when hot.
        # Integral floor (0.002) eliminates steady-state speed offset.
        airspeed_pid = PID(
            max(pid_s.get("kp", 0.05), 0.05),
            max(pid_s.get("ki", 0.002), 0.002),
            pid_s.get("kd", 0.004),
            integral_limit=20.0,  # lets integrator wind up for sustained error
        )
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
            max(float(safety_cfg.get("airspeed_kp", 0.05)), 0.05),
            max(float(safety_cfg.get("airspeed_ki", 0.002)), 0.002),
            float(safety_cfg.get("airspeed_kd", 0.004)),
            integral_limit=20.0,
        )
        print(f"[PEREGRINE] PID gains from YAML fallback")

    # ── Derive control gains from calibration (or use defaults) ──
    control_gains = derive_gains(envelope) if envelope else None
    if control_gains and envelope and getattr(envelope, 'calibration_confidence', 0) >= 0.3:
        print(f"[PEREGRINE] Control gains derived from calibration "
              f"(confidence={envelope.calibration_confidence:.0%})")
    else:
        print("[PEREGRINE] Using default control gains (uncalibrated)")

    # ── Altitude→throttle law: tune per-airframe from the yaml ──
    # airframes/<id>.yaml may carry an `alt_throttle:` block. Any key
    # present overrides the ControlGains default; absent keys keep it.
    at_cfg = airframe.get("alt_throttle") or {}
    if at_cfg:
        from uav.core.control.simple_fixedwing import ControlGains as _CG
        base_g = control_gains or _CG()
        control_gains = _CG(
            bank_per_hdg_error=base_g.bank_per_hdg_error,
            bank_inner_kp=base_g.bank_inner_kp,
            bank_inner_kd=base_g.bank_inner_kd,
            alt_throttle_kp=float(at_cfg.get("kp", base_g.alt_throttle_kp)),
            alt_throttle_ki=float(at_cfg.get("ki", base_g.alt_throttle_ki)),
            alt_throttle_kd=float(at_cfg.get("kd", base_g.alt_throttle_kd)),
            alt_throttle_p_clamp=float(
                at_cfg.get("p_clamp", base_g.alt_throttle_p_clamp)),
        )
        print(f"[PEREGRINE] Alt→throttle law from yaml: "
              f"kp={control_gains.alt_throttle_kp} "
              f"ki={control_gains.alt_throttle_ki} "
              f"kd={control_gains.alt_throttle_kd}")

    throttle_cfg = airframe["throttle"]
    controller = SimpleFixedWingController(
        heading_pid=heading_pid,
        altitude_pid=altitude_pid,
        airspeed_pid=airspeed_pid,
        cruise_throttle=throttle_cfg["cruise"],
        gains=control_gains,
    )

    safety_cfg = cfg.get("safety", {})
    safety = SafetyLimits()  # hardware-only clamps; no config knobs

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
    # ── Only one flight director: the ribbon ──
    mode_manager = FlightEngine(ctx=ctx)
    print("[PEREGRINE] Director: Ribbon")

    guidance = PassthroughGuidance()
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
        # Push the sync queue NOW, synchronously, before the process
        # exits. Without this we relied on the `finally` block at the
        # bottom of main() to flush — but if the daemon's SIGTERM
        # timeout fires before that block runs, the flight stays as
        # `in_progress` in Supabase forever even though the local
        # SQLite is correct. Pushing here closes that race window.
        try:
            from uav.db.sync import push_pending, stop_background_sync
            stop_background_sync()
            push_result = push_pending()
            if push_result.get("pushed", 0) > 0:
                print(
                    f"[PEREGRINE] Signal-handler sync: pushed "
                    f"{push_result['pushed']} items"
                )
        except Exception as e:
            print(f"[PEREGRINE] Signal-handler sync failed: {e}")
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
