from __future__ import annotations

import time

from uav.core.control.simple_fixedwing import SimpleFixedWingController
from uav.core.guidance.simple_guidance import SimpleGuidance
from uav.core.safety.limits import SafetyLimits, abort_actuators
from uav.core.safety.failsafe import is_telemetry_stale
from uav.logging.recorder import Recorder
from uav.scoring.flight_scorer import FlightScorer
from uav.scoring.accuracy_tracker import AccuracyTracker
from uav.learning.flight_observer import FlightObserver
from uav.sim.adapter_base import SimAdapter
from uav.sim.types import Actuators, Telemetry
from uav.comms import broadcast
from uav.nav.runway_detect import detect_runway, RunwayDetection
from uav.db import local_db


class Autopilot:
    def __init__(
        self,
        adapter: SimAdapter,
        controller: SimpleFixedWingController,
        guidance: SimpleGuidance,
        mode_manager: object,
        safety: SafetyLimits,
        recorder: Recorder,
        loop_rate_hz: float,
        telemetry_timeout_s: float,
        icao_type: str = "SF50",
        wait_for_fly_command: bool = False,
        aircraft_id: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.controller = controller
        self.guidance = guidance
        self.mode_manager = mode_manager
        self.safety = safety
        self.recorder = recorder
        self.loop_rate_hz = loop_rate_hz
        self.telemetry_timeout_s = telemetry_timeout_s
        self._running = False
        self._last_step = time.time()
        self._reset_last_ts = 0.0
        self._reset_cooldown_s = 5.0
        self._monitor: dict = {}  # timers for phase-limit watchdog
        self._scorer: FlightScorer | None = None
        self._accuracy: AccuracyTracker = AccuracyTracker()
        self._landed = False        # True once we've scored the current flight
        self._prev_phase = "GROUND"
        self._icao_type = icao_type
        # Flight observer: watches every tick, learns aircraft performance
        self._observer: FlightObserver | None = None
        self._observer_finalized = False

        # ── Pre-flight / Broadcast ──
        self._wait_for_fly = wait_for_fly_command
        self._fly_command_received = not wait_for_fly_command  # if not waiting, auto-fly
        self._fly_destination: dict | None = None  # set by app command
        self._runway_detection: RunwayDetection | None = None
        self._preflight_broadcasted = False
        self._broadcast_tick = 0  # counter for throttling broadcast rate
        self._heartbeat_next = 0.0  # next heartbeat time
        self._aircraft_id: str | None = aircraft_id
        # Flight DB record
        self._flight_id: str | None = None
        self._snapshot_next: float = 0.0  # next telemetry snapshot time
        self._snapshot_tick: int = 0      # tick counter for snapshots
        self._end_flight_requested = False  # set by app "end_flight" command

        # Safety envelope (V2): descent rate + stall + overspeed + bank
        self._safety_envelope = None  # set externally if desired
        self._v_stall = 77.0
        self._v_ne = 250.0

    def handle_command(self, data: dict) -> None:
        """Handle a command from the app (via Supabase Broadcast).

        Called from the broadcast thread — must be thread-safe.
        """
        action = data.get("action", "")
        print(f"[COMMAND] Received: {action} → {data}")

        if action == "fly":
            # App says go — optionally with a destination.
            # Only set _fly_destination here; the main loop flips
            # _fly_command_received once it has processed the destination.
            # This prevents stale broadcast replays from bypassing wait-for-fly.
            self._fly_destination = data.get("destination")  # {"icao": "KFFM", ...}
            print(f"[COMMAND] FLY command received" +
                  (f" → dest {self._fly_destination}" if self._fly_destination else ""))

        elif action == "abort":
            print("[COMMAND] ABORT received — stopping flight")
            self._running = False

        elif action == "set_destination":
            dest = data.get("destination")
            if dest and self.mode_manager.ctx:
                self.mode_manager.ctx["destination"] = dest
                print(f"[COMMAND] Destination changed to {dest}")

        elif action == "end_flight":
            print("[COMMAND] END FLIGHT received — finalizing and resetting")
            self._end_flight_requested = True

        elif action == "set_cruise_throttle":
            value = data.get("value")
            if value is not None:
                airframe = self.mode_manager.ctx.get("airframe", {})
                airframe["throttle"]["cruise"] = float(value)
                print(f"[COMMAND] Cruise throttle → {value}")

        elif action == "calibrate":
            print("[COMMAND] CALIBRATE received — switching to calibration mode")
            # Unlock the fly gate so the autopilot loop runs
            self._fly_command_received = True
            self._start_calibration()

    def _start_calibration(self) -> None:
        """Switch to calibration mode — automated maneuvers to measure sensitivity."""
        from uav.core.calibration_director import CalibrationDirector
        from uav.learning.flight_observer import SensitivityTracker
        from uav.learning.envelope_updater import update_envelope

        # Ensure observer exists (calibration needs it for sensitivity tracking)
        if not self._observer:
            self._observer = FlightObserver(self._icao_type)
            self._observer_finalized = False
            print(f"[CALIBRATION] Created flight observer for {self._icao_type}")

        tracker = self._observer._sensitivity

        def _on_progress(step: int, total: int, label: str, confidence: float) -> None:
            print(f"[CALIBRATION] Step {step}/{total}: {label} (confidence: {confidence:.0%})")
            broadcast.publish_status({
                "type": "calibration_progress",
                "step": step,
                "total": total,
                "label": label,
                "confidence": confidence,
            })

        def _on_complete(cal_data: dict) -> None:
            print(f"[CALIBRATION] Complete — saving to database")
            # Save via envelope updater (merges into learned envelope)
            try:
                update_envelope(self._icao_type, {"calibration": cal_data})
                print(f"[CALIBRATION] Saved to database for {self._icao_type}")
            except Exception as e:
                print(f"[CALIBRATION] Save failed: {e}")

            broadcast.publish_status({
                "type": "calibration_complete",
                "confidence": cal_data.get("confidence", 0),
                "sensitivity": {
                    "pitch": cal_data.get("pitch_sensitivity", 0),
                    "roll": cal_data.get("roll_sensitivity", 0),
                    "yaw": cal_data.get("yaw_sensitivity", 0),
                    "throttle": cal_data.get("throttle_sensitivity", 0),
                },
            })

        ctx = getattr(self.mode_manager, 'ctx', {})
        self._calibration_director = CalibrationDirector(
            ctx=ctx,
            sensitivity_tracker=tracker,
            on_progress=_on_progress,
            on_complete=_on_complete,
        )
        # Swap the mode manager to calibration director
        self._original_mode_manager = self.mode_manager
        self.mode_manager = self._calibration_director
        print("[CALIBRATION] Calibration director active")

    def _finalize_active_flight(self, telemetry, status: str = "aborted", reason: str = "") -> None:
        """Finalize the current in-progress flight (if any) and clear flight state."""
        if not self._flight_id:
            return
        try:
            # Build phase timeline from observer
            phase_tl = ""
            if self._observer:
                try:
                    compiled = self._observer.compile()
                    pt = compiled.get("phase_timeline", {})
                    phase_tl = ",".join(f"{p}:{int(d)}" for p, d in pt.items() if d > 0)
                except Exception:
                    pass

            # Partial scores if scorer is available
            duration_s = None
            if self._scorer and hasattr(self._scorer, "_start_time") and self._scorer._start_time:
                duration_s = time.time() - self._scorer._start_time

            local_db.finalize_flight(
                self._flight_id,
                status=status,
                duration_s=duration_s,
                phase_timeline=phase_tl if phase_tl else None,
            )

            # Log end event
            local_db.log_event(
                self._flight_id, "stopped",
                message=reason or f"Flight {status}",
                altitude_ft=telemetry.altitude_ft,
                airspeed_kts=telemetry.airspeed_kts,
                heading_deg=telemetry.heading_deg,
                lat=telemetry.lat_deg if telemetry.has_position() else None,
                lon=telemetry.lon_deg if telemetry.has_position() else None,
                phase=self.mode_manager.name if self.mode_manager else "UNKNOWN",
            )
            print(f"[PEREGRINE] Flight {self._flight_id[:8]}... finalized → {status}")
        except Exception as e:
            print(f"[PEREGRINE] Flight finalization failed: {e}")

        # Clear flight state
        self._flight_id = None
        self._snapshot_next = 0.0
        self._snapshot_tick = 0
        self._scorer = None
        self._observer = None
        self._accuracy = type(self._accuracy)()  # fresh accuracy tracker

    def _detect_runway_once(self, telemetry) -> None:
        """Detect which runway we're on (called once during pre-flight)."""
        if self._runway_detection is not None:
            return
        if not telemetry.has_position():
            return

        self._runway_detection = detect_runway(
            telemetry.lat_deg, telemetry.lon_deg, telemetry.heading_deg
        )
        print(f"[PREFLIGHT] {self._runway_detection.summary}")

    def _broadcast_preflight(self, telemetry) -> None:
        """Broadcast pre-flight status to the app (called every ~2s)."""
        rwy = self._runway_detection
        # Get takeoff roll distance from aircraft envelope
        takeoff_roll_ft = 2000.0  # default
        ctx = getattr(self.mode_manager, "ctx", None) or {}
        airframe = ctx.get("airframe", {})
        if airframe.get("takeoff_roll_ft"):
            takeoff_roll_ft = float(airframe["takeoff_roll_ft"])
        data = {
            "state": "preflight",
            "runway_detected": rwy.detected if rwy else False,
            "airport_icao": rwy.airport_icao if rwy else "",
            "airport_name": rwy.airport_name if rwy else "",
            "runway": rwy.runway_designator if rwy else "",
            "runway_heading": rwy.runway_heading_deg if rwy else 0,
            "runway_length_ft": rwy.runway_length_ft if rwy else 0,
            "aircraft": self._icao_type,
            "lat": telemetry.lat_deg if telemetry.has_position() else 0,
            "lon": telemetry.lon_deg if telemetry.has_position() else 0,
            "heading": telemetry.heading_deg,
            "waiting_for_fly": self._wait_for_fly and not self._fly_command_received,
            # For "just take off" mode — app shows required distances
            "required_takeoff_roll_ft": takeoff_roll_ft,
            "required_runway_width_ft": 75.0,  # minimum safe width
            "force_takeoff_allowed": True,  # always allow force takeoff
        }
        broadcast.publish_preflight(data)
        self._preflight_broadcasted = True

    def _broadcast_telemetry(self, telemetry, targets, act, phase: str) -> None:
        """Broadcast live telemetry to the app (every 3rd tick = ~5Hz)."""
        self._broadcast_tick += 1
        if self._broadcast_tick % 3 != 0:
            return

        ctx = self.mode_manager.ctx or {}
        dest = ctx.get("destination")

        data = {
            "tick": self._broadcast_tick,
            "phase": phase,
            "position": {
                "lat": telemetry.lat_deg if telemetry.has_position() else 0,
                "lon": telemetry.lon_deg if telemetry.has_position() else 0,
                "alt_ft": telemetry.altitude_ft,
                "agl_ft": telemetry.agl_m * 3.281 if telemetry.agl_m else 0,
                "heading_deg": telemetry.heading_deg,
            },
            "motion": {
                "airspeed_kts": telemetry.airspeed_kts,
                "groundspeed_kts": telemetry.groundspeed_kts if hasattr(telemetry, "groundspeed_kts") else 0,
                "vs_fpm": telemetry.vs_fpm if hasattr(telemetry, "vs_fpm") else 0,
                "pitch_deg": telemetry.pitch_deg,
                "roll_deg": telemetry.roll_deg,
            },
            "controls": {
                "throttle": act.throttle,
                "pitch_cmd": act.pitch,
                "roll_cmd": act.roll,
                "yaw_cmd": act.yaw,
                "flaps": act.flap_ratio if hasattr(act, "flap_ratio") else 0,
                "gear": act.gear_down,
                "brake": act.brake_ratio,
            },
        }

        if targets:
            data["targets"] = {
                "heading_deg": targets.heading_deg if hasattr(targets, "heading_deg") else 0,
                "altitude_ft": targets.altitude_ft if hasattr(targets, "altitude_ft") else 0,
                "airspeed_kts": targets.airspeed_kts if hasattr(targets, "airspeed_kts") else 0,
                "throttle": targets.throttle if hasattr(targets, "throttle") else None,
            }

        # Accuracy tracking
        data["accuracy"] = {
            "instant_pct": round(self._accuracy.instant_accuracy, 1),
            "flight_pct": round(self._accuracy.flight_accuracy, 1),
        }
        tick = self._accuracy.last_tick
        if tick and tick.metric_scores:
            data["accuracy"]["metrics"] = {
                k: round(v * 100, 1) for k, v in tick.metric_scores.items()
            }

        if dest:
            from uav.nav.geo import haversine_m, bearing_deg
            dist_m = haversine_m(telemetry.lat_deg, telemetry.lon_deg,
                                 float(dest["lat"]), float(dest["lon"]))
            brg = bearing_deg(telemetry.lat_deg, telemetry.lon_deg,
                              float(dest["lat"]), float(dest["lon"]))
            dist_nm = dist_m / 1852.0
            gs = telemetry.groundspeed_kts if hasattr(telemetry, "groundspeed_kts") and telemetry.groundspeed_kts > 10 else telemetry.airspeed_kts
            eta_s = (dist_nm / gs * 3600) if gs > 10 else 0

            data["nav"] = {
                "dest_icao": dest.get("icao", ""),
                "bearing_deg": round(brg, 1),
                "dist_nm": round(dist_nm, 2),
                "eta_s": round(eta_s),
            }

        broadcast.publish_telemetry(data)

    def _build_and_broadcast_plan(self, telemetry, ctx, dest, ground_msl_ft, initial_target, dist_nm) -> None:
        """Build flight plan (V2 ribbon or V1 waypoints), create scorer, observer, flight record."""
        # ── Try V2 ribbon path first ──
        from uav.core.flight_engine import FlightEngine
        if isinstance(self.mode_manager, FlightEngine):
            try:
                from uav.nav.flight_plan_v2 import plan_path, format_ribbon
                dest_rwy = ctx.get("dest_runway")
                rwy = self._runway_detection
                ribbon = plan_path(
                    dep_lat=telemetry.lat_deg,
                    dep_lon=telemetry.lon_deg,
                    dep_alt_ft=telemetry.altitude_ft,
                    dep_heading=rwy.runway_heading_deg if rwy and rwy.detected else telemetry.heading_deg,
                    dest_lat=float(dest["lat"]),
                    dest_lon=float(dest["lon"]),
                    dest_alt_ft=float(dest_rwy["elevation_ft"]) if dest_rwy and dest_rwy.get("elevation_ft") is not None else ground_msl_ft,
                    dest_rwy_heading=dest_rwy["heading"] if dest_rwy else None,
                    dest_threshold_lat=dest_rwy["threshold_lat"] if dest_rwy else None,
                    dest_threshold_lon=dest_rwy["threshold_lon"] if dest_rwy else None,
                    cruise_alt_ft=initial_target,
                    v_rotate=float(ctx["airframe"]["speeds_kts"].get("v_rotate", 90.0)),
                    v_climb=float(ctx["airframe"]["speeds_kts"].get("v_climb", 160.0)),
                    v_cruise=float(ctx["airframe"]["speeds_kts"].get("v_cruise", 200.0)),
                    v_approach=float(ctx["airframe"]["speeds_kts"].get("v_approach", 83.0)),
                    v_land=float(ctx["airframe"]["speeds_kts"].get("v_land", 77.0)),
                    climb_fpm=float(ctx["airframe"].get("rates_fpm", {}).get("climb", 1600.0)),
                    takeoff_roll_ft=float(ctx["airframe"].get("takeoff_roll_ft", 2000.0)),
                )
                # Pre-build ribbon so FlightEngine doesn't rebuild it
                self.mode_manager._ribbon = ribbon
                self.mode_manager._built = True
                self.mode_manager._idx = 0
                print(format_ribbon(ribbon))

                # Broadcast ribbon waypoints to the app for map display
                # Sample every ~10 points to keep the broadcast small
                step = max(1, len(ribbon.points) // 50)
                try:
                    broadcast.publish_status({
                        "event": "flight_plan",
                        "waypoints": [
                            {
                                "name": p.phase,
                                "lat": round(p.lat, 6),
                                "lon": round(p.lon, 6),
                                "alt_ft": round(p.alt_ft, 0),
                                "speed_kts": round(p.speed_kts, 1),
                                "phase": p.phase,
                                "heading": round(p.heading_deg, 1),
                            }
                            for i, p in enumerate(ribbon.points) if i % step == 0
                        ],
                    })
                except Exception:
                    pass
            except Exception as e:
                print(f"[RIBBON] Flight plan generation failed: {e}")
                import traceback; traceback.print_exc()
        else:
            # ── V1 flight plan (legacy) ──
            try:
                from uav.nav.flight_plan import build_flight_plan, format_plan
                dest_rwy = ctx.get("dest_runway")
                rwy = self._runway_detection
                plan = build_flight_plan(
                    dep_lat=telemetry.lat_deg,
                    dep_lon=telemetry.lon_deg,
                    dep_alt_ft=telemetry.altitude_ft,
                    dep_heading=rwy.runway_heading_deg if rwy and rwy.detected else telemetry.heading_deg,
                    dest_lat=float(dest["lat"]),
                    dest_lon=float(dest["lon"]),
                    dest_alt_ft=float(ctx.get("dest_runway", {}).get("elevation_ft", ground_msl_ft) or ground_msl_ft),
                    dest_rwy_heading=ctx.get("dest_runway", {}).get("heading"),
                    dest_threshold_lat=ctx.get("dest_runway", {}).get("threshold_lat"),
                    dest_threshold_lon=ctx.get("dest_runway", {}).get("threshold_lon"),
                    v_rotate=float(ctx["airframe"]["speeds_kts"].get("v_rotate", 90.0)),
                    v_climb=float(ctx["airframe"]["speeds_kts"].get("v_climb", 130.0)),
                    v_cruise=float(ctx["airframe"]["speeds_kts"].get("v_cruise", 200.0)),
                    v_approach=float(ctx["airframe"]["speeds_kts"].get("v_approach", 83.0)),
                    v_land=float(ctx["airframe"]["speeds_kts"].get("v_land", 77.0)),
                    cruise_alt_ft=initial_target,
                    takeoff_roll_ft=float(ctx["airframe"].get("takeoff_roll_ft", 2000.0)),
                )
                ctx["flight_plan"] = plan
                print(format_plan(plan))
                try:
                    broadcast.publish_status({
                        "event": "flight_plan",
                        "waypoints": [
                            {
                                "name": wp.name,
                                "lat": round(wp.lat, 6),
                                "lon": round(wp.lon, 6),
                                "alt_ft": round(wp.alt_ft, 0),
                                "speed_kts": round(wp.speed_kts, 1),
                                "phase": wp.phase,
                                "heading": round(wp.heading, 1) if wp.heading is not None else None,
                            }
                            for wp in plan.waypoints
                        ],
                    })
                except Exception:
                    pass
            except Exception as e:
                print(f"[GPS] Flight plan generation failed: {e}")
                import traceback; traceback.print_exc()

        # ── Scorer + Observer + Flight Record (shared by V1 and V2) ──
        dest_info = ctx["destination"]
        self._scorer = FlightScorer(
            dest_icao=dest_info.get("icao", "???"),
            dest_lat=float(dest_info["lat"]),
            dest_lon=float(dest_info["lon"]),
            cruise_target_ft=ctx["targets"]["target_alt_ft"],
            log_dir="logs",
        )
        self._landed = False

        self._observer = FlightObserver(self._icao_type)
        self._observer_finalized = False
        print(f"[PEREGRINE] Flight observer started for {self._icao_type}")

        # Finalize any previous in-progress flight
        if self._flight_id:
            self._finalize_active_flight(telemetry, status="aborted", reason="New flight started")

        # Create flight record in database
        try:
            rwy = self._runway_detection
            dep_icao = rwy.airport_icao if rwy and rwy.detected else None
            dep_rwy = rwy.runway_designator if rwy and rwy.detected else None
            arr_icao = dest_info.get("icao")
            arr_rwy_data = ctx.get("dest_runway")
            arr_rwy = arr_rwy_data.get("designator") if arr_rwy_data else None

            flight_rec = local_db.create_flight(
                aircraft_id=self._aircraft_id,
                dep_airport_icao=dep_icao,
                dep_runway_designator=dep_rwy,
                arr_airport_icao=arr_icao,
                arr_runway_designator=arr_rwy,
                route_distance_nm=dist_nm,
                cruise_alt_target_ft=ctx["targets"]["target_alt_ft"],
            )
            self._flight_id = flight_rec["id"]
            print(f"[PEREGRINE] Flight record created: {self._flight_id[:8]}... ({dep_icao} → {arr_icao})")

            local_db.log_event(
                self._flight_id, "takeoff_roll",
                message=f"Takeoff roll started on {dep_rwy or 'unknown'} at {dep_icao or 'unknown'}",
                altitude_ft=telemetry.altitude_ft,
                airspeed_kts=telemetry.airspeed_kts,
                heading_deg=telemetry.heading_deg,
                lat=telemetry.lat_deg if telemetry.has_position() else None,
                lon=telemetry.lon_deg if telemetry.has_position() else None,
                phase="GROUND",
            )
        except Exception as e:
            print(f"[PEREGRINE] Flight record creation failed: {e}")
            self._flight_id = None

    def stop(self) -> None:
        self._running = False

    def _check_phase_limits(self, mode: str, telemetry) -> str | None:
        """Return a failure reason string if the flight is in an unacceptable state, else None."""
        now = time.time()
        ctx = self.mode_manager.ctx
        ms = self._monitor

        if mode == "CLIMB":
            target_alt = ctx.get("targets", {}).get("target_alt_ft", telemetry.altitude_ft)
            # If we're massively above cruise target, X-Plane started mid-air at wrong altitude.
            if telemetry.altitude_ft > target_alt + 1000.0:
                if "climb_high_since" not in ms:
                    ms["climb_high_since"] = now
                elif now - ms["climb_high_since"] >= 5.0:
                    ms.pop("climb_high_since", None)
                    return f"CLIMB alt {telemetry.altitude_ft:.0f}ft >1000ft above cruise target {target_alt:.0f}ft"
            else:
                ms.pop("climb_high_since", None)

        elif mode == "CRUISE":
            target_alt = ctx.get("targets", {}).get("target_alt_ft", telemetry.altitude_ft)
            alt_error = telemetry.altitude_ft - target_alt

            # Hard upper limit: if the plane is way above target for a sustained
            # period, it's out of control.  Short overshoots during turns are normal.
            if alt_error > 800.0:
                if "high_since" not in ms:
                    ms["high_since"] = now
                elif now - ms["high_since"] >= 10.0:
                    ms.pop("high_since", None)
                    return f"CRUISE alt {telemetry.altitude_ft:.0f}ft is >800ft above target {target_alt:.0f}ft for >10s"
            else:
                ms.pop("high_since", None)

            # Sustained low altitude (altitude loss from turns not recovering).
            if alt_error < -500.0:
                if "low_since" not in ms:
                    ms["low_since"] = now
                elif now - ms["low_since"] >= 15.0:
                    ms.pop("low_since", None)
                    return f"CRUISE alt {telemetry.altitude_ft:.0f}ft >500ft below target for >15s"
            else:
                ms.pop("low_since", None)

            # Circling: stuck with large heading error for too long.
            dest = ctx.get("destination")
            if dest and telemetry.has_position():
                try:
                    from uav.nav.geo import bearing_deg as _bd
                    true_bearing = _bd(
                        telemetry.lat_deg, telemetry.lon_deg,
                        float(dest["lat"]), float(dest["lon"]),
                    )
                    raw_err = true_bearing - telemetry.heading_deg
                    while raw_err > 180.0:
                        raw_err -= 360.0
                    while raw_err < -180.0:
                        raw_err += 360.0
                    hdg_err = abs(raw_err)
                    if hdg_err > 90.0:
                        if "circle_since" not in ms:
                            ms["circle_since"] = now
                        elif now - ms["circle_since"] >= 30.0:
                            ms.pop("circle_since", None)
                            return f"CRUISE heading error {hdg_err:.0f}° for >30s (circling)"
                    else:
                        ms.pop("circle_since", None)
                except Exception:
                    pass

        elif mode == "APPROACH":
            # Airspeed is now managed by the PID, so no hard reset threshold here.
            pass

        # Clear stale timers when not in a monitored phase.
        if mode not in ("CRUISE",):
            ms.pop("low_since", None)
            ms.pop("high_since", None)
            ms.pop("circle_since", None)
        if mode not in ("CLIMB",):
            ms.pop("climb_high_since", None)

        return None

    def _safe_write(self, act: Actuators) -> None:
        self.adapter.write_actuators(act)

    def run(self, duration_s: float = 0.0) -> None:
        self._running = True
        status_next = time.time() + 1.0
        preflight_next = 0.0  # when to next broadcast preflight status
        end_at = (time.time() + duration_s) if duration_s and duration_s > 0 else 0.0

        # Clear stale status from any previous flight on startup
        if self._aircraft_id:
            try:
                broadcast.publish_heartbeat(self._aircraft_id, {"status": "preflight"})
            except Exception:
                pass
        while self._running:
            loop_start = time.time()
            telemetry = self.adapter.read_telemetry()

            # ── WAIT FOR FLY — top-level guard ──
            # If --wait-for-fly is set and no fly command received yet,
            # hold the plane with brakes and idle. Nothing else runs.
            if self._wait_for_fly and not self._fly_command_received:
                if time.time() >= status_next:
                    status_next = time.time() + 1.0
                    print(f"[WAIT-FOR-FLY] Holding... fly_dest={self._fly_destination is not None} fly_cmd={self._fly_command_received}")
                # Idle actuators: brakes on, throttle zero
                idle_act = Actuators(throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0,
                                     brake_ratio=1.0, gear_down=True)
                self._safe_write(idle_act)

                # Detect runway + broadcast preflight (so app can show preflight screen)
                if telemetry.is_valid():
                    if hasattr(telemetry, "has_position") and telemetry.has_position():
                        self._detect_runway_once(telemetry)
                    if time.time() >= preflight_next:
                        self._broadcast_preflight(telemetry)
                        preflight_next = time.time() + 2.0
                elif time.time() >= status_next:
                    print(f"[WAIT-FOR-FLY] No valid telemetry: alt={telemetry.altitude_ft} hdg={telemetry.heading_deg} pos={telemetry.lat_deg},{telemetry.lon_deg}")

                # Heartbeat so the app knows we're alive
                if self._aircraft_id and time.time() >= self._heartbeat_next:
                    self._heartbeat_next = time.time() + 5.0
                    hb = {
                        "status": "preflight",
                        "last_lat": telemetry.lat_deg if telemetry.has_position() else 0.0,
                        "last_lon": telemetry.lon_deg if telemetry.has_position() else 0.0,
                        "last_heading": round(telemetry.heading_deg, 1),
                        "altitude_ft": round(telemetry.altitude_ft, 0),
                        "airspeed_kts": 0,
                    }
                    broadcast.publish_heartbeat(self._aircraft_id, hb)

                # Accept fly command from broadcast OR DB poll
                if not self._fly_destination and self._aircraft_id:
                    # Poll DB every 3 seconds for fly_requested
                    if not hasattr(self, '_poll_next'):
                        self._poll_next = time.time() + 2.0
                    if time.time() >= self._poll_next:
                        self._poll_next = time.time() + 3.0
                        try:
                            row = broadcast.poll_fly_command(self._aircraft_id)
                            if row:
                                self._fly_destination = row
                                print(f"[COMMAND] Fly command from DB: {row}")
                        except Exception as e:
                            print(f"[POLL] Error: {e}")

                if self._fly_destination:
                    ctx = getattr(self.mode_manager, "ctx", None)
                    if ctx is not None and ctx.get("destination") is None:
                        dest = self._fly_destination
                        ctx["destination"] = {
                            "icao": dest.get("icao", "???"),
                            "name": dest.get("name", ""),
                            "lat": float(dest["lat"]),
                            "lon": float(dest["lon"]),
                        }
                        print(f"[COMMAND] Flying to {ctx['destination']['icao']}")
                        # Look up the best runway at the destination for approach alignment
                        try:
                            import sqlite3, os as _os
                            _db = sqlite3.connect(_os.path.expanduser("~/.peregrine/peregrine.db"))
                            _db.row_factory = sqlite3.Row
                            rwys = _db.execute("""
                                SELECT r.*, a.elevation_ft as airport_elevation_ft FROM runways r
                                JOIN airports a ON r.airport_id = a.id
                                WHERE a.icao_code = ?
                                ORDER BY r.length_ft DESC
                            """, (dest.get("icao", ""),)).fetchall()
                            if rwys:
                                # Pick the longest runway, use the heading closest to our approach bearing
                                from uav.nav.geo import bearing_deg as _bd2
                                approach_brg = _bd2(
                                    telemetry.lat_deg, telemetry.lon_deg,
                                    float(dest["lat"]), float(dest["lon"]),
                                ) if telemetry.has_position() else 0.0
                                best_rwy = None
                                best_diff = 999.0
                                for rwy in rwys:
                                    # Each runway has two directions (e.g. 09/27)
                                    # DB: heading_deg=090 for runway "09/27"
                                    #   threshold = runway 09 threshold (west end) — land HERE on rwy 09
                                    #   end = opposite end (east end) — land HERE on rwy 27
                                    # So: hdg1 (090) pairs with threshold, hdg2 (270) pairs with end
                                    hdg1 = float(rwy["heading_deg"])
                                    hdg2 = (hdg1 + 180.0) % 360.0
                                    for hdg, tlat, tlon in [
                                        (hdg1, rwy["threshold_lat"], rwy["threshold_lon"]),
                                        (hdg2, rwy["end_lat"], rwy["end_lon"]),
                                    ]:
                                        diff = abs(((approach_brg - hdg + 180) % 360) - 180)
                                        if diff < best_diff:
                                            best_diff = diff
                                            best_rwy = {
                                                "heading": hdg,
                                                "threshold_lat": float(tlat),
                                                "threshold_lon": float(tlon),
                                                "length_ft": float(rwy["length_ft"]),
                                                "designator": rwy["designator"],
                                                "elevation_ft": float(rwy["airport_elevation_ft"]) if rwy["airport_elevation_ft"] else None,
                                            }
                                if best_rwy:
                                    ctx["dest_runway"] = best_rwy
                                    print(f"[NAV] Landing runway: {best_rwy['designator']} hdg {best_rwy['heading']:.0f}° (approach diff {best_diff:.0f}°)")
                            _db.close()
                        except Exception as e:
                            print(f"[NAV] Runway lookup failed: {e}")
                        # Compute fixed cruise altitude based on distance
                        # (k=55: short flights stay low, long flights climb higher)
                        if telemetry.is_valid() and hasattr(telemetry, "has_position") and telemetry.has_position():
                            try:
                                from uav.nav.geo import haversine_m
                                dist_nm = haversine_m(
                                    telemetry.lat_deg, telemetry.lon_deg,
                                    float(dest["lat"]), float(dest["lon"]),
                                ) / 1852.0
                                ground_msl_ft = telemetry.altitude_ft
                                cruise_alt_agl = max(1500.0, min(5000.0, 55.0 * dist_nm))
                                initial_target = ground_msl_ft + cruise_alt_agl
                                ctx["targets"]["target_alt_ft"] = initial_target
                                ctx["takeoff_alt_ft"] = telemetry.altitude_ft
                                print(f"NAV: dist={dist_nm:.1f}nm → cruise alt {cruise_alt_agl:.0f}ft AGL ({initial_target:.0f}ft MSL) [fixed]")

                                # ── Build flight plan ──
                                self._build_and_broadcast_plan(telemetry, ctx, dest, ground_msl_ft, initial_target, dist_nm)
                            except Exception as e:
                                print(f"[NAV] Cruise alt calc failed: {e}")
                    self._fly_command_received = True
                    # Don't continue — fall through to normal loop on next iteration

                self._last_step = loop_start
                sleep_s = max(0.0, (1.0 / self.loop_rate_hz) - (time.time() - loop_start))
                time.sleep(sleep_s)
                continue

            # ── Pre-flight: detect runway + wait for fly command ──
            if getattr(self.mode_manager, "ctx", None) is not None:
                ctx = self.mode_manager.ctx
                on_ground = telemetry.agl_m < 5.0

                if on_ground and telemetry.is_valid() and hasattr(telemetry, "has_position") and telemetry.has_position():
                    self._detect_runway_once(telemetry)

                    # Broadcast pre-flight status every 2 seconds
                    if time.time() >= preflight_next and ctx.get("destination") is None:
                        self._broadcast_preflight(telemetry)
                        preflight_next = time.time() + 2.0

                    # If app sent a destination via fly command, use it
                    if self._fly_command_received and self._fly_destination and ctx.get("destination") is None:
                        dest = self._fly_destination
                        ctx["destination"] = {
                            "icao": dest.get("icao", "???"),
                            "name": dest.get("name", ""),
                            "lat": float(dest["lat"]),
                            "lon": float(dest["lon"]),
                        }
                        print(f"[COMMAND] Flying to {ctx['destination']['icao']}")

                # Wait for fly command if configured to do so
                if self._wait_for_fly and not self._fly_command_received:
                    # Don't drive anything — sit on the runway with brakes + idle
                    idle_act = Actuators(throttle=0.0, roll=0.0, pitch=0.0, yaw=0.0,
                                         brake_ratio=1.0, gear_down=True)
                    self._safe_write(idle_act)

                    # Still send heartbeat so app knows we're alive
                    if self._aircraft_id and time.time() >= self._heartbeat_next:
                        self._heartbeat_next = time.time() + 5.0
                        hb = {
                            "status": "preflight",
                            "last_lat": telemetry.lat_deg if telemetry.has_position() else 0.0,
                            "last_lon": telemetry.lon_deg if telemetry.has_position() else 0.0,
                            "last_heading": round(telemetry.heading_deg, 1),
                            "altitude_ft": round(telemetry.altitude_ft, 0),
                            "airspeed_kts": 0,
                        }
                        broadcast.publish_heartbeat(self._aircraft_id, hb)

                    self._last_step = loop_start
                    sleep_s = max(0.0, (1.0 / self.loop_rate_hz) - (time.time() - loop_start))
                    time.sleep(sleep_s)
                    continue

            # Lazy destination selection once position is available.
            # IMPORTANT: skip if waiting for fly command — don't auto-assign destination.
            if getattr(self.mode_manager, "ctx", None) is not None and self._fly_command_received:
                ctx = self.mode_manager.ctx
                on_ground = telemetry.agl_m < 5.0
                if ctx.get("destination") is None and on_ground and telemetry.is_valid() and hasattr(telemetry, "has_position") and telemetry.has_position():
                    nav = ctx.get("nav", {})
                    apt = nav.get("apt_dat_path")
                    fixed = nav.get("fixed_destination")
                    if fixed:
                        ctx["destination"] = {
                            "icao": fixed.get("icao", "???"),
                            "name": fixed.get("name", "fixed"),
                            "lat": float(fixed["lat"]),
                            "lon": float(fixed["lon"]),
                        }
                        print(f"NAV: fixed destination {ctx['destination']['icao']} ({ctx['destination']['name']})")
                    elif apt:
                        try:
                            dest = __import__("uav.nav.select_destination", fromlist=["pick_nearest_airport"]).pick_nearest_airport(
                                apt,
                                telemetry.lat_deg,
                                telemetry.lon_deg,
                                float(nav.get("min_distance_nm", 10.0)),
                                float(nav.get("max_distance_nm", 80.0)),
                            )
                            if dest:
                                ctx["destination"] = {"icao": dest.icao, "name": dest.name, "lat": dest.lat, "lon": dest.lon}
                                print(f"NAV: destination selected {dest.icao} ({dest.name})")
                        except Exception:
                            pass
                    # Compute cruise altitude: h_agl = k × D, clamped.
                    # Fixed for the whole cruise — no dynamic descent.
                    if ctx.get("destination"):
                        from uav.nav.geo import haversine_m
                        dist_nm = haversine_m(
                            telemetry.lat_deg, telemetry.lon_deg,
                            ctx["destination"]["lat"], ctx["destination"]["lon"],
                        ) / 1852.0
                        ground_msl_ft = telemetry.altitude_ft  # on the ground, alt ≈ ground MSL
                        cruise_alt_agl = max(1500.0, min(5000.0, 55.0 * dist_nm))
                        initial_target = ground_msl_ft + cruise_alt_agl
                        ctx["targets"]["target_alt_ft"] = initial_target
                        ctx["takeoff_alt_ft"] = telemetry.altitude_ft
                        # Store home position so reset can teleport back to runway.
                        alt_m = telemetry.altitude_ft / 3.28084
                        if hasattr(self.adapter, "set_home"):
                            self.adapter.set_home(telemetry.lat_deg, telemetry.lon_deg, alt_m, telemetry.heading_deg)
                        print(f"NAV: dist={dist_nm:.1f}nm → cruise alt {cruise_alt_agl:.0f}ft AGL ({initial_target:.0f}ft MSL) [fixed]")

                        # ── Build flight plan + scorer + flight record ──
                        dest = ctx["destination"]
                        self._build_and_broadcast_plan(telemetry, ctx, dest, ground_msl_ft, initial_target, dist_nm)
            stale = (not telemetry.is_valid()) or is_telemetry_stale(telemetry, self.telemetry_timeout_s)

            now = time.time()
            # Grace window after a reset: ignore stale blips while X-Plane is settling.
            grace_until = getattr(self, "_reset_grace_until", 0.0)
            in_grace = now < grace_until

            # Phase-limit watchdog: catch unacceptable altitude/heading conditions.
            if not stale and not in_grace:
                failure = self._check_phase_limits(self.mode_manager.name, telemetry)
                if failure:
                    print(f"MONITOR: reset → {failure}")
                    self._monitor.clear()
                    stale = True  # reuse the existing stale-reset path

            if stale or in_grace:
                # Don't drive modes/controllers on stale data. Neutralize.
                act = abort_actuators()
                targets = None

                # Try a reset (option 1) when stale, but rate-limit it.
                if stale and (now - self._reset_last_ts >= self._reset_cooldown_s):
                    self._reset_last_ts = now
                    ok = False
                    try:
                        ok = bool(self.adapter.reset_flight())
                    except Exception:
                        ok = False
                    if ok:
                        # After reset, return state machine to ground so it can re-enter sequence.
                        try:
                            self.mode_manager.reset()
                        except Exception:
                            pass
                        # 3s grace window to allow telemetry to resume
                        self._reset_grace_until = time.time() + 3.0
                        print("RESET: sent reset flight command; waiting for telemetry...")
            else:
                desired = self.mode_manager.step(telemetry, stale=False)
                targets = self.guidance.compute(telemetry, desired)
                dt = max(loop_start - self._last_step, 1e-3)
                act = self.controller.compute(telemetry, targets, dt)

                phase = self.mode_manager.name

                # ── Flight observer: feed every tick ──
                if self._observer is not None:
                    self._observer.observe(phase, telemetry, act)

                # Update scorer every tick during flight
                if self._scorer is not None:
                    self._scorer.update(phase, telemetry.altitude_ft)

                # Update accuracy tracker every tick
                if targets is not None:
                    self._accuracy.update(phase, telemetry, targets, act.throttle)

                # Broadcast phase changes + log to DB
                if phase != self._prev_phase:
                    broadcast.publish_status({
                        "event": "phase_change",
                        "from": self._prev_phase,
                        "to": phase,
                        "alt_ft": telemetry.altitude_ft,
                        "speed_kts": telemetry.airspeed_kts,
                    })
                    # Log phase change event to database
                    if self._flight_id:
                        try:
                            agl_ft = (telemetry.agl_m * 3.28084) if not __import__("math").isnan(telemetry.agl_m) else 0.0
                            event_type = "phase_change"
                            message = f"{self._prev_phase} → {phase}"
                            # Special event types for key transitions
                            if phase == "CLIMB" and self._prev_phase == "GROUND":
                                event_type = "liftoff"
                                message = f"Liftoff at {telemetry.airspeed_kts:.0f} kts"
                            elif phase == "LAND":
                                event_type = "touchdown"
                                message = f"Touchdown at {telemetry.airspeed_kts:.0f} kts"
                            local_db.log_event(
                                self._flight_id, event_type,
                                message=message,
                                altitude_ft=telemetry.altitude_ft,
                                agl_ft=agl_ft,
                                airspeed_kts=telemetry.airspeed_kts,
                                heading_deg=telemetry.heading_deg,
                                lat=telemetry.lat_deg if telemetry.has_position() else None,
                                lon=telemetry.lon_deg if telemetry.has_position() else None,
                                phase=phase,
                            )
                        except Exception:
                            pass

                # ── Save telemetry snapshot every ~2s for flight path ──
                if self._flight_id and time.time() >= self._snapshot_next:
                    self._snapshot_next = time.time() + 2.0
                    self._snapshot_tick += 1
                    try:
                        import math as _m
                        agl = (telemetry.agl_m * 3.28084) if not _m.isnan(telemetry.agl_m) else 0.0
                        local_db.save_telemetry_snapshot(self._flight_id, {
                            "tick_num": self._snapshot_tick,
                            "lat": telemetry.lat_deg if telemetry.has_position() else None,
                            "lon": telemetry.lon_deg if telemetry.has_position() else None,
                            "altitude_ft": telemetry.altitude_ft,
                            "agl_ft": agl,
                            "heading_deg": telemetry.heading_deg,
                            "airspeed_kts": telemetry.airspeed_kts,
                            "vertical_speed_fpm": getattr(telemetry, "vs_fpm", None),
                            "pitch_deg": telemetry.pitch_deg,
                            "roll_deg": telemetry.roll_deg,
                            "throttle": act.throttle,
                            "pitch_cmd": act.pitch,
                            "roll_cmd": act.roll,
                            "yaw_cmd": act.yaw,
                            "brake_ratio": act.brake_ratio,
                            "phase": phase,
                        })
                    except Exception:
                        pass

                # Finalize score + observer on first LAND tick with wheels on ground
                if (phase == "LAND"
                        and self._prev_phase != "LAND"
                        and not self._landed
                        and self._scorer is not None
                        and telemetry.has_position()):
                    self._landed = True
                    score = self._scorer.finalize(
                        landing_lat=telemetry.lat_deg,
                        landing_lon=telemetry.lon_deg,
                        landing_speed_kts=telemetry.airspeed_kts,
                    )
                    self._scorer.print_summary(score)
                    self._scorer.save(score)

                    # ── Finalize flight record in database ──
                    if self._flight_id:
                        try:
                            # Build phase_timeline string from observer
                            phase_tl = ""
                            if self._observer:
                                compiled = None
                                try:
                                    compiled = self._observer.compile()
                                    pt = compiled.get("phase_timeline", {})
                                    phase_tl = ",".join(f"{p}:{int(d)}" for p, d in pt.items() if d > 0)
                                except Exception:
                                    pass

                            local_db.finalize_flight(
                                self._flight_id,
                                status="completed",
                                duration_s=score.duration_s,
                                score_total=score.total,
                                score_accuracy=score.pts_accuracy,
                                score_speed=score.pts_speed,
                                score_time=score.pts_time,
                                score_stability=score.pts_stability,
                                touchdown_speed_kts=score.landing_speed_kts,
                                touchdown_distance_ft=score.landing_dist_m * 3.28084 if score.landing_dist_m else None,
                                cruise_alt_stddev_ft=score.cruise_alt_std_ft,
                                cruise_speed_avg_kts=getattr(score, "cruise_speed_avg_kts", None),
                                phase_timeline=phase_tl if phase_tl else None,
                                is_personal_best=1 if score.is_personal_best else 0,
                            )
                            print(f"[PEREGRINE] Flight {self._flight_id[:8]}... finalized → score {score.total:.0f}")

                            # Log final stop event
                            local_db.log_event(
                                self._flight_id, "stopped",
                                message=f"Flight complete. Score: {score.total:.0f}/100",
                                altitude_ft=telemetry.altitude_ft,
                                airspeed_kts=telemetry.airspeed_kts,
                                heading_deg=telemetry.heading_deg,
                                lat=telemetry.lat_deg if telemetry.has_position() else None,
                                lon=telemetry.lon_deg if telemetry.has_position() else None,
                                phase="LAND",
                            )
                        except Exception as e:
                            print(f"[PEREGRINE] Flight finalization failed: {e}")

                    # ── Finalize observer: compile + write to Supabase ──
                    if self._observer is not None and not self._observer_finalized:
                        self._observer_finalized = True
                        try:
                            print(self._observer.summary())
                            learned = self._observer.compile()
                            from uav.learning.envelope_updater import update_envelope
                            updated = update_envelope(self._icao_type, learned)
                            flights = updated.get("total_flights", "?")
                            print(f"[PEREGRINE] Envelope updated → flight #{flights}")
                        except Exception as e:
                            print(f"[PEREGRINE] Envelope update failed: {e}")

                self._prev_phase = phase

                # ── Broadcast telemetry to app (~5Hz) ──
                self._broadcast_telemetry(telemetry, targets, act, phase)

            # ── Poll DB for end-flight request (reliable fallback) ─────
            if self._aircraft_id and not self._end_flight_requested:
                if not hasattr(self, '_end_poll_next'):
                    self._end_poll_next = time.time() + 3.0
                if time.time() >= self._end_poll_next:
                    self._end_poll_next = time.time() + 3.0
                    try:
                        if broadcast.poll_end_flight(self._aircraft_id):
                            print("[COMMAND] End flight from DB poll")
                            self._end_flight_requested = True
                    except Exception:
                        pass

            # ── End-flight request (from app) ──────────────────────────
            if self._end_flight_requested:
                self._end_flight_requested = False
                self._finalize_active_flight(telemetry, status="aborted", reason="User ended flight")
                # Reset aircraft back to runway
                try:
                    self.adapter.reset_flight()
                    self.mode_manager.reset()
                    self._reset_grace_until = time.time() + 3.0
                except Exception as e:
                    print(f"[END FLIGHT] Reset failed: {e}")
                # Go back to waiting for fly command
                self._fly_command_received = False
                self._fly_destination = None
                self._runway_detection = None
                self._preflight_broadcasted = False
                self._prev_phase = "GROUND"
                self._landed = False
                self._observer_finalized = False
                broadcast.publish_status({"event": "flight_ended", "reason": "user_ended"})
                print("[END FLIGHT] Flight ended — back to preflight")

            # Safety envelope: descent rate, stall, overspeed, bank limits
            if targets is not None and self._safety_envelope:
                targets, act, _corr = self._safety_envelope(
                    telemetry, targets, act,
                    v_stall=self._v_stall, v_never_exceed=self._v_ne,
                )

            act = self.safety.clamp(act)
            self._safe_write(act)

            # If we were in a reset grace window and telemetry is good again, re-arm modes.
            grace_until = getattr(self, "_reset_grace_until", 0.0)
            if grace_until and time.time() >= grace_until and telemetry.is_valid():
                # clear grace
                self._reset_grace_until = 0.0
                # ensure state machine can auto-start again
                try:
                    self.mode_manager.reset()
                except Exception:
                    pass
                print("RESET: telemetry stable; re-armed.")

            self.recorder.record(
                mode=self.mode_manager.name,
                telemetry=telemetry,
                targets=targets,
                actuators=act,
            )

            if time.time() >= status_next:
                status_next = time.time() + 1.0
                age = time.time() - telemetry.timestamp if telemetry.timestamp else 999.0
                acc_str = f"Acc={self._accuracy.instant_accuracy:.0f}%/{self._accuracy.flight_accuracy:.0f}%"
                print(
                    f"Mode={self.mode_manager.name} "
                    f"Alt={telemetry.altitude_ft:.1f}ft "
                    f"AGL={telemetry.agl_m:.1f}m "
                    f"Hdg={telemetry.heading_deg:.1f}deg "
                    f"Spd={telemetry.airspeed_kts:.1f}kts "
                    f"{acc_str} "
                    f"age={age:.2f}s "
                    f"cmd[T={act.throttle:.2f} P={act.pitch:+.2f} R={act.roll:+.2f} Y={act.yaw:+.2f} B={act.brake_ratio:.2f} G={'dn' if act.gear_down else 'UP'}]"
                )

            # ── Heartbeat: update aircraft row in Supabase every 5s ──
            if self._aircraft_id and time.time() >= self._heartbeat_next:
                self._heartbeat_next = time.time() + 5.0
                rwy = self._runway_detection
                phase = self.mode_manager.name if not stale else "offline"
                ctx = getattr(self.mode_manager, "ctx", None) or {}
                dest = ctx.get("destination")
                has_pos = telemetry.has_position()

                hb = {
                    "status": phase.lower(),
                    # Position
                    "last_lat": telemetry.lat_deg if has_pos else 0.0,
                    "last_lon": telemetry.lon_deg if has_pos else 0.0,
                    "last_heading": round(telemetry.heading_deg, 1),
                    # Altitude
                    "altitude_ft": round(telemetry.altitude_ft, 0),
                    "agl_ft": round(telemetry.agl_m * 3.281, 0) if telemetry.agl_m else 0,
                    "vs_fpm": round(telemetry.vs_fpm, 0) if hasattr(telemetry, "vs_fpm") else 0,
                    # Speed
                    "airspeed_kts": round(telemetry.airspeed_kts, 1),
                    "groundspeed_kts": round(telemetry.groundspeed_kts, 1) if hasattr(telemetry, "groundspeed_kts") else 0,
                    # Attitude
                    "pitch_deg": round(telemetry.pitch_deg, 1),
                    "roll_deg": round(telemetry.roll_deg, 1),
                    # Controls
                    "throttle_pct": round(act.throttle * 100, 0),
                }

                # Targets: prefer live guidance output, fall back to ctx targets
                if targets:
                    hb["target_alt_ft"] = round(targets.altitude_ft, 0) if hasattr(targets, "altitude_ft") else None
                    hb["target_speed_kts"] = round(targets.airspeed_kts, 1) if hasattr(targets, "airspeed_kts") else None
                    hb["target_heading_deg"] = round(targets.heading_deg, 1) if hasattr(targets, "heading_deg") else None
                else:
                    # During stale/reset phases, targets is None — read from ctx instead
                    ctx_targets = ctx.get("targets", {})
                    if ctx_targets.get("target_alt_ft") is not None:
                        hb["target_alt_ft"] = round(float(ctx_targets["target_alt_ft"]), 0)
                    if ctx_targets.get("target_airspeed_kts") is not None:
                        hb["target_speed_kts"] = round(float(ctx_targets["target_airspeed_kts"]), 1)
                    # Target heading: use ctx target_hdg_deg, or dest bearing as fallback
                    if ctx_targets.get("target_hdg_deg") is not None:
                        hb["target_heading_deg"] = round(float(ctx_targets["target_hdg_deg"]), 1)
                    elif dest and has_pos:
                        try:
                            from uav.nav.geo import bearing_deg as _bd
                            hb["target_heading_deg"] = round(_bd(
                                telemetry.lat_deg, telemetry.lon_deg,
                                float(dest["lat"]), float(dest["lon"])), 1)
                        except Exception:
                            pass

                # Accuracy — only include if columns exist in DB
                # TODO: add columns via Supabase SQL editor then uncomment:
                # hb["accuracy_pct"] = round(self._accuracy.instant_accuracy, 1)
                # hb["flight_accuracy_pct"] = round(self._accuracy.flight_accuracy, 1)

                # Runway
                if rwy and rwy.detected:
                    hb["runway_icao"] = rwy.airport_icao
                    hb["runway_designator"] = rwy.runway_designator

                # Nav / destination
                if dest and has_pos:
                    try:
                        from uav.nav.geo import haversine_m, bearing_deg
                        dist_m = haversine_m(telemetry.lat_deg, telemetry.lon_deg,
                                             float(dest["lat"]), float(dest["lon"]))
                        hb["dest_icao"] = dest.get("icao", "")
                        hb["dest_dist_nm"] = round(dist_m / 1852.0, 2)
                        hb["dest_bearing_deg"] = round(
                            bearing_deg(telemetry.lat_deg, telemetry.lon_deg,
                                        float(dest["lat"]), float(dest["lon"])), 1)
                        gs = telemetry.groundspeed_kts if hasattr(telemetry, "groundspeed_kts") and telemetry.groundspeed_kts > 10 else telemetry.airspeed_kts
                        hb["eta_seconds"] = round((dist_m / 1852.0) / gs * 3600) if gs > 10 else 0
                    except Exception:
                        pass

                broadcast.publish_heartbeat(self._aircraft_id, hb)

            self._last_step = loop_start
            sleep_s = max(0.0, (1.0 / self.loop_rate_hz) - (time.time() - loop_start))
            time.sleep(sleep_s)

            if end_at and time.time() >= end_at:
                self._running = False
