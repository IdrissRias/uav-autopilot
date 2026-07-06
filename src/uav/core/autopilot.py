from __future__ import annotations

import time

from uav.core.control.simple_fixedwing import SimpleFixedWingController
from uav.core.guidance.base import Guidance
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
        # Cached ribbon waypoints + periodic re-broadcast timer. The
        # ribbon is published once at flight start via `flight_plan`
        # broadcast. If the app's subscription wasn't ready at that
        # moment (wedged channel, just-launched app, network blip),
        # the ribbon never arrives and the live map shows the plane
        # without its planned line. Re-broadcasting every 10 s gives
        # late-arriving subscribers a chance to catch up. Keyed by
        # flight_id so a stale ribbon doesn't bleed into a new flight.
        self._ribbon_waypoints_cache: list | None = None
        self._ribbon_rebroadcast_next: float = 0.0
        self._aircraft_id: str | None = aircraft_id
        # Flight DB record
        self._flight_id: str | None = None
        self._snapshot_next: float = 0.0  # next telemetry snapshot time
        self._snapshot_tick: int = 0      # tick counter for snapshots
        self._end_flight_requested = False  # set by app "end_flight" command

        # ── Crash detector ──
        # Two signals, either triggers a "crashed" finalize:
        #   1. AGL < -2 m for _crash_underground_ticks ticks in a row
        #      (plane is underground — X-Plane sometimes does this on hard impacts).
        #   2. Phase stuck at ABORT for _crash_abort_seconds (telemetry has
        #      been stale that long — the sim likely crashed or froze).
        # Tunables kept conservative so we don't false-trip on brief transients.
        self._underground_ticks = 0
        self._abort_start_ts: float | None = None
        self._crash_underground_ticks = 20   # ~1s at 20Hz
        self._crash_abort_seconds = 15.0

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

        elif action == "preview_ribbon":
            # Build a ribbon preview without starting the flight.
            # Broadcasts waypoints back to the app for map display.
            dest = data.get("destination")
            if dest and self.mode_manager and hasattr(self.mode_manager, 'ctx'):
                print(f"[COMMAND] PREVIEW RIBBON for {dest.get('icao', '?')}")
                self._preview_ribbon(dest)

        elif action == "set_cruise_throttle":
            value = data.get("value")
            if value is not None:
                airframe = self.mode_manager.ctx.get("airframe", {})
                airframe["throttle"]["cruise"] = float(value)
                print(f"[COMMAND] Cruise throttle → {value}")

    def _preview_ribbon(self, dest: dict) -> None:
        """Build a ribbon preview and broadcast waypoints without starting a flight."""
        try:
            from uav.nav.flight_plan_v2 import plan_path, format_ribbon
            from uav.nav.geo import haversine_m
            from uav.comms import broadcast

            ctx = self.mode_manager.ctx
            airframe = ctx.get("airframe", {})
            speeds = airframe.get("speeds_kts", {})
            rates = airframe.get("rates_fpm", {})

            # Use last known telemetry for departure position
            telemetry = self._last_telemetry if hasattr(self, '_last_telemetry') else None
            if telemetry is None or not telemetry.is_valid():
                print("[PREVIEW] No valid telemetry — can't build ribbon")
                return

            dep_lat = telemetry.lat_deg
            dep_lon = telemetry.lon_deg
            dep_alt = telemetry.altitude_ft
            dep_hdg = telemetry.heading_deg

            dest_lat = float(dest.get("lat", 0))
            dest_lon = float(dest.get("lon", 0))

            # Compute cruise altitude
            dist_nm = haversine_m(dep_lat, dep_lon, dest_lat, dest_lon) / 1852.0
            cruise_alt_agl = max(1500.0, min(5000.0, 55.0 * dist_nm))
            cruise_alt = dep_alt + cruise_alt_agl

            # Look up destination runway. User-drawn runways are packed
            # into dest["runway"] by broadcast.poll_fly_command; for those
            # we skip the SQL lookup entirely. Otherwise (published-airport
            # destination) we fall back to the local SQLite runways table.
            dest_rwy_heading = None
            dest_thr_lat = None
            dest_thr_lon = None
            dest_alt_ft = dep_alt  # fallback

            packed_rwy = dest.get("runway")
            if packed_rwy:
                dest_rwy_heading = packed_rwy.get("heading")
                dest_thr_lat = packed_rwy.get("threshold_lat")
                dest_thr_lon = packed_rwy.get("threshold_lon")
                if packed_rwy.get("elevation_ft") is not None:
                    dest_alt_ft = float(packed_rwy["elevation_ft"])
            else:
                try:
                    import sqlite3, os
                    db = sqlite3.connect(os.path.expanduser("~/.peregrine/peregrine.db"))
                    db.row_factory = sqlite3.Row
                    rwys = db.execute("""
                        SELECT r.*, a.elevation_ft as airport_elevation_ft FROM runways r
                        JOIN airports a ON r.airport_id = a.id
                        WHERE a.icao_code = ?
                        ORDER BY r.length_ft DESC
                    """, (dest.get("icao", ""),)).fetchall()
                    if rwys:
                        from uav.nav.geo import bearing_deg as _bd
                        approach_brg = _bd(dep_lat, dep_lon, dest_lat, dest_lon)
                        best_rwy = None
                        best_diff = 999.0
                        for rwy in rwys:
                            hdg1 = float(rwy["heading_deg"])
                            hdg2 = (hdg1 + 180.0) % 360.0
                            for hdg, tlat, tlon in [
                                (hdg1, rwy["threshold_lat"], rwy["threshold_lon"]),
                                (hdg2, rwy["end_lat"], rwy["end_lon"]),
                            ]:
                                diff = abs(((approach_brg - hdg + 180) % 360) - 180)
                                if diff < best_diff:
                                    best_diff = diff
                                    best_rwy = (hdg, float(tlat), float(tlon))
                                    dest_alt_ft = float(rwy["airport_elevation_ft"]) if rwy["airport_elevation_ft"] else dep_alt
                        if best_rwy:
                            dest_rwy_heading, dest_thr_lat, dest_thr_lon = best_rwy
                    db.close()
                except Exception as e:
                    print(f"[PREVIEW] Runway lookup failed: {e}")

            # Prefer v_land (landing reference) over v_stall; default 77.
            v_stall = float(speeds.get("v_land", 77.0))

            ribbon = plan_path(
                dep_lat=dep_lat, dep_lon=dep_lon,
                dep_alt_ft=dep_alt, dep_heading=dep_hdg,
                dest_lat=dest_lat, dest_lon=dest_lon,
                dest_alt_ft=dest_alt_ft,
                dest_rwy_heading=dest_rwy_heading,
                dest_threshold_lat=dest_thr_lat,
                dest_threshold_lon=dest_thr_lon,
                cruise_alt_ft=cruise_alt,
                v_stall=v_stall,
                v_cruise_kts=(float(speeds["v_cruise"])
                              if speeds.get("v_cruise") else None),
            )
            print(format_ribbon(ribbon))

            # Broadcast waypoints to the app
            # Sample every Nth point to keep payload small
            step = max(1, len(ribbon.points) // 60)
            waypoints = []
            for i in range(0, len(ribbon.points), step):
                p = ribbon.points[i]
                waypoints.append({
                    "name": p.phase,
                    "lat": round(p.lat, 6),
                    "lon": round(p.lon, 6),
                    "alt_ft": round(p.alt_ft, 0),
                    "speed_kts": round(p.speed_kts, 1),
                    "phase": p.phase,
                    "heading": round(p.heading_deg, 1),
                })
            # Always include last point
            p = ribbon.points[-1]
            waypoints.append({
                "name": "END",
                "lat": round(p.lat, 6),
                "lon": round(p.lon, 6),
                "alt_ft": round(p.alt_ft, 0),
                "speed_kts": round(p.speed_kts, 1),
                "phase": p.phase,
                "heading": round(p.heading_deg, 1),
            })

            broadcast.publish_status({
                "event": "flight_plan",
                "waypoints": waypoints,
            })
            print(f"[PREVIEW] Broadcast {len(waypoints)} waypoints to app")

        except Exception as e:
            print(f"[PREVIEW] Failed: {e}")
            import traceback; traceback.print_exc()

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

            # Partial scores if scorer is available.
            # Every flight must get a score — completed, aborted, crashed.
            # finalize_partial handles the "never landed" case by scoring
            # on whatever dimensions have data (distance-to-dest, cruise
            # stability so far, duration). pts_accuracy will be 0 when
            # the flight ends far from destination, which is honest.
            duration_s = None
            score = None
            if self._scorer and self._scorer._start_ts:
                duration_s = time.time() - self._scorer._start_ts
                if telemetry.has_position():
                    try:
                        score = self._scorer.finalize_partial(
                            current_lat=telemetry.lat_deg,
                            current_lon=telemetry.lon_deg,
                            current_speed_kts=telemetry.airspeed_kts,
                            outcome=status,
                        )
                        self._scorer.print_summary(score)
                        self._scorer.save(score)
                    except Exception as e:
                        print(f"[PEREGRINE] Partial score failed: {e}")

            # Build kwargs — include score_* only if we produced a score.
            finalize_kwargs = {
                "status": status,
                "duration_s": duration_s,
                "phase_timeline": phase_tl if phase_tl else None,
                "abort_reason": reason if reason else None,
            }
            if score is not None:
                finalize_kwargs.update({
                    "score_total": score.total,
                    "score_accuracy": score.pts_accuracy,
                    "score_speed": score.pts_speed,
                    "score_time": score.pts_time,
                    "score_stability": score.pts_stability,
                    "touchdown_speed_kts": score.landing_speed_kts,
                    "touchdown_distance_ft": (score.landing_dist_m * 3.28084
                                               if score.landing_dist_m else None),
                    "cruise_alt_stddev_ft": score.cruise_alt_std_ft,
                    "is_personal_best": 1 if score.is_personal_best else 0,
                })

            local_db.finalize_flight(self._flight_id, **finalize_kwargs)

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

        # ── Ribbon adherence (L1 track follower output) ──
        # cross_track_nm: how many nm the plane is OFF the ribbon line.
        #   Positive = plane is to the right of track.
        # along_track_nm: distance progressed along the full polyline.
        # segment_idx / ribbon_length_nm: which segment, total length.
        # This is the "religiously following the ribbon" meter. <0.3 nm
        # sustained = good. Anything larger and we are flying parallel
        # to the ribbon, not on it — same failure mode as today's bug.
        track_state = ctx.get("track_state")
        if track_state:
            data["ribbon"] = {
                "cross_track_nm": round(track_state.get("cross_track_nm", 0.0), 3),
                "along_track_nm": round(track_state.get("along_track_nm", 0.0), 2),
                "segment_idx": track_state.get("segment_idx", 0),
                "segment_progress": round(track_state.get("segment_progress", 0.0), 3),
                "ribbon_length_nm": round(track_state.get("ribbon_length_nm", 0.0), 2),
            }

        broadcast.publish_telemetry(data)

    def _build_and_broadcast_plan(self, telemetry, ctx, dest, ground_msl_ft, initial_target, dist_nm) -> None:
        """Build flight plan (V2 ribbon or V1 waypoints), create scorer, observer, flight record."""
        # ── Try V2 ribbon path first ──
        from uav.core.flight_engine import FlightEngine
        if isinstance(self.mode_manager, FlightEngine):
            try:
                from uav.nav.flight_plan_v2 import plan_path, format_ribbon
                # Prefer dest["runway"] (user-drawn runway, packed in
                # broadcast.poll_fly_command) before ctx["dest_runway"]
                # (published-airport SQL lookup). Either one provides
                # heading + threshold so the planner can anchor the
                # landing chain on the real runway axis.
                dest_rwy = dest.get("runway") or ctx.get("dest_runway")
                rwy = self._runway_detection

                # Prefer v_land (landing reference) over v_stall; default 77.
                v_stall = float(ctx["airframe"]["speeds_kts"].get("v_land", 77.0))
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
                    v_stall=v_stall,
                    v_cruise_kts=(float(ctx["airframe"]["speeds_kts"]["v_cruise"])
                                  if ctx["airframe"]["speeds_kts"].get("v_cruise")
                                  else None),
                )
                # Pre-build ribbon so FlightEngine doesn't rebuild it
                self.mode_manager._ribbon = ribbon
                self.mode_manager._built = True
                self.mode_manager._idx = 0
                print(format_ribbon(ribbon))

                # Broadcast ribbon waypoints to the app for map display
                # Sample every ~10 points to keep the broadcast small
                step = max(1, len(ribbon.points) // 50)
                waypoints = [
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
                ]
                try:
                    broadcast.publish_status({
                        "event": "flight_plan",
                        "waypoints": waypoints,
                    })
                except Exception:
                    pass
                # Cache for periodic re-broadcast — see __init__ doc.
                self._ribbon_waypoints_cache = waypoints
                self._ribbon_rebroadcast_next = time.time() + 10.0
            except Exception as e:
                print(f"[RIBBON] Flight plan generation failed: {e}")
                import traceback; traceback.print_exc()
        else:
            # ── V1 flight plan (legacy) ──
            try:
                from uav.nav.flight_plan import build_flight_plan, format_plan
                # Same precedence as the V2 path: user-drawn runway
                # (packed by broadcast) wins over the published-airport
                # SQL lookup.
                dest_rwy = dest.get("runway") or ctx.get("dest_runway") or {}
                rwy = self._runway_detection
                plan = build_flight_plan(
                    dep_lat=telemetry.lat_deg,
                    dep_lon=telemetry.lon_deg,
                    dep_alt_ft=telemetry.altitude_ft,
                    dep_heading=rwy.runway_heading_deg if rwy and rwy.detected else telemetry.heading_deg,
                    dest_lat=float(dest["lat"]),
                    dest_lon=float(dest["lon"]),
                    dest_alt_ft=float(dest_rwy.get("elevation_ft", ground_msl_ft) or ground_msl_ft),
                    dest_rwy_heading=dest_rwy.get("heading"),
                    dest_threshold_lat=dest_rwy.get("threshold_lat"),
                    dest_threshold_lon=dest_rwy.get("threshold_lon"),
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

            # Store ribbon waypoints in the flight record for app reload
            try:
                from uav.core.flight_engine import FlightEngine
                if isinstance(self.mode_manager, FlightEngine) and self.mode_manager._ribbon:
                    import json
                    ribbon = self.mode_manager._ribbon
                    step = max(1, len(ribbon.points) // 60)
                    wp_json = json.dumps([
                        {"lat": round(p.lat, 6), "lon": round(p.lon, 6),
                         "alt_ft": round(p.alt_ft, 0), "speed_kts": round(p.speed_kts, 1),
                         "phase": p.phase, "heading": round(p.heading_deg, 1)}
                        for i, p in enumerate(ribbon.points) if i % step == 0
                    ])
                    # Use raw SQL since finalize_flight adds ended_at
                    conn = local_db.get_connection()
                    conn.execute("UPDATE flights SET ribbon_waypoints = ? WHERE id = ?",
                                 (wp_json, self._flight_id))
                    conn.commit()
                    print(f"[PEREGRINE] Ribbon stored in flight record ({len(ribbon.points)} pts → {len(wp_json)} bytes)")
            except Exception as e:
                print(f"[PEREGRINE] Ribbon storage failed: {e}")

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
            self._last_telemetry = telemetry

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

                # Heartbeat so the app knows we're alive.
                # Do NOT include `status` here — the app writes
                # `status="fly_requested"` on FLY click, and a 5-sec
                # heartbeat with `status="preflight"` would race-overwrite
                # the click before the poll picks it up.
                if self._aircraft_id and time.time() >= self._heartbeat_next:
                    self._heartbeat_next = time.time() + 5.0
                    hb = {
                        "last_lat": telemetry.lat_deg if telemetry.has_position() else 0.0,
                        "last_lon": telemetry.lon_deg if telemetry.has_position() else 0.0,
                        "last_heading": round(telemetry.heading_deg, 1),
                        "altitude_ft": round(telemetry.altitude_ft, 0),
                        "airspeed_kts": 0,
                    }
                    broadcast.publish_heartbeat(self._aircraft_id, hb)

                # Accept fly/preview command from broadcast OR DB poll
                if not self._fly_destination and self._aircraft_id:
                    if not hasattr(self, '_poll_next'):
                        self._poll_next = time.time() + 2.0
                    if time.time() >= self._poll_next:
                        self._poll_next = time.time() + 3.0
                        try:
                            row = broadcast.poll_fly_command(self._aircraft_id)
                            if row:
                                action = row.pop("_action", "fly")
                                if action == "preview":
                                    # Preview only — build ribbon, broadcast, don't fly
                                    print(f"[COMMAND] Preview ribbon from DB: {row}")
                                    self._preview_ribbon(row)
                                else:
                                    self._fly_destination = row
                                    print(f"[COMMAND] Fly command from DB: {row}")
                        except Exception as e:
                            print(f"[POLL] Error: {e}")

                if self._fly_destination:
                    ctx = getattr(self.mode_manager, "ctx", None)
                    if ctx is not None and ctx.get("destination") is None:
                        dest = self._fly_destination

                        # ── Reject same-spot flights ─────────────────────
                        # The app can command the plane back to where it
                        # already sits. Autopilot would happily spin a 180
                        # and fly to itself. Gate on 0.1 nm (~608 ft / 185 m) —
                        # permissive enough that legitimate short hops still
                        # fly, strict enough to block the "fly to where I
                        # already am" bug. Only enforced when we have a
                        # position fix; a missing fix is not a reason to
                        # refuse, the regular takeoff path catches that.
                        if telemetry.has_position():
                            from uav.nav.geo import haversine_m as _hv_gate
                            _dist_nm_gate = _hv_gate(
                                telemetry.lat_deg, telemetry.lon_deg,
                                float(dest["lat"]), float(dest["lon"]),
                            ) / 1852.0
                            if _dist_nm_gate < 0.1:
                                print(f"[COMMAND] Rejecting fly to "
                                      f"{dest.get('icao','???')}: already "
                                      f"there ({_dist_nm_gate:.3f} nm < 0.1 nm min)")
                                broadcast.publish_status({
                                    "event": "fly_rejected",
                                    "reason": "too_close",
                                    "icao": dest.get("icao", ""),
                                    "distance_nm": round(_dist_nm_gate, 3),
                                    "min_distance_nm": 0.1,
                                    "message": (
                                        f"Too close to "
                                        f"{dest.get('icao', 'destination')} "
                                        f"— must be more than 0.1 nm away to fly."
                                    ),
                                })
                                # Drop the request so the next loop iteration
                                # doesn't re-try the same command forever.
                                self._fly_destination = None
                                self._last_step = loop_start
                                sleep_s = max(
                                    0.0,
                                    (1.0 / self.loop_rate_hz) - (time.time() - loop_start),
                                )
                                time.sleep(sleep_s)
                                continue

                        ctx["destination"] = {
                            "icao": dest.get("icao", "???"),
                            "name": dest.get("name", ""),
                            "lat": float(dest["lat"]),
                            "lon": float(dest["lon"]),
                        }
                        print(f"[COMMAND] Flying to {ctx['destination']['icao']}")

                        # If broadcast already packed runway geometry
                        # (user-drawn runway path), use it directly and
                        # skip the SQL airport lookup entirely.
                        if dest.get("runway"):
                            ctx["dest_runway"] = dest["runway"]
                            print(
                                f"[NAV] Landing runway (user-drawn): "
                                f"{dest['runway'].get('designator', 'USR')} "
                                f"hdg {dest['runway']['heading']:.0f}° "
                                f"len {dest['runway'].get('length_ft', 0):.0f}ft"
                            )

                        # Look up the best runway at the destination for approach alignment.
                        # Skipped when broadcast already supplied user-drawn-runway geometry.
                        try:
                            if dest.get("runway"):
                                rwys = []
                            else:
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
                            # Only close the SQLite handle if we actually opened it
                            # (skipped when broadcast already supplied user runway).
                            if not dest.get("runway"):
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

                    # Still send heartbeat so app knows we're alive.
                    # `status` omitted — see other heartbeat site for
                    # the race-condition rationale.
                    if self._aircraft_id and time.time() >= self._heartbeat_next:
                        self._heartbeat_next = time.time() + 5.0
                        hb = {
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

            if stale or in_grace:
                # Don't drive modes/controllers on stale data. Neutralize.
                act = abort_actuators()
                targets = None

                # Only reset if we've EVER had valid telemetry (timestamp>0).
                # Cold-start with no RREF replies must NOT trigger a reset — that
                # spams sim/operation/reset_flight at X-Plane and knocks it over.
                had_telemetry = telemetry.timestamp > 0.0
                if stale and had_telemetry and (now - self._reset_last_ts >= self._reset_cooldown_s):
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
                elif stale and not had_telemetry:
                    # Throttled warning so we don't spam the log
                    if (now - self._reset_last_ts) >= 5.0:
                        self._reset_last_ts = now
                        print("[TELEMETRY] No RREF data from X-Plane yet — "
                              "check Settings → Network → Data Output is enabled "
                              "and 'Send to IP 127.0.0.1 port 49005' is set.")
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

                # ── Crash detection ──────────────────────────────────
                # Only arm the detector once we have an active flight
                # record AND we're past GROUND — a plane sitting on the
                # runway pre-takeoff shouldn't be scored as crashed.
                if self._flight_id and phase not in ("GROUND", "LAND"):
                    import math as _m
                    # 1) Underground check: AGL < -2m sustained.
                    agl = telemetry.agl_m
                    if not _m.isnan(agl) and agl < -2.0:
                        self._underground_ticks += 1
                    else:
                        self._underground_ticks = 0

                    # 2) Stuck-ABORT check: phase="ABORT" for too long.
                    if phase == "ABORT":
                        if self._abort_start_ts is None:
                            self._abort_start_ts = time.time()
                    else:
                        self._abort_start_ts = None

                    crashed = False
                    reason = ""
                    if self._underground_ticks >= self._crash_underground_ticks:
                        crashed = True
                        reason = f"Underground for {self._underground_ticks} ticks (AGL {agl:.1f}m)"
                    elif (self._abort_start_ts is not None
                          and time.time() - self._abort_start_ts >= self._crash_abort_seconds):
                        crashed = True
                        reason = f"Stuck in ABORT for {time.time() - self._abort_start_ts:.0f}s"

                    if crashed:
                        print(f"[CRASH] {reason} — finalizing flight as crashed")
                        # Reset counters so we don't double-fire.
                        self._underground_ticks = 0
                        self._abort_start_ts = None
                        self._finalize_active_flight(
                            telemetry, status="crashed", reason=reason,
                        )
                        broadcast.publish_status({
                            "event": "flight_crashed",
                            "reason": reason,
                        })
                        # Drop back to preflight state.
                        self._fly_command_received = False
                        self._fly_destination = None
                        self._runway_detection = None
                        self._preflight_broadcasted = False
                        self._prev_phase = "GROUND"
                        self._landed = False
                        self._observer_finalized = False

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

                # ── Periodic ribbon re-broadcast (every 10 s) ──
                # Survives wedged/late-joining app subscriptions. The
                # ribbon is otherwise sent ONCE at flight start, so a
                # missed broadcast = no ribbon on the map for the whole
                # flight. Re-broadcasting keeps the line fresh and lets
                # the app re-render if it lost the original.
                if (self._ribbon_waypoints_cache
                        and time.time() >= self._ribbon_rebroadcast_next):
                    self._ribbon_rebroadcast_next = time.time() + 10.0
                    try:
                        broadcast.publish_status({
                            "event": "flight_plan",
                            "waypoints": self._ribbon_waypoints_cache,
                        })
                    except Exception:
                        pass

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
                # Clear cached ribbon so the next flight's broadcast
                # isn't preceded by a re-broadcast of the old one.
                self._ribbon_waypoints_cache = None
                broadcast.publish_status({"event": "flight_ended", "reason": "user_ended"})
                print("[END FLIGHT] Flight ended — back to preflight")

            act = self.safety.clamp(act)
            if act.flap_ratio > 0.01 and not getattr(self, '_flap_dbg', False):
                print(f"[DEBUG-FLAP] flap_ratio={act.flap_ratio:.2f} phase={getattr(targets, 'flap_ratio', '?')}", flush=True)
                self._flap_dbg = True
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

            # Pull ribbon follower stats so flight CSV can be graded for
            # "religious following" offline (cross_track_nm over time).
            _track_state = None
            _ctx_for_rec = getattr(self.mode_manager, "ctx", None)
            if isinstance(_ctx_for_rec, dict):
                _track_state = _ctx_for_rec.get("track_state")

            self.recorder.record(
                mode=self.mode_manager.name,
                telemetry=telemetry,
                targets=targets,
                actuators=act,
                ribbon=_track_state,
            )

            if time.time() >= status_next:
                status_next = time.time() + 1.0
                age = time.time() - telemetry.timestamp if telemetry.timestamp else 999.0
                acc_str = f"Acc={self._accuracy.instant_accuracy:.0f}%/{self._accuracy.flight_accuracy:.0f}%"
                # Prefer the real keyframe name (e.g. CLIMB_GEAR_UP) over the
                # mapped scoring phase so the log shows what the ribbon is
                # actually doing.  Falls back to .name on older managers.
                mode_label = getattr(self.mode_manager, "keyframe_name",
                                     self.mode_manager.name)
                print(
                    f"Mode={mode_label} "
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
