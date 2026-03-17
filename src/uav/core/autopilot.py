from __future__ import annotations

import time

from uav.core.control.simple_fixedwing import SimpleFixedWingController
from uav.core.guidance.simple_guidance import SimpleGuidance
from uav.core.safety.limits import SafetyLimits, abort_actuators
from uav.core.safety.failsafe import is_telemetry_stale
from uav.logging.recorder import Recorder
from uav.scoring.flight_scorer import FlightScorer
from uav.sim.adapter_base import SimAdapter
from uav.sim.types import Actuators, Telemetry


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
        self._landed = False        # True once we've scored the current flight
        self._prev_phase = "GROUND"

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

            # Hard upper limit: if the plane is way above target, it's out of control.
            if alt_error > 400.0:
                return f"CRUISE alt {telemetry.altitude_ft:.0f}ft is >400ft above target {target_alt:.0f}ft"

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
            ms.pop("circle_since", None)
        if mode not in ("CLIMB",):
            ms.pop("climb_high_since", None)

        return None

    def _safe_write(self, act: Actuators) -> None:
        self.adapter.write_actuators(act)

    def run(self, duration_s: float = 0.0) -> None:
        self._running = True
        status_next = time.time() + 1.0
        end_at = (time.time() + duration_s) if duration_s and duration_s > 0 else 0.0
        while self._running:
            loop_start = time.time()
            telemetry = self.adapter.read_telemetry()
            # Lazy destination selection once position is available.
            if getattr(self.mode_manager, "ctx", None) is not None:
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
                    # Adaptive cruise altitude: scale with distance so the full cycle always fits.
                    if ctx.get("destination"):
                        from uav.nav.geo import haversine_m
                        dist_nm = haversine_m(
                            telemetry.lat_deg, telemetry.lon_deg,
                            ctx["destination"]["lat"], ctx["destination"]["lon"],
                        ) / 1852.0
                        cruise_alt_agl = max(3500.0, min(25000.0, dist_nm * 200.0))
                        ctx["targets"]["target_alt_ft"] = telemetry.altitude_ft + cruise_alt_agl
                        ctx["takeoff_alt_ft"] = telemetry.altitude_ft  # used by APPROACH for final altitude
                        # Store home position so reset can teleport back to runway.
                        alt_m = telemetry.altitude_ft / 3.28084
                        if hasattr(self.adapter, "set_home"):
                            self.adapter.set_home(telemetry.lat_deg, telemetry.lon_deg, alt_m, telemetry.heading_deg)
                        print(f"NAV: dist={dist_nm:.1f}nm → cruise alt {cruise_alt_agl:.0f}ft AGL ({ctx['targets']['target_alt_ft']:.0f}ft MSL)")
                        # Start scorer for this flight
                        dest = ctx["destination"]
                        self._scorer = FlightScorer(
                            dest_icao=dest.get("icao", "???"),
                            dest_lat=float(dest["lat"]),
                            dest_lon=float(dest["lon"]),
                            cruise_target_ft=ctx["targets"]["target_alt_ft"],
                            log_dir="logs",
                        )
                        self._landed = False
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
                # Update scorer every tick during flight
                if self._scorer is not None:
                    self._scorer.update(phase, telemetry.altitude_ft)
                # Finalize score once on first LAND tick with wheels on ground
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
                self._prev_phase = phase

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
                print(
                    f"Mode={self.mode_manager.name} "
                    f"Alt={telemetry.altitude_ft:.1f}ft "
                    f"AGL={telemetry.agl_m:.1f}m "
                    f"Hdg={telemetry.heading_deg:.1f}deg "
                    f"Spd={telemetry.airspeed_kts:.1f}kts "
                    f"age={age:.2f}s "
                    f"cmd[T={act.throttle:.2f} P={act.pitch:+.2f} R={act.roll:+.2f} Y={act.yaw:+.2f} B={act.brake_ratio:.2f} G={'dn' if act.gear_down else 'UP'}]"
                )

            self._last_step = loop_start
            sleep_s = max(0.0, (1.0 / self.loop_rate_hz) - (time.time() - loop_start))
            time.sleep(sleep_s)

            if end_at and time.time() >= end_at:
                self._running = False
