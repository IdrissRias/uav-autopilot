from __future__ import annotations

import time

from uav.core.control.simple_fixedwing import SimpleFixedWingController
from uav.core.guidance.simple_guidance import SimpleGuidance
from uav.core.mode_manager import ModeManager
from uav.core.safety.limits import SafetyLimits, abort_actuators
from uav.core.safety.failsafe import is_telemetry_stale
from uav.logging.recorder import Recorder
from uav.sim.adapter_base import SimAdapter
from uav.sim.types import Actuators, Telemetry


class Autopilot:
    def __init__(
        self,
        adapter: SimAdapter,
        controller: SimpleFixedWingController,
        guidance: SimpleGuidance,
        mode_manager: ModeManager,
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

    def stop(self) -> None:
        self._running = False

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
                if ctx.get("destination") is None and hasattr(telemetry, "has_position") and telemetry.has_position():
                    nav = ctx.get("nav", {})
                    apt = nav.get("apt_dat_path")
                    if apt:
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
            stale = (not telemetry.is_valid()) or is_telemetry_stale(telemetry, self.telemetry_timeout_s)

            now = time.time()
            # Grace window after a reset: ignore stale blips while X-Plane is settling.
            grace_until = getattr(self, "_reset_grace_until", 0.0)
            in_grace = now < grace_until

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
