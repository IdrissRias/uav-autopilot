"""Peregrine — Calibration Flight Director.

Runs automated calibration maneuvers to measure control sensitivity
for the current aircraft. Replaces the normal flight director during
calibration.

Maneuver sequence (~3 minutes total):
  Step 1: Pitch test — small elevator pulses, measure pitch rate response
  Step 2: Roll test — aileron inputs, measure roll rate response
  Step 3: Throttle test — throttle steps, measure acceleration response
  Step 4: Combined — gentle turns + climbs to cross-validate
  Step 5: Save calibration to envelope

The director keeps the aircraft safe: all maneuvers are gentle and
bounded. If anything goes wrong (stall, excessive bank, low altitude),
it immediately levels off and holds heading.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Optional

from uav.sim.types import Telemetry, Targets
from uav.learning.flight_observer import SensitivityTracker


def _wrap_deg(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


class CalibrationStep:
    """Definition of a single calibration maneuver."""

    def __init__(self, name: str, label: str, duration_s: float = 30.0):
        self.name = name
        self.label = label
        self.duration_s = duration_s
        self.started_at: float = 0.0
        self.completed = False

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.started_at else 0.0


class CalibrationDirector:
    """Automated calibration flight director.

    Interface matches ReactiveFlightDirector:
        .name   — current phase label
        .ctx    — shared context dict
        .step() — returns Targets
        .reset() — reset state
    """

    # Safety limits during calibration
    _MIN_AGL_FT = 500.0          # abort maneuver if below this
    _MAX_BANK_DEG = 20.0         # max bank during calibration
    _MAX_PITCH_DEG = 15.0        # max pitch during calibration
    _STALL_MARGIN_KTS = 15.0     # keep above stall + this margin

    # Altitude AGL at which we transition from takeoff to calibration
    _CALIBRATION_ALT_AGL_FT = 2000.0

    def __init__(
        self,
        ctx: dict,
        sensitivity_tracker: SensitivityTracker | None = None,
        on_progress: Callable[[int, int, str, float], None] | None = None,
        on_complete: Callable[[dict], None] | None = None,
    ) -> None:
        """
        Args:
            ctx: Shared autopilot context (airframe, targets, etc.)
            sensitivity_tracker: Tracker to feed measurements into.
                If None, creates a new one internally.
            on_progress: Callback(step, total, label, confidence) for broadcasting.
            on_complete: Callback(calibration_data) when calibration finishes.
        """
        self.ctx = ctx
        self._tracker = sensitivity_tracker or SensitivityTracker()
        self._on_progress = on_progress
        self._on_complete = on_complete

        self._phase = "CALIBRATING"
        self._hold_heading: float = 0.0
        self._hold_altitude: float = 0.0
        self._hold_speed: float = 0.0

        # Takeoff delegation — use the reactive director for takeoff + climb.
        # Enable demo_sequence so the reactive director doesn't require a destination.
        from uav.core.reactive_director import ReactiveFlightDirector
        if "mode" not in ctx:
            ctx["mode"] = {}
        ctx["mode"]["demo_sequence"] = True
        self._takeoff_director = ReactiveFlightDirector(ctx=ctx)
        self._takeoff_complete = False
        self._field_elevation_set = False

        # Define calibration steps (takeoff is handled separately before these)
        self._steps = [
            CalibrationStep("takeoff", "Taking off and climbing to safe altitude", 120.0),
            CalibrationStep("stabilize", "Stabilizing — straight and level", 10.0),
            CalibrationStep("pitch_test", "Pitch sensitivity test", 30.0),
            CalibrationStep("roll_test", "Roll sensitivity test", 30.0),
            CalibrationStep("throttle_test", "Throttle response test", 30.0),
            CalibrationStep("combined", "Combined maneuver validation", 40.0),
            CalibrationStep("save", "Saving calibration data", 2.0),
        ]
        self._current_step_idx = 0
        self._started = False
        self._completed = False

        # Maneuver timing
        self._maneuver_phase = 0     # sub-phase within a step
        self._maneuver_timer = 0.0   # timer for sub-maneuver timing

    # ------------------------------------------------------------------
    # Public interface (matches ReactiveFlightDirector)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._phase

    def reset(self) -> None:
        self._phase = "GROUND"
        self._current_step_idx = 0
        self._started = False
        self._completed = False

    def step(self, telemetry: Telemetry, stale: bool = False) -> Targets:
        """Run one calibration tick. Returns Targets for the controller."""
        if stale or self._completed:
            return self._hold_steady(telemetry)

        agl_ft = (telemetry.agl_m * 3.28084) if not math.isnan(telemetry.agl_m) else 0.0
        now = time.monotonic()

        # First tick: start the takeoff step
        if not self._started:
            self._started = True
            self._steps[0].started_at = now
            self._phase = "TAKEOFF"
            self._report_progress()
            print(f"[CALIBRATION] Taking off — will climb to "
                  f"{self._CALIBRATION_ALT_AGL_FT:.0f}ft AGL before starting maneuvers")

        current = self._steps[self._current_step_idx]

        # ── Step 0: Takeoff — delegate to ReactiveFlightDirector ──
        if current.name == "takeoff":
            if not self._takeoff_complete:
                # Set calibration target altitude based on field elevation
                if not self._field_elevation_set and agl_ft < 50.0:
                    field_elev = telemetry.altitude_ft
                    cal_target = field_elev + self._CALIBRATION_ALT_AGL_FT + 500.0
                    if "targets" in self.ctx:
                        self.ctx["targets"]["target_alt_ft"] = cal_target
                    self._field_elevation_set = True
                    print(f"[CALIBRATION] Field elevation ~{field_elev:.0f}ft MSL, "
                          f"climbing to {cal_target:.0f}ft MSL")

                # Let the reactive director handle takeoff + climb
                targets = self._takeoff_director.step(telemetry, stale)
                self._phase = f"TAKEOFF ({self._takeoff_director.name})"

                # Check if we've reached calibration altitude
                if agl_ft >= self._CALIBRATION_ALT_AGL_FT:
                    self._takeoff_complete = True
                    # Capture reference values at calibration altitude
                    self._hold_heading = telemetry.heading_deg
                    self._hold_altitude = telemetry.altitude_ft
                    v_cruise = float(self.ctx.get("airframe", {}).get("speeds_kts", {}).get("v_cruise", 200.0))
                    self._hold_speed = v_cruise
                    self._phase = "CALIBRATING"
                    print(f"[CALIBRATION] Reached {agl_ft:.0f}ft AGL — starting calibration maneuvers")
                    print(f"[CALIBRATION] Hold HDG={self._hold_heading:.0f} "
                          f"ALT={self._hold_altitude:.0f} SPD={self._hold_speed:.0f}")
                    # Advance to next step
                    current.completed = True
                    self._current_step_idx += 1
                    self._steps[self._current_step_idx].started_at = now
                    self._maneuver_phase = 0
                    self._maneuver_timer = now
                    self._report_progress()
                return targets

        # ── Safety check for calibration maneuvers (not during takeoff) ──
        airframe = self.ctx.get("airframe", {})
        v_stall = float(airframe.get("speeds_kts", {}).get("v_stall", 60.0))
        if agl_ft < self._MIN_AGL_FT or telemetry.airspeed_kts < v_stall + self._STALL_MARGIN_KTS:
            return self._hold_steady(telemetry)

        # Get current step
        if not current.started_at:
            current.started_at = now
            self._maneuver_phase = 0
            self._maneuver_timer = now
            self._report_progress()

        # Check if current step is done
        if current.elapsed >= current.duration_s:
            current.completed = True
            self._current_step_idx += 1
            if self._current_step_idx >= len(self._steps):
                return self._finish_calibration(telemetry)
            self._steps[self._current_step_idx].started_at = now
            self._maneuver_phase = 0
            self._maneuver_timer = now
            self._report_progress()
            print(f"[CALIBRATION] Step {self._current_step_idx + 1}/{len(self._steps)}: "
                  f"{self._steps[self._current_step_idx].label}")

        current = self._steps[self._current_step_idx]

        # Dispatch to step-specific logic
        if current.name == "stabilize":
            return self._hold_steady(telemetry)
        elif current.name == "pitch_test":
            return self._run_pitch_test(telemetry, current, now)
        elif current.name == "roll_test":
            return self._run_roll_test(telemetry, current, now)
        elif current.name == "throttle_test":
            return self._run_throttle_test(telemetry, current, now)
        elif current.name == "combined":
            return self._run_combined(telemetry, current, now)
        elif current.name == "save":
            return self._finish_calibration(telemetry)
        else:
            return self._hold_steady(telemetry)

    # ------------------------------------------------------------------
    # Calibration maneuvers
    # ------------------------------------------------------------------

    def _hold_steady(self, telemetry: Telemetry) -> Targets:
        """Hold heading, altitude, and speed — the baseline state."""
        return Targets(
            heading_deg=self._hold_heading or telemetry.heading_deg,
            altitude_ft=self._hold_altitude or telemetry.altitude_ft,
            airspeed_kts=self._hold_speed or telemetry.airspeed_kts,
            gear_down=False,
            flap_ratio=0.0,
        )

    def _run_pitch_test(self, telemetry: Telemetry, step: CalibrationStep, now: float) -> Targets:
        """Pitch test: alternate between nose-up and nose-down pulses.

        Pattern: 5s up → 5s level → 5s down → 5s level → repeat
        The controller applies these as pitch commands; the tracker
        measures the resulting pitch rate.
        """
        cycle_t = (now - step.started_at) % 20.0  # 20s cycle

        if cycle_t < 5.0:
            # Nose up: target 200ft above hold altitude
            target_alt = self._hold_altitude + 200.0
        elif cycle_t < 10.0:
            # Level
            target_alt = self._hold_altitude
        elif cycle_t < 15.0:
            # Nose down: target 200ft below hold altitude
            target_alt = self._hold_altitude - 200.0
        else:
            # Level
            target_alt = self._hold_altitude

        return Targets(
            heading_deg=self._hold_heading,
            altitude_ft=target_alt,
            airspeed_kts=self._hold_speed,
            gear_down=False,
            flap_ratio=0.0,
        )

    def _run_roll_test(self, telemetry: Telemetry, step: CalibrationStep, now: float) -> Targets:
        """Roll test: gentle heading changes to exercise ailerons.

        Pattern: turn 20° right → back to center → 20° left → back to center
        """
        cycle_t = (now - step.started_at) % 24.0  # 24s cycle

        if cycle_t < 6.0:
            # Turn 20° right
            target_hdg = self._hold_heading + 20.0
        elif cycle_t < 12.0:
            # Back to center
            target_hdg = self._hold_heading
        elif cycle_t < 18.0:
            # Turn 20° left
            target_hdg = self._hold_heading - 20.0
        else:
            # Back to center
            target_hdg = self._hold_heading

        # Normalize heading to [0, 360)
        target_hdg = target_hdg % 360.0

        return Targets(
            heading_deg=target_hdg,
            altitude_ft=self._hold_altitude,
            airspeed_kts=self._hold_speed,
            gear_down=False,
            flap_ratio=0.0,
        )

    def _run_throttle_test(self, telemetry: Telemetry, step: CalibrationStep, now: float) -> Targets:
        """Throttle test: step between low and high throttle.

        Pattern: cruise power → high power → cruise → low power → cruise
        We hold altitude constant so all speed changes come from throttle.
        """
        cycle_t = (now - step.started_at) % 24.0

        airframe = self.ctx.get("airframe", {})
        cruise_thr = float(airframe.get("throttle", {}).get("cruise", 0.65))

        if cycle_t < 6.0:
            throttle = cruise_thr
        elif cycle_t < 12.0:
            throttle = min(cruise_thr + 0.2, 0.95)
        elif cycle_t < 18.0:
            throttle = cruise_thr
        else:
            throttle = max(cruise_thr - 0.2, 0.15)

        return Targets(
            heading_deg=self._hold_heading,
            altitude_ft=self._hold_altitude,
            airspeed_kts=self._hold_speed,
            throttle=throttle,
            gear_down=False,
            flap_ratio=0.0,
        )

    def _run_combined(self, telemetry: Telemetry, step: CalibrationStep, now: float) -> Targets:
        """Combined test: gentle climbing turn to cross-validate all axes.

        Pattern: climbing right turn → level → descending left turn → level
        """
        cycle_t = (now - step.started_at) % 32.0

        if cycle_t < 8.0:
            # Climbing right turn
            target_hdg = self._hold_heading + 30.0
            target_alt = self._hold_altitude + 300.0
        elif cycle_t < 16.0:
            # Level, back to center
            target_hdg = self._hold_heading
            target_alt = self._hold_altitude
        elif cycle_t < 24.0:
            # Descending left turn
            target_hdg = self._hold_heading - 30.0
            target_alt = self._hold_altitude - 300.0
        else:
            # Level, back to center
            target_hdg = self._hold_heading
            target_alt = self._hold_altitude

        target_hdg = target_hdg % 360.0

        return Targets(
            heading_deg=target_hdg,
            altitude_ft=target_alt,
            airspeed_kts=self._hold_speed,
            gear_down=False,
            flap_ratio=0.0,
        )

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def _finish_calibration(self, telemetry: Telemetry) -> Targets:
        """Save calibration data and signal completion."""
        if not self._completed:
            self._completed = True
            self._phase = "CALIBRATION_COMPLETE"

            cal_data = self._tracker.compile()
            print(f"\n[CALIBRATION] ═══════════════════════════════════")
            print(f"[CALIBRATION]   CALIBRATION COMPLETE")
            print(f"[CALIBRATION]   Pitch sensitivity:    {cal_data['pitch_sensitivity']:.1f} deg/s per unit")
            print(f"[CALIBRATION]   Roll sensitivity:     {cal_data['roll_sensitivity']:.1f} deg/s per unit")
            print(f"[CALIBRATION]   Yaw sensitivity:      {cal_data['yaw_sensitivity']:.1f} deg/s per unit")
            print(f"[CALIBRATION]   Throttle sensitivity: {cal_data['throttle_sensitivity']:.1f} kts/s per unit")
            print(f"[CALIBRATION]   Confidence:           {cal_data['confidence']:.0%}")
            print(f"[CALIBRATION]   Samples:              {cal_data['samples']}")
            print(f"[CALIBRATION] ═══════════════════════════════════\n")

            if self._on_complete:
                self._on_complete(cal_data)

            self._report_progress()

        return self._hold_steady(telemetry)

    def _report_progress(self) -> None:
        """Send progress update via callback."""
        if not self._on_progress:
            return

        step_idx = min(self._current_step_idx, len(self._steps) - 1)
        current = self._steps[step_idx]
        confidence = self._tracker.confidence

        self._on_progress(
            step_idx + 1,
            len(self._steps),
            current.label,
            confidence,
        )

    @property
    def tracker(self) -> SensitivityTracker:
        """Access the underlying sensitivity tracker."""
        return self._tracker

    @property
    def is_complete(self) -> bool:
        return self._completed
