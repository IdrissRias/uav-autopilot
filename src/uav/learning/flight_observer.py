"""Peregrine — Flight Observer.

Watches every tick of a flight, measures actual aircraft performance,
and writes learned values back to the aircraft's envelope in Supabase
after landing.

Every value the autopilot uses (V_rotate, V_cruise, climb rate, etc.)
should eventually come from measurements made here — not from hardcoded
seeds.

The observer is passive: it never commands the aircraft.  It only reads
telemetry and records what actually happened.

KEY MEASUREMENTS (the 4 energy curves):

1. ACCELERATION CURVE — speed vs distance during takeoff roll.
   Tells us: how much runway we need, where the abort point is.

2. DECELERATION CURVE — speed vs time/distance when throttle is cut in the air.
   Tells us: when to start slowing for approach (if we're at 200kts and need
   83kts at the runway, we need to know how many nm that takes).

3. BRAKING CURVE — speed vs distance after touchdown.
   Tells us: how much runway we need to stop after landing.

4. MINIMUM LIFTOFF SPEED — the actual speed the plane gets airborne.
   Not the book number. The real number for this plane in this sim.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ── Measurement helpers ──────────────────────────────────────

@dataclass
class SpeedEvent:
    """A speed measurement at a specific moment."""
    speed_kts: float
    altitude_ft: float
    agl_ft: float
    timestamp: float


@dataclass
class PhaseTiming:
    """When a phase started and ended."""
    phase: str
    entered_at: float
    exited_at: Optional[float] = None
    duration_s: float = 0.0


@dataclass
class CurveSample:
    """A single point on a speed-vs-distance or speed-vs-time curve."""
    time_s: float           # seconds since curve started
    speed_kts: float
    distance_ft: float      # distance traveled since curve started
    altitude_ft: float
    lat: float
    lon: float


# ── Control Sensitivity Tracker ──────────────────────────────

class SensitivityTracker:
    """Measures control sensitivity: how much aircraft response per unit input.

    Passively observes actuator commands vs telemetry response.
    Correlates input changes with delayed response changes (~0.3s lag).

    Outputs:
        pitch_sensitivity:    deg/s pitch rate per unit elevator input
        roll_sensitivity:     deg/s roll rate per unit aileron input
        yaw_sensitivity:      deg/s yaw rate per unit rudder input
        throttle_sensitivity: kts/s acceleration per unit throttle
    """

    # Response delay: typical aircraft control lag
    _RESPONSE_DELAY_S = 0.3
    # Minimum input change to count as a meaningful sample
    _MIN_INPUT_DELTA = 0.02
    # Rolling window size for computing sensitivity
    _WINDOW_SIZE = 200

    def __init__(self) -> None:
        # Buffers: (timestamp, input_value, response_rate)
        self._pitch_samples: list[tuple[float, float, float]] = []
        self._roll_samples: list[tuple[float, float, float]] = []
        self._yaw_samples: list[tuple[float, float, float]] = []
        self._throttle_samples: list[tuple[float, float, float]] = []

        # Previous values for delta computation
        self._prev_pitch_cmd: float | None = None
        self._prev_roll_cmd: float | None = None
        self._prev_yaw_cmd: float | None = None
        self._prev_throttle: float | None = None

        # Delayed input buffer: store (timestamp, cmd) to correlate with later response
        self._pitch_cmd_history: list[tuple[float, float]] = []
        self._roll_cmd_history: list[tuple[float, float]] = []
        self._yaw_cmd_history: list[tuple[float, float]] = []
        self._throttle_cmd_history: list[tuple[float, float]] = []

        # Previous telemetry for rate computation
        self._prev_pitch_deg: float | None = None
        self._prev_roll_deg: float | None = None
        self._prev_hdg_deg: float | None = None
        self._prev_speed_kts: float | None = None
        self._prev_time: float | None = None

    def tick(self, telemetry: Any, actuators: Any, now: float) -> None:
        """Call every autopilot tick with telemetry and actuator commands."""
        if actuators is None:
            return

        pitch_deg = getattr(telemetry, 'pitch_deg', 0.0)
        roll_deg = getattr(telemetry, 'roll_deg', 0.0)
        hdg_deg = getattr(telemetry, 'heading_deg', 0.0)
        speed_kts = getattr(telemetry, 'airspeed_kts', 0.0)

        pitch_cmd = getattr(actuators, 'pitch', 0.0)
        roll_cmd = getattr(actuators, 'roll', 0.0)
        yaw_cmd = getattr(actuators, 'yaw', 0.0)
        throttle = getattr(actuators, 'throttle', 0.0)

        # Compute telemetry rates
        if self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0.01:
                pitch_rate = (pitch_deg - (self._prev_pitch_deg or 0)) / dt
                roll_rate = (roll_deg - (self._prev_roll_deg or 0)) / dt
                # Heading rate with wrapping
                hdg_delta = hdg_deg - (self._prev_hdg_deg or 0)
                if hdg_delta > 180: hdg_delta -= 360
                if hdg_delta < -180: hdg_delta += 360
                hdg_rate = hdg_delta / dt
                speed_rate = (speed_kts - (self._prev_speed_kts or 0)) / dt

                # Store command history for delayed correlation
                self._pitch_cmd_history.append((now, pitch_cmd))
                self._roll_cmd_history.append((now, roll_cmd))
                self._yaw_cmd_history.append((now, yaw_cmd))
                self._throttle_cmd_history.append((now, throttle))

                # Trim old history
                cutoff = now - 2.0
                self._pitch_cmd_history = [(t, v) for t, v in self._pitch_cmd_history if t > cutoff]
                self._roll_cmd_history = [(t, v) for t, v in self._roll_cmd_history if t > cutoff]
                self._yaw_cmd_history = [(t, v) for t, v in self._yaw_cmd_history if t > cutoff]
                self._throttle_cmd_history = [(t, v) for t, v in self._throttle_cmd_history if t > cutoff]

                # Correlate: find command from RESPONSE_DELAY_S ago
                delayed_pitch = self._get_delayed_cmd(self._pitch_cmd_history, now)
                delayed_roll = self._get_delayed_cmd(self._roll_cmd_history, now)
                delayed_yaw = self._get_delayed_cmd(self._yaw_cmd_history, now)
                delayed_thr = self._get_delayed_cmd(self._throttle_cmd_history, now)

                # Record samples where input was non-trivial
                if delayed_pitch is not None and abs(delayed_pitch) > self._MIN_INPUT_DELTA:
                    self._pitch_samples.append((now, delayed_pitch, pitch_rate))
                if delayed_roll is not None and abs(delayed_roll) > self._MIN_INPUT_DELTA:
                    self._roll_samples.append((now, delayed_roll, roll_rate))
                if delayed_yaw is not None and abs(delayed_yaw) > self._MIN_INPUT_DELTA:
                    self._yaw_samples.append((now, delayed_yaw, hdg_rate))
                if delayed_thr is not None and abs(delayed_thr) > self._MIN_INPUT_DELTA:
                    self._throttle_samples.append((now, delayed_thr, speed_rate))

                # Trim to window size
                for buf in (self._pitch_samples, self._roll_samples,
                            self._yaw_samples, self._throttle_samples):
                    if len(buf) > self._WINDOW_SIZE:
                        del buf[:-self._WINDOW_SIZE]

        self._prev_pitch_deg = pitch_deg
        self._prev_roll_deg = roll_deg
        self._prev_hdg_deg = hdg_deg
        self._prev_speed_kts = speed_kts
        self._prev_time = now

        self._prev_pitch_cmd = pitch_cmd
        self._prev_roll_cmd = roll_cmd
        self._prev_yaw_cmd = yaw_cmd
        self._prev_throttle = throttle

    def _get_delayed_cmd(self, history: list[tuple[float, float]], now: float) -> float | None:
        """Find the command value from RESPONSE_DELAY_S ago."""
        target_time = now - self._RESPONSE_DELAY_S
        best = None
        best_dt = float('inf')
        for t, v in history:
            dt = abs(t - target_time)
            if dt < best_dt:
                best_dt = dt
                best = v
        # Only use if we found something within 0.1s of target
        return best if best_dt < 0.1 else None

    def _compute_sensitivity(self, samples: list[tuple[float, float, float]]) -> float:
        """Compute sensitivity as slope of response_rate vs input using least squares.

        Returns response_rate per unit input (deg/s per unit, or kts/s per unit).
        """
        if len(samples) < 20:
            return 0.0

        # Use numpy-free least squares: slope = Σ(xi*yi) / Σ(xi²)
        # where x = input command, y = response rate
        sum_xy = 0.0
        sum_xx = 0.0
        for _, cmd, rate in samples:
            sum_xy += cmd * rate
            sum_xx += cmd * cmd

        if sum_xx < 1e-8:
            return 0.0

        slope = sum_xy / sum_xx
        return abs(slope)  # sensitivity is always positive

    @property
    def pitch_sensitivity(self) -> float:
        return self._compute_sensitivity(self._pitch_samples)

    @property
    def roll_sensitivity(self) -> float:
        return self._compute_sensitivity(self._roll_samples)

    @property
    def yaw_sensitivity(self) -> float:
        return self._compute_sensitivity(self._yaw_samples)

    @property
    def throttle_sensitivity(self) -> float:
        return self._compute_sensitivity(self._throttle_samples)

    @property
    def total_samples(self) -> int:
        return (len(self._pitch_samples) + len(self._roll_samples)
                + len(self._yaw_samples) + len(self._throttle_samples))

    @property
    def confidence(self) -> float:
        """Calibration confidence based on sample count and diversity."""
        min_samples = min(
            len(self._pitch_samples),
            len(self._roll_samples),
            len(self._throttle_samples),
        )
        # Need at least 20 samples per axis for any confidence
        if min_samples < 20:
            return 0.0
        # Confidence grows: 50 samples → 0.5, 100 → 0.77, 200 → 0.95
        return min(0.95, min_samples / (min_samples + 50))

    def compile(self) -> dict:
        """Return calibration data for storage."""
        return {
            "pitch_sensitivity": round(self.pitch_sensitivity, 2),
            "roll_sensitivity": round(self.roll_sensitivity, 2),
            "yaw_sensitivity": round(self.yaw_sensitivity, 2),
            "throttle_sensitivity": round(self.throttle_sensitivity, 2),
            "confidence": round(self.confidence, 3),
            "samples": self.total_samples,
            "per_axis_samples": {
                "pitch": len(self._pitch_samples),
                "roll": len(self._roll_samples),
                "yaw": len(self._yaw_samples),
                "throttle": len(self._throttle_samples),
            },
        }


def _haversine_ft(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in feet between two lat/lon points."""
    R = 20_902_231.0  # earth radius in feet
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Main observer ────────────────────────────────────────────

class FlightObserver:
    """Observes a single flight and extracts learned performance values.

    Usage:
        observer = FlightObserver(icao_type="SF50")
        # Every autopilot tick:
        observer.observe(phase, telemetry, actuators)
        # After landing:
        learned = observer.compile()
        # Write to DB via envelope_updater.update_envelope()
    """

    def __init__(self, icao_type: str):
        self.icao_type = icao_type
        self._start_time = time.monotonic()
        self._tick_count = 0

        # ── Phase tracking ──
        self._current_phase: str = ""
        self._phase_history: List[PhaseTiming] = []

        # ── Takeoff measurements ──
        self._ground_start_speed: Optional[float] = None
        self._on_ground = True
        self._liftoff_speed: Optional[float] = None
        self._liftoff_time: Optional[float] = None
        self._rotate_speed: Optional[float] = None
        self._rotate_detected = False
        self._takeoff_start_pos: Optional[Tuple[float, float]] = None
        self._takeoff_roll_started = False

        # ── CURVE 1: Acceleration (takeoff roll) ──
        # Every tick during ground roll: record (time, speed, distance_from_start)
        self._accel_curve: List[CurveSample] = []
        self._accel_start_time: Optional[float] = None

        # ── CURVE 2: Deceleration (air braking) ──
        # Record whenever throttle drops to idle during flight
        # Tracks: how fast speed bleeds at different starting speeds
        self._decel_curve: List[CurveSample] = []
        self._decel_tracking = False
        self._decel_start_time: Optional[float] = None
        self._decel_start_pos: Optional[Tuple[float, float]] = None
        self._decel_start_speed: Optional[float] = None
        self._decel_segments: List[Dict[str, Any]] = []  # completed segments

        # ── CURVE 3: Braking (ground after touchdown) ──
        self._brake_curve: List[CurveSample] = []
        self._brake_start_time: Optional[float] = None
        self._brake_start_pos: Optional[Tuple[float, float]] = None
        self._brake_start_speed: Optional[float] = None
        self._full_stop_detected = False

        # ── Climb measurements ──
        self._climb_samples: List[Dict[str, float]] = []
        self._best_climb_rate_fpm: float = 0.0
        self._best_climb_speed_kts: float = 0.0

        # ── Cruise measurements ──
        self._cruise_speeds: List[float] = []
        self._cruise_altitudes: List[float] = []
        self._cruise_throttles: List[float] = []

        # ── Approach/landing measurements ──
        self._approach_speeds: List[float] = []
        self._approach_descent_rates: List[float] = []
        self._touchdown_speed: Optional[float] = None
        self._touchdown_time: Optional[float] = None
        self._touchdown_pos: Optional[Tuple[float, float]] = None
        self._pre_flare_speed: Optional[float] = None
        self._landing_detected = False

        # ── Speed gate measurements ──
        # Record speed at specific AGL altitudes during approach
        self._speed_gates: Dict[int, float] = {}  # {agl_ft: speed_kts}
        self._speed_gate_altitudes = [500, 200, 100, 50]
        self._speed_gates_captured: set = set()

        # ── Stall detection ──
        self._min_speed_observed: float = float('inf')
        self._stall_events: List[SpeedEvent] = []

        # ── Altitude tracking for vertical speed computation ──
        self._prev_alt: Optional[float] = None
        self._prev_time: Optional[float] = None
        self._prev_speed: Optional[float] = None
        self._vertical_speed_fpm: float = 0.0

        # ── Control sensitivity tracker (calibration) ──
        self._sensitivity = SensitivityTracker()

    # ── Tick-level observation ───────────────────────────────

    def observe(self, phase: str, telemetry: Any, actuators: Any = None) -> None:
        """Call every autopilot tick with current phase and telemetry."""
        self._tick_count += 1
        now = time.monotonic()

        # Extract telemetry
        speed = getattr(telemetry, 'airspeed_kts', 0.0)
        alt = getattr(telemetry, 'altitude_ft', 0.0)
        agl_m = getattr(telemetry, 'agl_m', 0.0)
        agl_ft = agl_m * 3.28084 if not math.isnan(agl_m) else 0.0
        pitch = getattr(telemetry, 'pitch_deg', 0.0)
        roll = getattr(telemetry, 'roll_deg', 0.0)
        lat = getattr(telemetry, 'lat_deg', 0.0)
        lon = getattr(telemetry, 'lon_deg', 0.0)
        throttle = getattr(actuators, 'throttle', 0.0) if actuators else 0.0
        brake = getattr(actuators, 'brake_ratio', 0.0) if actuators else 0.0

        # Compute vertical speed
        if self._prev_alt is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if dt > 0.01:
                self._vertical_speed_fpm = (alt - self._prev_alt) / dt * 60.0
        self._prev_alt = alt
        self._prev_time = now

        # Track min speed (in controlled flight, ignore ground/taxi)
        if speed > 30.0 and agl_ft > 50.0:
            self._min_speed_observed = min(self._min_speed_observed, speed)

        # ── Phase transitions ──
        if phase != self._current_phase:
            if self._phase_history:
                self._phase_history[-1].exited_at = now
                self._phase_history[-1].duration_s = now - self._phase_history[-1].entered_at
            self._phase_history.append(PhaseTiming(phase=phase, entered_at=now))
            self._current_phase = phase

        # ── Phase-specific observations ──
        if phase == "GROUND":
            self._observe_ground(speed, alt, agl_ft, pitch, lat, lon, throttle, now)
        elif phase == "CLIMB":
            self._observe_climb(speed, alt, agl_ft, throttle, now)
        elif phase == "CRUISE":
            self._observe_cruise(speed, alt, throttle, lat, lon, now)
        elif phase == "APPROACH":
            self._observe_approach(speed, alt, agl_ft, throttle, lat, lon, now)
        elif phase == "LAND":
            self._observe_land(speed, alt, agl_ft, lat, lon, throttle, brake, now)

        # ── Air deceleration tracking (runs across phases) ──
        self._track_deceleration(speed, alt, agl_ft, throttle, lat, lon, now)

        # ── Control sensitivity measurement (runs every tick) ──
        self._sensitivity.tick(telemetry, actuators, now)

        # ── Passive stall detection ──
        if (speed < self._min_speed_observed * 1.2
                and pitch < -10.0
                and abs(roll) > 15.0
                and agl_ft > 500.0):
            self._stall_events.append(SpeedEvent(
                speed_kts=speed, altitude_ft=alt,
                agl_ft=agl_ft, timestamp=now
            ))

        self._prev_speed = speed

    # ── GROUND phase ─────────────────────────────────────────

    def _observe_ground(self, speed, alt, agl_ft, pitch, lat, lon, throttle, now):
        """Measure takeoff roll + acceleration curve."""
        # Detect start of takeoff roll
        if throttle > 0.5 and not self._takeoff_roll_started:
            self._takeoff_roll_started = True
            self._ground_start_speed = speed
            self._takeoff_start_pos = (lat, lon)
            self._accel_start_time = now
            log.info(f"[OBSERVER] Takeoff roll started at {speed:.1f} kts")

        # Record acceleration curve point
        if self._takeoff_roll_started and self._takeoff_start_pos:
            dist_ft = _haversine_ft(
                self._takeoff_start_pos[0], self._takeoff_start_pos[1],
                lat, lon
            )
            self._accel_curve.append(CurveSample(
                time_s=now - (self._accel_start_time or now),
                speed_kts=speed,
                distance_ft=dist_ft,
                altitude_ft=alt,
                lat=lat, lon=lon,
            ))

        # Rotation detection
        if (self._takeoff_roll_started and not self._rotate_detected
                and pitch > 5.0 and speed > 30.0):
            self._rotate_speed = speed
            self._rotate_detected = True
            log.info(f"[OBSERVER] Rotation at {speed:.1f} kts, "
                     f"{self._accel_curve[-1].distance_ft:.0f}ft roll" if self._accel_curve else "")

        # Liftoff detection
        if self._on_ground and agl_ft > 5.0 and speed > 30.0:
            self._on_ground = False
            self._liftoff_speed = speed
            self._liftoff_time = now
            roll_dist = self._accel_curve[-1].distance_ft if self._accel_curve else 0
            log.info(f"[OBSERVER] Liftoff at {speed:.1f} kts, {roll_dist:.0f}ft ground roll")

    # ── CLIMB phase ──────────────────────────────────────────

    def _observe_climb(self, speed, alt, agl_ft, throttle, now):
        """Measure climb performance."""
        if self._on_ground and agl_ft > 5.0:
            self._on_ground = False
            self._liftoff_speed = speed
            self._liftoff_time = now

        vs = self._vertical_speed_fpm
        self._climb_samples.append({
            "speed_kts": speed,
            "altitude_ft": alt,
            "vs_fpm": vs,
            "throttle": throttle,
            "time": now - self._start_time,
        })

        if vs > self._best_climb_rate_fpm:
            self._best_climb_rate_fpm = vs
            self._best_climb_speed_kts = speed

    # ── CRUISE phase ─────────────────────────────────────────

    def _observe_cruise(self, speed, alt, throttle, lat, lon, now):
        """Measure cruise performance."""
        if speed > 20.0:
            self._cruise_speeds.append(speed)
            self._cruise_altitudes.append(alt)
            self._cruise_throttles.append(throttle)

    # ── APPROACH phase ───────────────────────────────────────

    def _observe_approach(self, speed, alt, agl_ft, throttle, lat, lon, now):
        """Measure approach performance + speed gates."""
        if speed > 20.0:
            self._approach_speeds.append(speed)
            self._approach_descent_rates.append(abs(self._vertical_speed_fpm))

        # Speed gate captures at specific altitudes
        for gate_alt in self._speed_gate_altitudes:
            if gate_alt not in self._speed_gates_captured:
                # Capture when we cross through the gate altitude (within ±20ft)
                if abs(agl_ft - gate_alt) < 20.0:
                    self._speed_gates[gate_alt] = speed
                    self._speed_gates_captured.add(gate_alt)
                    log.info(f"[OBSERVER] {gate_alt}ft gate: {speed:.1f} kts")

        # 50ft pre-flare speed
        if self._pre_flare_speed is None and agl_ft <= 50.0 and agl_ft > 10.0:
            self._pre_flare_speed = speed
            log.info(f"[OBSERVER] 50ft gate speed: {speed:.1f} kts")

    # ── LAND phase ───────────────────────────────────────────

    def _observe_land(self, speed, alt, agl_ft, lat, lon, throttle, brake, now):
        """Measure landing performance + braking curve."""
        # Touchdown detection
        if not self._landing_detected and agl_ft < 3.0 and speed > 20.0:
            self._landing_detected = True
            self._touchdown_speed = speed
            self._touchdown_time = now
            self._touchdown_pos = (lat, lon)
            # Start braking curve from this point
            self._brake_start_time = now
            self._brake_start_pos = (lat, lon)
            self._brake_start_speed = speed
            log.info(f"[OBSERVER] Touchdown at {speed:.1f} kts")

        # Record braking curve after touchdown
        if self._landing_detected and self._brake_start_pos and not self._full_stop_detected:
            dist_ft = _haversine_ft(
                self._brake_start_pos[0], self._brake_start_pos[1],
                lat, lon
            )
            self._brake_curve.append(CurveSample(
                time_s=now - (self._brake_start_time or now),
                speed_kts=speed,
                distance_ft=dist_ft,
                altitude_ft=alt,
                lat=lat, lon=lon,
            ))

            # Detect full stop (speed drops below 5 kts)
            if speed < 5.0:
                self._full_stop_detected = True
                log.info(f"[OBSERVER] Full stop after {dist_ft:.0f}ft rollout "
                         f"({now - (self._brake_start_time or now):.1f}s)")

    # ── Air deceleration tracking ────────────────────────────

    def _track_deceleration(self, speed, alt, agl_ft, throttle, lat, lon, now):
        """Track speed bleed when throttle is at idle in the air.

        This runs across all phases. Whenever throttle drops below 10%
        and we're above 500ft AGL, we start recording a deceleration segment.
        When throttle goes back up or we descend below 200ft, the segment ends.

        This gives us: "from 200kts at idle, it takes X seconds / Y nm to reach 83kts"
        """
        in_air = agl_ft > 200.0
        idle = throttle < 0.10
        has_speed = speed > 30.0

        if idle and in_air and has_speed and not self._decel_tracking:
            # Start a new deceleration segment
            self._decel_tracking = True
            self._decel_start_time = now
            self._decel_start_pos = (lat, lon)
            self._decel_start_speed = speed
            self._decel_curve = []

        if self._decel_tracking:
            if not idle or not in_air or speed < 20.0:
                # End this segment
                self._decel_tracking = False
                if (self._decel_start_speed is not None
                        and len(self._decel_curve) > 5
                        and self._decel_start_speed - speed > 10.0):
                    # Only save segments where we actually decelerated > 10kts
                    self._decel_segments.append({
                        "start_speed_kts": round(self._decel_start_speed, 1),
                        "end_speed_kts": round(speed, 1),
                        "speed_lost_kts": round(self._decel_start_speed - speed, 1),
                        "duration_s": round(now - (self._decel_start_time or now), 1),
                        "distance_ft": round(self._decel_curve[-1].distance_ft, 0) if self._decel_curve else 0,
                        "altitude_ft": round(alt, 0),
                        "samples": len(self._decel_curve),
                        "decel_rate_kts_per_s": round(
                            (self._decel_start_speed - speed)
                            / max(0.1, now - (self._decel_start_time or now)),
                            2
                        ),
                    })
                    log.info(f"[OBSERVER] Decel segment: {self._decel_segments[-1]}")
            else:
                # Record point
                dist_ft = _haversine_ft(
                    self._decel_start_pos[0], self._decel_start_pos[1],
                    lat, lon
                ) if self._decel_start_pos else 0
                self._decel_curve.append(CurveSample(
                    time_s=now - (self._decel_start_time or now),
                    speed_kts=speed,
                    distance_ft=dist_ft,
                    altitude_ft=alt,
                    lat=lat, lon=lon,
                ))

    # ── Compile learned values ───────────────────────────────

    def compile(self) -> Dict[str, Any]:
        """Compile all observations into a learned envelope update.

        Returns a dict structured for merging into the aircraft's
        learned_envelope JSON column.
        """
        learned: Dict[str, Any] = {
            "speeds_kts": {},
            "performance": {},
            "curves": {},
            "meta": {
                "observed_at": time.time(),
                "ticks": self._tick_count,
                "duration_s": time.monotonic() - self._start_time,
                "phases_seen": [p.phase for p in self._phase_history],
            }
        }

        speeds = learned["speeds_kts"]
        perf = learned["performance"]
        curves = learned["curves"]

        # ────────────────────────────────────────────────────
        # CURVE 1: Acceleration (takeoff roll)
        # ────────────────────────────────────────────────────
        if len(self._accel_curve) > 5:
            # Downsample to key points: every 10 kts
            accel_points = []
            last_speed_bucket = 0
            for sample in self._accel_curve:
                bucket = int(sample.speed_kts / 10) * 10
                if bucket > last_speed_bucket:
                    accel_points.append({
                        "speed_kts": round(sample.speed_kts, 1),
                        "distance_ft": round(sample.distance_ft, 0),
                        "time_s": round(sample.time_s, 1),
                    })
                    last_speed_bucket = bucket

            curves["acceleration"] = {
                "points": accel_points,
                "total_distance_ft": round(self._accel_curve[-1].distance_ft, 0),
                "total_time_s": round(self._accel_curve[-1].time_s, 1),
                "start_speed_kts": round(self._accel_curve[0].speed_kts, 1),
                "end_speed_kts": round(self._accel_curve[-1].speed_kts, 1),
                "samples": len(self._accel_curve),
            }

            # Compute acceleration rate (kts per second, average)
            if self._accel_curve[-1].time_s > 0:
                accel_rate = (
                    (self._accel_curve[-1].speed_kts - self._accel_curve[0].speed_kts)
                    / self._accel_curve[-1].time_s
                )
                perf["ground_accel_kts_per_s"] = {
                    "value": round(accel_rate, 2),
                    "confidence": min(0.85, len(self._accel_curve) / 50),
                    "samples": len(self._accel_curve),
                    "source": "flight_observation",
                }

            # Takeoff roll distance
            perf["takeoff_roll_ft"] = {
                "value": round(self._accel_curve[-1].distance_ft, 0),
                "confidence": 0.8,
                "samples": 1,
                "source": "flight_observation",
            }

        # ────────────────────────────────────────────────────
        # CURVE 2: Air deceleration
        # ────────────────────────────────────────────────────
        if self._decel_segments:
            curves["deceleration"] = {
                "segments": self._decel_segments,
                "count": len(self._decel_segments),
            }

            # Average deceleration rate across all segments
            rates = [s["decel_rate_kts_per_s"] for s in self._decel_segments]
            avg_decel = sum(rates) / len(rates)
            perf["air_decel_kts_per_s"] = {
                "value": round(avg_decel, 2),
                "confidence": min(0.85, len(self._decel_segments) / 3),
                "samples": len(self._decel_segments),
                "source": "flight_observation",
                "note": "average speed loss per second at idle throttle",
            }

            # Compute: how many nm to slow from cruise to approach
            # This is the KEY number for knowing when to start descent
            total_speed_loss = sum(s["speed_lost_kts"] for s in self._decel_segments)
            total_distance = sum(s["distance_ft"] for s in self._decel_segments)
            if total_speed_loss > 0:
                perf["decel_ft_per_kt"] = {
                    "value": round(total_distance / total_speed_loss, 0),
                    "confidence": min(0.8, len(self._decel_segments) / 3),
                    "samples": len(self._decel_segments),
                    "source": "flight_observation",
                    "note": "feet of travel needed to lose 1 kt of airspeed at idle",
                }

        # ────────────────────────────────────────────────────
        # CURVE 3: Braking (ground after touchdown)
        # ────────────────────────────────────────────────────
        if len(self._brake_curve) > 3:
            # Downsample to key points
            brake_points = []
            last_bucket = 999
            for sample in self._brake_curve:
                bucket = int(sample.speed_kts / 10) * 10
                if bucket < last_bucket:
                    brake_points.append({
                        "speed_kts": round(sample.speed_kts, 1),
                        "distance_ft": round(sample.distance_ft, 0),
                        "time_s": round(sample.time_s, 1),
                    })
                    last_bucket = bucket

            curves["braking"] = {
                "points": brake_points,
                "touchdown_speed_kts": round(self._brake_start_speed or 0, 1),
                "total_distance_ft": round(self._brake_curve[-1].distance_ft, 0),
                "total_time_s": round(self._brake_curve[-1].time_s, 1),
                "full_stop": self._full_stop_detected,
                "samples": len(self._brake_curve),
            }

            perf["braking_distance_ft"] = {
                "value": round(self._brake_curve[-1].distance_ft, 0),
                "confidence": 0.75 if self._full_stop_detected else 0.4,
                "samples": 1,
                "source": "flight_observation",
                "note": "distance from touchdown to full stop" if self._full_stop_detected
                        else "distance from touchdown to end of observation (did not fully stop)",
            }

            # Braking deceleration rate
            if self._brake_curve[-1].time_s > 0 and self._brake_start_speed:
                brake_rate = (
                    (self._brake_start_speed - self._brake_curve[-1].speed_kts)
                    / self._brake_curve[-1].time_s
                )
                perf["brake_decel_kts_per_s"] = {
                    "value": round(brake_rate, 2),
                    "confidence": 0.7 if self._full_stop_detected else 0.4,
                    "samples": len(self._brake_curve),
                    "source": "flight_observation",
                }

        # ────────────────────────────────────────────────────
        # SPEED VALUES
        # ────────────────────────────────────────────────────

        # V_rotate
        if self._rotate_speed is not None:
            speeds["v_rotate"] = {
                "value": round(self._rotate_speed, 1),
                "confidence": 0.8,
                "samples": 1,
                "source": "flight_observation",
            }

        # V_liftoff (actual speed when wheels left the ground)
        if self._liftoff_speed is not None:
            speeds["v_liftoff"] = {
                "value": round(self._liftoff_speed, 1),
                "confidence": 0.85,
                "samples": 1,
                "source": "flight_observation",
            }

        # Best climb rate & speed
        if self._best_climb_rate_fpm > 0:
            perf["best_climb_rate_fpm"] = {
                "value": round(self._best_climb_rate_fpm, 0),
                "confidence": min(0.9, len(self._climb_samples) / 100),
                "samples": len(self._climb_samples),
                "source": "flight_observation",
            }
            speeds["v_best_climb"] = {
                "value": round(self._best_climb_speed_kts, 1),
                "confidence": min(0.7, len(self._climb_samples) / 100),
                "samples": len(self._climb_samples),
                "source": "flight_observation",
            }

        # Cruise speed
        if len(self._cruise_speeds) > 10:
            sorted_speeds = sorted(self._cruise_speeds)
            median_speed = sorted_speeds[len(sorted_speeds) // 2]
            speeds["v_cruise"] = {
                "value": round(median_speed, 1),
                "confidence": min(0.9, len(self._cruise_speeds) / 200),
                "samples": len(self._cruise_speeds),
                "source": "flight_observation",
            }
            if self._cruise_throttles:
                sorted_thr = sorted(self._cruise_throttles)
                perf["cruise_throttle"] = {
                    "value": round(sorted_thr[len(sorted_thr) // 2], 3),
                    "confidence": min(0.8, len(self._cruise_throttles) / 200),
                    "samples": len(self._cruise_throttles),
                    "source": "flight_observation",
                }

        # Approach speed
        if len(self._approach_speeds) > 10:
            sorted_app = sorted(self._approach_speeds)
            median_app = sorted_app[len(sorted_app) // 2]
            speeds["v_approach"] = {
                "value": round(median_app, 1),
                "confidence": min(0.8, len(self._approach_speeds) / 100),
                "samples": len(self._approach_speeds),
                "source": "flight_observation",
            }
            if self._approach_descent_rates:
                sorted_dr = sorted(self._approach_descent_rates)
                perf["approach_descent_fpm"] = {
                    "value": round(sorted_dr[len(sorted_dr) // 2], 0),
                    "confidence": min(0.7, len(self._approach_descent_rates) / 100),
                    "samples": len(self._approach_descent_rates),
                    "source": "flight_observation",
                }

        # Landing speed
        if self._touchdown_speed is not None:
            speeds["v_land"] = {
                "value": round(self._touchdown_speed, 1),
                "confidence": 0.7,
                "samples": 1,
                "source": "flight_observation",
            }

        # Speed gates (speed at specific AGL altitudes during approach)
        if self._speed_gates:
            curves["speed_gates"] = {
                f"{alt}ft": round(spd, 1)
                for alt, spd in sorted(self._speed_gates.items(), reverse=True)
            }

        # 50ft gate speed
        if self._pre_flare_speed is not None:
            speeds["v_50ft_gate"] = {
                "value": round(self._pre_flare_speed, 1),
                "confidence": 0.7,
                "samples": 1,
                "source": "flight_observation",
            }

        # Stall speed estimate
        if self._min_speed_observed < float('inf') and self._min_speed_observed > 20.0:
            perf["min_observed_speed_kts"] = {
                "value": round(self._min_speed_observed, 1),
                "confidence": 0.5,
                "samples": 1,
                "source": "flight_observation",
                "note": "lowest speed in controlled flight, NOT stall speed",
            }

        # Stall events
        if self._stall_events:
            stall_speeds = [e.speed_kts for e in self._stall_events]
            speeds["v_stall_observed"] = {
                "value": round(max(stall_speeds), 1),
                "confidence": min(0.9, len(stall_speeds) / 3),
                "samples": len(stall_speeds),
                "source": "flight_observation",
            }

        # Phase timeline
        learned["phase_timeline"] = [
            {
                "phase": p.phase,
                "entered_s": round(p.entered_at - self._start_time, 1),
                "duration_s": round(p.duration_s, 1) if p.exited_at else None,
            }
            for p in self._phase_history
        ]

        # Control sensitivity (calibration data)
        learned["calibration"] = self._sensitivity.compile()

        return learned

    def summary(self) -> str:
        """Human-readable summary of what was learned this flight."""
        data = self.compile()
        lines = [
            f"╔══════════════════════════════════════════════╗",
            f"║  PEREGRINE FLIGHT OBSERVER — {self.icao_type:>6s}        ║",
            f"╠══════════════════════════════════════════════╣",
            f"║  Duration: {data['meta']['duration_s']:.0f}s | Ticks: {data['meta']['ticks']:<6d}       ║",
            f"║  Phases: {' → '.join(data['meta']['phases_seen']):<35s}║",
            f"╠══════════════════════════════════════════════╣",
        ]

        # Speeds
        lines.append("║  LEARNED SPEEDS                              ║")
        for key, val in data["speeds_kts"].items():
            lines.append(f"║  {key:22s} {val['value']:>6.1f} kts ({val['confidence']:>3.0%}) ║")

        # Performance
        lines.append("╠══════════════════════════════════════════════╣")
        lines.append("║  LEARNED PERFORMANCE                         ║")
        for key, val in data["performance"].items():
            v = val['value']
            unit = "fpm" if "fpm" in key else "ft" if "_ft" in key else "kts/s" if "per_s" in key else "ft/kt" if "per_kt" in key else ""
            lines.append(f"║  {key:22s} {v:>8.1f} {unit:<5s} ({val['confidence']:>3.0%}) ║")

        # Curves
        c = data.get("curves", {})
        if c:
            lines.append("╠══════════════════════════════════════════════╣")
            lines.append("║  ENERGY CURVES                               ║")
            if "acceleration" in c:
                a = c["acceleration"]
                lines.append(f"║  Takeoff: {a['total_distance_ft']:.0f}ft in {a['total_time_s']:.1f}s"
                           f" ({a['start_speed_kts']:.0f}→{a['end_speed_kts']:.0f}kts)")
            if "deceleration" in c:
                d = c["deceleration"]
                lines.append(f"║  Air decel: {d['count']} segments recorded")
                for seg in d["segments"]:
                    lines.append(f"║    {seg['start_speed_kts']}→{seg['end_speed_kts']}kts "
                               f"in {seg['duration_s']}s ({seg['distance_ft']}ft)")
            if "braking" in c:
                b = c["braking"]
                stop = "✓ STOPPED" if b["full_stop"] else "⚠ partial"
                lines.append(f"║  Braking: {b['total_distance_ft']:.0f}ft in {b['total_time_s']:.1f}s"
                           f" ({b['touchdown_speed_kts']:.0f}kts) {stop}")
            if "speed_gates" in c:
                gates = c["speed_gates"]
                gate_str = " | ".join(f"{k}={v}kts" for k, v in gates.items())
                lines.append(f"║  Gates: {gate_str}")

        # Calibration
        cal = data.get("calibration", {})
        if cal.get("confidence", 0) > 0:
            lines.append("╠══════════════════════════════════════════════╣")
            lines.append("║  CONTROL SENSITIVITY (CALIBRATION)           ║")
            lines.append(f"║  Pitch:    {cal.get('pitch_sensitivity', 0):>6.1f} deg/s per unit       ║")
            lines.append(f"║  Roll:     {cal.get('roll_sensitivity', 0):>6.1f} deg/s per unit       ║")
            lines.append(f"║  Yaw:      {cal.get('yaw_sensitivity', 0):>6.1f} deg/s per unit       ║")
            lines.append(f"║  Throttle: {cal.get('throttle_sensitivity', 0):>6.1f} kts/s per unit       ║")
            lines.append(f"║  Confidence: {cal.get('confidence', 0):>3.0%} ({cal.get('samples', 0)} samples)       ║")

        lines.append("╚══════════════════════════════════════════════╝")
        return "\n".join(lines)
