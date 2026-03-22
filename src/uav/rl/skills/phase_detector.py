"""Telemetry-based flight phase detection.

Uses a state machine with hysteresis to determine what phase of flight
the aircraft is in. Used both during recording (to auto-label demos)
and during chain flight (to select the right skill model).
"""
from __future__ import annotations

import math
import time
from collections import deque

from uav.sim.types import Telemetry
from uav.rl.skills.skill_registry import Phase


class PhaseDetector:
    """Detect flight phase from telemetry stream."""

    def __init__(self, initial_phase: Phase = Phase.GROUND) -> None:
        self._phase = initial_phase
        self._phase_start = time.time()

        # Rolling history for derived values
        self._alt_history: deque[tuple[float, float]] = deque(maxlen=50)  # (time, alt_ft)
        self._hdg_history: deque[tuple[float, float]] = deque(maxlen=50)  # (time, hdg_deg)

        # Hysteresis timers — require condition to hold for N seconds
        self._candidate_phase: Phase | None = None
        self._candidate_since: float = 0.0

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def phase_duration(self) -> float:
        """Seconds in current phase."""
        return time.time() - self._phase_start

    def update(self, t: Telemetry) -> Phase:
        """Process one telemetry sample and return current phase."""
        now = time.time()

        if not t.is_valid():
            return self._phase

        # Update history
        self._alt_history.append((now, t.altitude_ft))
        self._hdg_history.append((now, t.heading_deg))

        # Derived values
        agl_ft = t.agl_m * 3.28084 if not math.isnan(t.agl_m) else 0.0
        climb_rate = self._compute_climb_rate()
        hdg_rate = self._compute_hdg_rate()

        # Determine candidate next phase
        candidate = self._evaluate_transitions(t, agl_ft, climb_rate, hdg_rate)

        if candidate != self._phase:
            if candidate != self._candidate_phase:
                # New candidate — start hysteresis timer
                self._candidate_phase = candidate
                self._candidate_since = now
            elif now - self._candidate_since >= self._hysteresis_time(candidate):
                # Candidate held long enough — transition
                self._phase = candidate
                self._phase_start = now
                self._candidate_phase = None
        else:
            # Current phase still valid — reset candidate
            self._candidate_phase = None

        return self._phase

    def get_extras(self) -> dict:
        """Return derived values useful for observation building."""
        return {
            "climb_rate_fpm": self._compute_climb_rate(),
            "hdg_rate_dps": self._compute_hdg_rate(),
        }

    def _evaluate_transitions(
        self, t: Telemetry, agl_ft: float, climb_rate: float, hdg_rate: float,
    ) -> Phase:
        """Evaluate which phase we should be in given current telemetry."""
        phase = self._phase

        if phase == Phase.GROUND:
            if agl_ft > 5.0 and t.airspeed_kts > 30.0:
                return Phase.TAKEOFF

        elif phase == Phase.TAKEOFF:
            if agl_ft > 100.0 and climb_rate > 200.0:
                return Phase.CLIMB

        elif phase == Phase.CLIMB:
            if abs(climb_rate) < 200.0:
                return Phase.CRUISE

        elif phase == Phase.CRUISE:
            if abs(t.roll_deg) > 10.0 and abs(hdg_rate) > 2.0:
                return Phase.TURN
            if climb_rate < -200.0:
                return Phase.DESCEND

        elif phase == Phase.TURN:
            if abs(t.roll_deg) < 5.0 and abs(hdg_rate) < 1.0:
                return Phase.CRUISE

        elif phase == Phase.DESCEND:
            if agl_ft < 200.0 and climb_rate < -100.0:
                return Phase.LAND
            if abs(climb_rate) < 100.0:
                return Phase.CRUISE  # leveled off

        elif phase == Phase.LAND:
            if agl_ft < 5.0 and t.airspeed_kts < 40.0:
                return Phase.GROUND

        return phase  # no transition

    def _hysteresis_time(self, phase: Phase) -> float:
        """How long a candidate phase must hold before we transition."""
        return {
            Phase.GROUND: 0.5,
            Phase.TAKEOFF: 0.5,
            Phase.CLIMB: 1.0,
            Phase.TURN: 0.5,
            Phase.CRUISE: 3.0,   # need 3s of stable flight to call it cruise
            Phase.DESCEND: 3.0,  # need 3s of descent to call it
            Phase.LAND: 1.0,
        }.get(phase, 2.0)

    def _compute_climb_rate(self) -> float:
        """Compute climb rate in ft/min from altitude history (linear regression)."""
        if len(self._alt_history) < 5:
            return 0.0

        times = [h[0] for h in self._alt_history]
        alts = [h[1] for h in self._alt_history]
        dt = times[-1] - times[0]
        if dt < 0.5:
            return 0.0

        # Simple linear regression slope
        n = len(times)
        t0 = times[0]
        sx = sum(t - t0 for t in times)
        sy = sum(alts)
        sxx = sum((t - t0) ** 2 for t in times)
        sxy = sum((t - t0) * a for t, a in zip(times, alts))

        denom = n * sxx - sx * sx
        if abs(denom) < 1e-6:
            return 0.0

        slope = (n * sxy - sx * sy) / denom  # ft/s
        return slope * 60.0  # ft/min

    def _compute_hdg_rate(self) -> float:
        """Compute heading rate in deg/s from heading history."""
        if len(self._hdg_history) < 5:
            return 0.0

        dt = self._hdg_history[-1][0] - self._hdg_history[0][0]
        if dt < 0.5:
            return 0.0

        # Sum angular deltas (handling wrap-around)
        total_delta = 0.0
        for i in range(1, len(self._hdg_history)):
            delta = self._hdg_history[i][1] - self._hdg_history[i - 1][1]
            while delta > 180.0:
                delta -= 360.0
            while delta < -180.0:
                delta += 360.0
            total_delta += delta

        return total_delta / dt  # deg/s
