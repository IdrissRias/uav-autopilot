"""
Target-tracking accuracy scorer.

Computes a continuous accuracy percentage (0–100%) measuring how well
the aircraft is tracking its targets at each moment.  Each flight phase
defines which metrics matter and how they're weighted.

Usage:
    tracker = AccuracyTracker()
    # Every autopilot tick:
    tracker.update(phase, telemetry, targets)
    pct = tracker.instant_accuracy   # live % for this tick
    avg = tracker.flight_accuracy    # rolling average for the whole flight

The tracker is completely separate from FlightScorer (which scores
landing precision + time).  This is the "how well are you flying right now?"
metric.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from uav.sim.types import Telemetry, Targets


# ── Tolerance bands ──────────────────────────────────────────────────────────
# Each metric has a "perfect" band (100% score) and a "zero" threshold (0%).
# Between them, score is linear.

@dataclass(frozen=True)
class Band:
    """Tolerance band for a single metric."""
    perfect: float   # error <= this → 100%
    zero: float      # error >= this → 0%

    def score(self, error: float) -> float:
        """Return 0.0–1.0 score for the given absolute error."""
        if error <= self.perfect:
            return 1.0
        if error >= self.zero:
            return 0.0
        return 1.0 - (error - self.perfect) / (self.zero - self.perfect)


# Per-metric tolerance bands
BANDS = {
    "altitude_ft":  Band(perfect=100.0, zero=500.0),   # ±100ft perfect, ±500ft fail
    "heading_deg":  Band(perfect=5.0,   zero=30.0),    # ±5° perfect, ±30° fail
    "airspeed_kts": Band(perfect=5.0,   zero=25.0),    # ±5kts perfect, ±25kts fail
    "roll_deg":     Band(perfect=3.0,   zero=20.0),    # ±3° perfect, ±20° fail
    "power_pct":    Band(perfect=5.0,   zero=30.0),    # ±5% perfect, ±30% fail
}

# Per-phase weights: which metrics matter and how much.
# Weights per phase sum to 1.0.
PHASE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "GROUND": {
        "heading_deg":  0.40,   # hold runway heading
        "airspeed_kts": 0.30,   # accelerate to rotation speed
        "roll_deg":     0.30,   # wings level on ground
    },
    "CLIMB": {
        "altitude_ft":  0.25,   # climbing toward target
        "heading_deg":  0.25,   # turn toward destination
        "airspeed_kts": 0.25,   # maintain climb speed
        "roll_deg":     0.15,   # reasonable bank during turn
        "power_pct":    0.10,   # climb power
    },
    "CRUISE": {
        "altitude_ft":  0.40,   # hold altitude (most important)
        "heading_deg":  0.30,   # track to destination
        "roll_deg":     0.10,   # wings mostly level
        "power_pct":    0.20,   # hold 80% power
    },
    "APPROACH": {
        "altitude_ft":  0.25,   # follow glidepath
        "heading_deg":  0.30,   # track to runway
        "airspeed_kts": 0.25,   # hold approach speed
        "roll_deg":     0.20,   # wings level on final
    },
    "LAND": {
        "heading_deg":  0.30,   # hold runway heading
        "airspeed_kts": 0.20,   # decelerate to v_land
        "roll_deg":     0.40,   # WINGS LEVEL — critical
        "altitude_ft":  0.10,   # flare
    },
}


def _wrap_deg(a: float) -> float:
    """Wrap angle difference to [-180, +180]."""
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


@dataclass
class TickScore:
    """Score for a single autopilot tick."""
    phase: str
    accuracy_pct: float          # 0–100
    metric_scores: Dict[str, float]  # per-metric 0–1.0
    metric_errors: Dict[str, float]  # raw errors for debugging


class AccuracyTracker:
    """Tracks target-following accuracy across a flight."""

    def __init__(self) -> None:
        self._tick_scores: List[float] = []     # all tick accuracies
        self._phase_scores: Dict[str, List[float]] = {}  # per-phase
        self._last_tick: Optional[TickScore] = None
        self._total_ticks = 0

    @property
    def instant_accuracy(self) -> float:
        """Last tick's accuracy percentage (0–100)."""
        return self._last_tick.accuracy_pct if self._last_tick else 0.0

    @property
    def flight_accuracy(self) -> float:
        """Rolling average accuracy across all ticks (0–100)."""
        if not self._tick_scores:
            return 0.0
        return sum(self._tick_scores) / len(self._tick_scores)

    @property
    def phase_accuracy(self) -> Dict[str, float]:
        """Average accuracy per phase."""
        result = {}
        for phase, scores in self._phase_scores.items():
            if scores:
                result[phase] = sum(scores) / len(scores)
        return result

    @property
    def last_tick(self) -> Optional[TickScore]:
        return self._last_tick

    def update(
        self,
        phase: str,
        telemetry: Telemetry,
        targets: Targets,
        actual_throttle: float = 0.0,
    ) -> TickScore:
        """Score one autopilot tick. Call every cycle (~15Hz).

        Parameters
        ----------
        phase : str
            Current flight phase (GROUND, CLIMB, CRUISE, APPROACH, LAND).
        telemetry : Telemetry
            Current aircraft state.
        targets : Targets
            What the autopilot wants the plane to achieve.
        actual_throttle : float
            Actual throttle command (0–1), from actuators.
        """
        weights = PHASE_WEIGHTS.get(phase, PHASE_WEIGHTS.get("CRUISE", {}))
        metric_scores: Dict[str, float] = {}
        metric_errors: Dict[str, float] = {}
        weighted_sum = 0.0
        weight_total = 0.0

        # ── ALTITUDE ────────────────────────────────────────────────────
        if "altitude_ft" in weights and targets.altitude_ft is not None:
            error = abs(telemetry.altitude_ft - targets.altitude_ft)
            score = BANDS["altitude_ft"].score(error)
            metric_scores["altitude_ft"] = score
            metric_errors["altitude_ft"] = error
            w = weights["altitude_ft"]
            weighted_sum += w * score
            weight_total += w

        # ── HEADING ─────────────────────────────────────────────────────
        if "heading_deg" in weights and targets.heading_deg is not None:
            error = abs(_wrap_deg(telemetry.heading_deg - targets.heading_deg))
            score = BANDS["heading_deg"].score(error)
            metric_scores["heading_deg"] = score
            metric_errors["heading_deg"] = error
            w = weights["heading_deg"]
            weighted_sum += w * score
            weight_total += w

        # ── AIRSPEED ────────────────────────────────────────────────────
        # During CRUISE with fixed power (no airspeed target), skip this metric.
        if "airspeed_kts" in weights and targets.airspeed_kts is not None:
            # Only score airspeed if it's a real target (not just echoing current speed)
            # In cruise, airspeed_kts == current speed (informational), so skip.
            if phase != "CRUISE" or targets.throttle is None:
                error = abs(telemetry.airspeed_kts - targets.airspeed_kts)
                score = BANDS["airspeed_kts"].score(error)
                metric_scores["airspeed_kts"] = score
                metric_errors["airspeed_kts"] = error
                w = weights["airspeed_kts"]
                weighted_sum += w * score
                weight_total += w

        # ── ROLL (wings level) ──────────────────────────────────────────
        if "roll_deg" in weights:
            # Target roll = 0° (wings level). Some phases tolerate more bank.
            error = abs(telemetry.roll_deg)
            score = BANDS["roll_deg"].score(error)
            metric_scores["roll_deg"] = score
            metric_errors["roll_deg"] = error
            w = weights["roll_deg"]
            weighted_sum += w * score
            weight_total += w

        # ── POWER ───────────────────────────────────────────────────────
        if "power_pct" in weights and targets.throttle is not None:
            # Compare actual throttle to target throttle (both 0–1 scale)
            error = abs(actual_throttle - targets.throttle) * 100.0  # as percentage
            score = BANDS["power_pct"].score(error)
            metric_scores["power_pct"] = score
            metric_errors["power_pct"] = error
            w = weights["power_pct"]
            weighted_sum += w * score
            weight_total += w

        # ── Compute final accuracy ──────────────────────────────────────
        if weight_total > 0:
            accuracy_pct = (weighted_sum / weight_total) * 100.0
        else:
            accuracy_pct = 100.0  # no metrics to score

        tick = TickScore(
            phase=phase,
            accuracy_pct=round(accuracy_pct, 1),
            metric_scores=metric_scores,
            metric_errors=metric_errors,
        )

        self._last_tick = tick
        self._tick_scores.append(accuracy_pct)
        self._total_ticks += 1

        if phase not in self._phase_scores:
            self._phase_scores[phase] = []
        self._phase_scores[phase].append(accuracy_pct)

        return tick

    def summary(self) -> Dict:
        """Return a summary dict suitable for logging/broadcast."""
        return {
            "flight_accuracy_pct": round(self.flight_accuracy, 1),
            "total_ticks": self._total_ticks,
            "phase_accuracy": {
                phase: round(avg, 1)
                for phase, avg in self.phase_accuracy.items()
            },
        }
