"""
Flight performance scorer.

After each flight, scores across four dimensions and saves to logs/scores.json.
Every flight gets compared to the personal best — improvements are highlighted.

Score (0–100):
  40 pts  Landing accuracy   — distance from destination (0 = >500m, 40 = <50m)
  30 pts  Landing speed      — vs v_land (0 = >110kts, 30 = ≤v_land+5kts)
  20 pts  Flight time        — vs personal best for this route (20 = PB, 0 = 2× PB)
  10 pts  Cruise stability   — std-dev of altitude vs cruise target (0 = ±300ft, 10 = ±10ft)
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ── geo helper ────────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ── result ────────────────────────────────────────────────────────────────────

@dataclass
class FlightScore:
    timestamp: str
    dest_icao: str

    # Raw measurements
    duration_s: float
    landing_dist_m: float          # distance from dest coords at touchdown
    landing_speed_kts: float       # airspeed at moment of touchdown
    cruise_alt_std_ft: float       # std-dev of altitude vs target during cruise

    # Partial scores
    pts_accuracy: float            # 0–40
    pts_speed: float               # 0–30
    pts_time: float                # 0–20
    pts_stability: float           # 0–10
    total: float                   # 0–100

    # Context
    is_personal_best: bool = False
    prev_best_total: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    # Outcome of the flight — drives the icon in Flight History and makes
    # it obvious at a glance why a score is low. "completed" = touched down
    # normally on LAND tick; "aborted" = user ended early or process killed;
    # "crashed" = underground/stuck in ABORT phase detected by the engine.
    outcome: str = "completed"


# ── scorer ────────────────────────────────────────────────────────────────────

class FlightScorer:
    """
    Instantiate at the start of a flight, call update() every tick,
    finalize() on touchdown, then save() to persist the score.
    """

    # Fallback only — the real reference comes from the learned envelope
    # via v_land_kts. The old hardcoded 65 judged the SF50 (v_land 98.4)
    # as "way too fast" on every landing it ever made, including an
    # 88-kt touchdown 10 kts UNDER its own book number.
    V_LAND_KTAS = 65.0

    def __init__(
        self,
        dest_icao: str,
        dest_lat: float,
        dest_lon: float,
        cruise_target_ft: float,
        log_dir: str = "logs",
        v_land_kts: float | None = None,
    ) -> None:
        self.dest_icao = dest_icao
        self.dest_lat = dest_lat
        self.dest_lon = dest_lon
        self.cruise_target_ft = cruise_target_ft
        self.log_dir = log_dir
        if v_land_kts and v_land_kts > 0:
            self.V_LAND_KTAS = float(v_land_kts)

        self._start_ts = time.time()
        self._cruise_alt_samples: List[float] = []  # deviation from target each tick

    def update(self, phase: str, altitude_ft: float) -> None:
        """Call every autopilot tick to accumulate cruise stability samples."""
        if phase == "CRUISE":
            self._cruise_alt_samples.append(altitude_ft - self.cruise_target_ft)

    def finalize(
        self,
        landing_lat: float,
        landing_lon: float,
        landing_speed_kts: float,
    ) -> FlightScore:
        duration_s = time.time() - self._start_ts
        landing_dist_m = _haversine_m(landing_lat, landing_lon, self.dest_lat, self.dest_lon)

        # Cruise alt stability
        if self._cruise_alt_samples:
            mean = sum(self._cruise_alt_samples) / len(self._cruise_alt_samples)
            variance = sum((x - mean) ** 2 for x in self._cruise_alt_samples) / len(self._cruise_alt_samples)
            cruise_alt_std = math.sqrt(variance)
        else:
            cruise_alt_std = 999.0

        # ── Scoring ──────────────────────────────────────────────────────────

        # Accuracy: 40 pts linear 50m → 500m
        pts_acc = max(0.0, 40.0 * (1.0 - max(0.0, landing_dist_m - 50.0) / 450.0))

        # Speed: 30 pts — full points at v_land+5kts, 0 at v_land+35kts
        spd_over = max(0.0, landing_speed_kts - (self.V_LAND_KTAS + 5.0))
        pts_spd = max(0.0, 30.0 * (1.0 - spd_over / 30.0))

        # Time: 20 pts vs personal best for this route
        prev_best_time = self._load_best_time()
        if prev_best_time is None:
            pts_time = 20.0  # first flight always gets full time score
        else:
            ratio = duration_s / max(1.0, prev_best_time)  # 1.0 = matches PB, 2.0 = twice as slow
            pts_time = max(0.0, 20.0 * (2.0 - ratio))

        # Stability: 10 pts — full at ±10ft std-dev, 0 at ±300ft
        pts_stab = max(0.0, 10.0 * (1.0 - max(0.0, cruise_alt_std - 10.0) / 290.0))

        total = pts_acc + pts_spd + pts_time + pts_stab

        # Personal best?
        prev_best = self._load_best_total()
        is_pb = prev_best is None or total > prev_best

        notes = []
        if landing_dist_m < 50:
            notes.append("🎯 Landed on the dot!")
        elif landing_dist_m < 150:
            notes.append("✅ Very accurate landing")
        elif landing_dist_m > 400:
            notes.append("⚠️  Missed by a lot — approach heading needs work")
        if landing_speed_kts > self.V_LAND_KTAS + 15.0:
            notes.append("🔴 Way too fast on touchdown — speed bleed needs improvement")
        elif landing_speed_kts > self.V_LAND_KTAS + 5.0:
            notes.append("🟡 A bit fast on touchdown")
        else:
            notes.append("✅ Good landing speed")
        if is_pb:
            notes.append(f"🏆 NEW PERSONAL BEST! (+{total - (prev_best or 0):.1f} pts)")

        return FlightScore(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            dest_icao=self.dest_icao,
            duration_s=round(duration_s, 1),
            landing_dist_m=round(landing_dist_m, 1),
            landing_speed_kts=round(landing_speed_kts, 1),
            cruise_alt_std_ft=round(cruise_alt_std, 1),
            pts_accuracy=round(pts_acc, 1),
            pts_speed=round(pts_spd, 1),
            pts_time=round(pts_time, 1),
            pts_stability=round(pts_stab, 1),
            total=round(total, 1),
            is_personal_best=is_pb,
            prev_best_total=round(prev_best, 1) if prev_best is not None else None,
            notes=notes,
            outcome="completed",
        )

    def finalize_partial(
        self,
        current_lat: float,
        current_lon: float,
        current_speed_kts: float,
        outcome: str,
    ) -> FlightScore:
        """Score a flight that didn't reach a proper touchdown.

        Policy: every flight produces a score, including aborts and crashes.
        We use whatever measurements are honest:
          • accuracy  → distance from current position to destination
                        (if user aborts at the destination, they still get credit;
                         if they abort mid-cruise, the score reflects that)
          • speed     → current airspeed (penalizes bailing at high speed)
          • time      → 0 for partial flights (no personal-best comparison
                        for something that didn't complete the route)
          • stability → std-dev of cruise alt samples accumulated so far;
                        0 if never reached cruise
        Crashed flights get an extra -10 "crash penalty" note baked into
        the total (but clamped at 0).
        """
        if outcome not in ("completed", "aborted", "crashed"):
            outcome = "aborted"

        duration_s = time.time() - self._start_ts
        landing_dist_m = _haversine_m(current_lat, current_lon, self.dest_lat, self.dest_lon)

        # Cruise alt stability — same as finalize()
        if self._cruise_alt_samples:
            mean = sum(self._cruise_alt_samples) / len(self._cruise_alt_samples)
            variance = sum((x - mean) ** 2 for x in self._cruise_alt_samples) / len(self._cruise_alt_samples)
            cruise_alt_std = math.sqrt(variance)
        else:
            # No cruise samples → 0 points. Don't use the 999 sentinel because
            # that would give a meaningless but defined zero regardless; explicit is cleaner.
            cruise_alt_std = 999.0

        pts_acc = max(0.0, 40.0 * (1.0 - max(0.0, landing_dist_m - 50.0) / 450.0))
        spd_over = max(0.0, current_speed_kts - (self.V_LAND_KTAS + 5.0))
        pts_spd = max(0.0, 30.0 * (1.0 - spd_over / 30.0))
        # Partial flights don't earn time points — they never completed the route.
        pts_time = 0.0
        pts_stab = max(0.0, 10.0 * (1.0 - max(0.0, cruise_alt_std - 10.0) / 290.0))

        total = pts_acc + pts_spd + pts_time + pts_stab
        if outcome == "crashed":
            total = max(0.0, total - 10.0)  # crash penalty

        prev_best = self._load_best_total()
        # Only "completed" flights can set a personal best.
        is_pb = (outcome == "completed") and (prev_best is None or total > prev_best)

        notes: List[str] = []
        if outcome == "aborted":
            notes.append("⏹️  Flight aborted — partial score")
        elif outcome == "crashed":
            notes.append("💥 Crashed — score reduced by 10 pts")
        if landing_dist_m > 400:
            notes.append(f"📍 Ended {landing_dist_m:.0f}m from destination")
        if current_speed_kts > 90:
            notes.append("🔴 Ended at high speed")

        return FlightScore(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            dest_icao=self.dest_icao,
            duration_s=round(duration_s, 1),
            landing_dist_m=round(landing_dist_m, 1),
            landing_speed_kts=round(current_speed_kts, 1),
            cruise_alt_std_ft=round(cruise_alt_std, 1),
            pts_accuracy=round(pts_acc, 1),
            pts_speed=round(pts_spd, 1),
            pts_time=round(pts_time, 1),
            pts_stability=round(pts_stab, 1),
            total=round(total, 1),
            is_personal_best=is_pb,
            prev_best_total=round(prev_best, 1) if prev_best is not None else None,
            notes=notes,
            outcome=outcome,
        )

    def save(self, score: FlightScore) -> str:
        """Append score to logs/scores.json and return the file path."""
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "scores.json")
        history = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(asdict(score))
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        return path

    def print_summary(self, score: FlightScore) -> None:
        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  FLIGHT SCORE — {score.dest_icao}  {score.timestamp}")
        print(sep)
        print(f"  Duration         {score.duration_s:.0f}s")
        print(f"  Landing distance {score.landing_dist_m:.0f}m from target")
        print(f"  Landing speed    {score.landing_speed_kts:.0f}kts")
        print(f"  Cruise alt σ     ±{score.cruise_alt_std_ft:.0f}ft")
        print(sep)
        print(f"  Accuracy  {score.pts_accuracy:5.1f} / 40")
        print(f"  Speed     {score.pts_speed:5.1f} / 30")
        print(f"  Time      {score.pts_time:5.1f} / 20")
        print(f"  Stability {score.pts_stability:5.1f} / 10")
        print(f"  ─────────────────")
        print(f"  TOTAL     {score.total:5.1f} / 100")
        if score.prev_best_total is not None:
            delta = score.total - score.prev_best_total
            sign = "+" if delta >= 0 else ""
            print(f"  vs PB     {sign}{delta:.1f} pts")
        print(sep)
        for n in score.notes:
            print(f"  {n}")
        print(sep + "\n")

    # ── persistence helpers ───────────────────────────────────────────────────

    def _load_history(self) -> list:
        path = os.path.join(self.log_dir, "scores.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return []

    def _load_best_total(self) -> Optional[float]:
        scores = [s["total"] for s in self._load_history() if s.get("dest_icao") == self.dest_icao]
        return max(scores) if scores else None

    def _load_best_time(self) -> Optional[float]:
        times = [s["duration_s"] for s in self._load_history() if s.get("dest_icao") == self.dest_icao]
        return min(times) if times else None
