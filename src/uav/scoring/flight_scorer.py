"""
Flight performance scorer.

After each flight, scores across four dimensions and saves to logs/scores.json.
Every flight gets compared to the personal best — improvements are highlighted.

Score (0–100) — QUALITY of flight, not just "did it land":
  30 pts  Path adherence     — RMS of altitude error vs the PLANNED target,
                               every airborne tick (climb, cruise, descent,
                               approach). Full ≤20 ft, 0 ≥170 ft. A porpoising
                               descent that balloons ±150 ft off the glideslope
                               is scored here, where it was invisible before.
  25 pts  Ride smoothness    — vertical-speed reversal rate (the wobble). A
                               smooth flight barely changes VS sign; a
                               porpoising one flips it constantly.
                               Full ≤0.04 /s, 0 ≥0.28 /s.
  30 pts  Landing accuracy   — touchdown distance from the aim point.
                               Full ≤40 m, 0 ≥300 m.
  15 pts  Landing speed      — vs v_land. Full ≤v_land+3, 0 ≥v_land+22.

Adherence + smoothness = 55 % of the score, so a wobbly flight cannot buy a
high score with a lucky touchdown. (Field names pts_time/pts_stability are
retained for storage compatibility; pts_time now carries adherence and
pts_stability carries smoothness.)
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
        # Whole-flight quality accumulators (see update()).
        self._alt_err_samples: List[float] = []  # |alt − target| every airborne tick
        self._vs_reversals: int = 0
        self._vs_prev_sign: int = 0
        self._airborne_ticks: int = 0
        # Pitch-command thrash: the elevator railing ±full is the clearest
        # wobble signal (a porpoising descent slams the stick; the VS
        # reversal count alone missed it, diluted by smooth climb/cruise).
        self._pitch_reversals: int = 0
        self._pitch_prev: Optional[float] = None
        self._pitch_dir: int = 0

    # Phases that are NOT airborne flying — excluded from adherence/wobble.
    _GROUND_PHASES = {"GROUND", "TAKEOFF", "TAKEOFF_ROLL", "ROLLOUT", "STOP"}

    def update(
        self,
        phase: str,
        altitude_ft: float,
        vs_fpm: float | None = None,
        target_alt_ft: float | None = None,
        pitch_cmd: float | None = None,
    ) -> None:
        """Call every autopilot tick. Accumulates whole-flight quality:
        altitude error vs the planned target, and vertical-speed reversals
        (the wobble). Ground phases are excluded — flare sink and rollout
        are not 'off target'."""
        if phase == "CRUISE":
            self._cruise_alt_samples.append(altitude_ft - self.cruise_target_ft)

        if phase in self._GROUND_PHASES:
            return
        self._airborne_ticks += 1

        # Adherence: distance from the commanded altitude at this instant.
        # FLARE is excluded from adherence (it is intentionally leaving the
        # glideslope for the runway) but still counts for wobble.
        if (target_alt_ft is not None and phase != "FLARE"
                and not math.isnan(target_alt_ft)):
            self._alt_err_samples.append(abs(altitude_ft - target_alt_ft))

        # Wobble: count vertical-speed sign flips above a noise floor.
        if vs_fpm is not None and not math.isnan(vs_fpm):
            sign = 1 if vs_fpm > 60.0 else (-1 if vs_fpm < -60.0 else 0)
            if sign != 0:
                if self._vs_prev_sign != 0 and sign != self._vs_prev_sign:
                    self._vs_reversals += 1
                self._vs_prev_sign = sign

        # Pitch-command thrash: count direction reversals of the elevator
        # command larger than a deadband. A porpoising descent reverses
        # constantly (the ±full railing we see); a smooth one holds.
        if pitch_cmd is not None and not math.isnan(pitch_cmd):
            if self._pitch_prev is not None:
                delta = pitch_cmd - self._pitch_prev
                if abs(delta) > 0.08:
                    d = 1 if delta > 0 else -1
                    if self._pitch_dir != 0 and d != self._pitch_dir:
                        self._pitch_reversals += 1
                    self._pitch_dir = d
            self._pitch_prev = pitch_cmd

    def _quality_points(self) -> tuple:
        """(pts_adherence 0–30, pts_smoothness 0–25, alt_rms_ft, vs_rev_rate).
        Shared by finalize() and finalize_partial()."""
        if self._alt_err_samples:
            alt_rms = math.sqrt(
                sum(e * e for e in self._alt_err_samples)
                / len(self._alt_err_samples))
        else:
            alt_rms = 999.0
        pts_adh = max(0.0, 30.0 * (1.0 - max(0.0, alt_rms - 20.0) / 150.0))

        # Reversals per second of airborne time (tick rate independent).
        secs = max(1.0, time.time() - self._start_ts)
        vs_rate = self._vs_reversals / secs
        pitch_rate = self._pitch_reversals / secs
        # Smoothness is the WORSE of the two signals: a flight that rails
        # the elevator (pitch thrash) is wobbly even if the resulting VS
        # swings are slow. Pitch thrash is faster, so its scale is tighter.
        pts_vs = max(0.0, 25.0 * (1.0 - max(0.0, vs_rate - 0.04) / 0.24))
        pts_pitch = max(0.0, 25.0 * (1.0 - max(0.0, pitch_rate - 0.15) / 0.85))
        pts_smooth = min(pts_vs, pts_pitch)
        # Report the dominant wobble rate for the notes.
        rev_rate = max(vs_rate, pitch_rate)
        return pts_adh, pts_smooth, alt_rms, rev_rate

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

        # ── Scoring (quality-first) ──────────────────────────────────────────
        # Accuracy: 30 pts, full ≤40 m, 0 ≥300 m (stricter than the old 50/500).
        pts_acc = max(0.0, 30.0 * (1.0 - max(0.0, landing_dist_m - 40.0) / 260.0))

        # Speed: 15 pts, full ≤v_land+3, 0 ≥v_land+22.
        spd_over = max(0.0, landing_speed_kts - (self.V_LAND_KTAS + 3.0))
        pts_spd = max(0.0, 15.0 * (1.0 - spd_over / 19.0))

        # Adherence (stored in pts_time) + smoothness (stored in pts_stability).
        pts_time, pts_stab, alt_rms, rev_rate = self._quality_points()

        total = pts_acc + pts_spd + pts_time + pts_stab

        # Personal best?
        prev_best = self._load_best_total()
        is_pb = prev_best is None or total > prev_best

        notes = []
        if landing_dist_m < 40:
            notes.append("🎯 Landed on the dot!")
        elif landing_dist_m < 120:
            notes.append("✅ Accurate landing")
        elif landing_dist_m > 300:
            notes.append("⚠️  Missed by a lot — approach needs work")
        # Flight-quality notes (the point of the rescoring).
        if alt_rms > 100.0:
            notes.append(f"🔴 Wandered off the planned path ({alt_rms:.0f} ft RMS)")
        elif alt_rms < 30.0:
            notes.append("✅ Held the path tightly")
        if rev_rate > 0.18:
            notes.append(f"🔴 Wobbly ride — porpoised ({rev_rate:.2f} VS flips/s)")
        elif rev_rate < 0.06:
            notes.append("✅ Smooth ride")
        if landing_speed_kts > self.V_LAND_KTAS + 12.0:
            notes.append("🔴 Too fast on touchdown")
        elif landing_speed_kts > self.V_LAND_KTAS + 3.0:
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
            cruise_alt_std_ft=round(alt_rms, 1),  # airborne path RMS
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

        pts_acc = max(0.0, 30.0 * (1.0 - max(0.0, landing_dist_m - 40.0) / 260.0))
        spd_over = max(0.0, current_speed_kts - (self.V_LAND_KTAS + 3.0))
        pts_spd = max(0.0, 15.0 * (1.0 - spd_over / 19.0))
        # Adherence + smoothness earned up to the point it stopped flying —
        # a smooth-then-aborted flight still shows the quality it had.
        pts_time, pts_stab, alt_rms, rev_rate = self._quality_points()

        total = pts_acc + pts_spd + pts_time + pts_stab
        if outcome == "crashed":
            total = max(0.0, total - 25.0)  # crash penalty (was 10)

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
            cruise_alt_std_ft=round(alt_rms, 1),  # airborne path RMS
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
