"""Peregrine — Load aircraft envelope from local SQLite database.

The autopilot config (default.yaml) only specifies *which* aircraft to fly
via `airframe.icao_type`. All speeds, PID gains, and performance numbers
are pulled from the local SQLite `aircraft` table.

Priority: learned_envelope (from Learn Mode) > seed values (from POH).
Background sync keeps SQLite ↔ Supabase in agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from . import local_db


@dataclass
class AircraftEnvelope:
    """Everything the autopilot needs to know about the current aircraft."""

    # Identity
    icao_type: str = ""
    name: str = ""
    category: str = ""  # single_engine, light_jet, twin_turboprop, etc.

    # Speeds (kts) — learned values override seeds
    v_stall_clean: float = 0.0
    v_stall_flap: float = 0.0
    v_rotate: float = 0.0
    v_best_climb: float = 0.0
    v_cruise: float = 0.0
    v_approach: float = 0.0
    v_land: float = 0.0
    v_never_exceed: float = 0.0

    # Performance
    takeoff_roll_ft: float = 0.0
    best_climb_fpm: float = 0.0
    idle_sink_fpm: float = 500.0
    service_ceiling: float = 0.0

    # PID gains
    pid_heading: Dict[str, float] = field(default_factory=lambda: {"kp": 0.003, "ki": 0.0, "kd": 0.002})
    pid_altitude: Dict[str, float] = field(default_factory=lambda: {"kp": 0.005, "ki": 0.0, "kd": 0.002})
    pid_airspeed: Dict[str, float] = field(default_factory=lambda: {"kp": 0.015, "ki": 0.0, "kd": 0.004})

    # Cruise throttle (derived: throttle that sustains v_cruise in level flight)
    cruise_throttle: float = 0.65

    # Confidence (0.0 = seed only, 1.0 = fully learned)
    confidence: float = 0.0

    # Source tracking
    source: str = "seed"  # "seed", "learned", "hybrid"

    # Control sensitivity (measured during calibration)
    pitch_sensitivity: float = 0.0    # deg/s pitch rate per unit elevator input
    roll_sensitivity: float = 0.0     # deg/s roll rate per unit aileron input
    yaw_sensitivity: float = 0.0      # deg/s yaw rate per unit rudder input
    throttle_sensitivity: float = 0.0 # kts/s acceleration per unit throttle
    calibration_confidence: float = 0.0  # 0.0 = uncalibrated, 1.0 = fully calibrated
    calibration_samples: int = 0


def load_aircraft(icao_type: str) -> AircraftEnvelope:
    """Fetch aircraft envelope from local SQLite database.

    Reads the `aircraft` row for the given ICAO type code.
    If a `learned_envelope` JSON blob exists, its values take priority
    over the seed values.

    Raises RuntimeError if the aircraft isn't in the database.
    """
    row = local_db.get_aircraft(icao_type)

    if not row:
        raise RuntimeError(
            f"Aircraft '{icao_type}' not found in local database. "
            f"Run a sync or check seed data."
        )

    learned: Dict[str, Any] = row.get("envelope") or {}
    learned_speeds = learned.get("speeds_kts", {})
    learned_perf = learned.get("performance", {})
    learned_pids = learned.get("pid_tuned", {})

    def _pick(learned_key: str, seed_key: str, default: float = 0.0) -> float:
        """Use learned value if it exists and has confidence > 0.5, else seed."""
        lv = learned_speeds.get(learned_key) or learned_perf.get(learned_key)
        if isinstance(lv, dict) and lv.get("confidence", 0) > 0.5:
            return float(lv["value"])
        if isinstance(lv, (int, float)):
            return float(lv)
        val = row.get(seed_key)
        return float(val) if val is not None else default

    # Build PID gains — learned overrides seed
    pid_gains = row.get("pid_gains") or {}
    for key in ("heading", "altitude", "airspeed"):
        if key in learned_pids:
            pid_gains[key] = learned_pids[key]

    # Compute confidence: what fraction of speeds come from Learn Mode
    speed_keys = ["v_stall_clean", "v_stall_flap", "v_rotate", "v_best_climb",
                  "v_cruise", "v_approach", "v_land"]
    learned_count = sum(1 for k in speed_keys if k in learned_speeds)
    confidence = learned_count / len(speed_keys) if speed_keys else 0.0
    source = "learned" if confidence > 0.7 else "hybrid" if confidence > 0 else "seed"

    # Load calibration sensitivity data
    calibration = learned.get("calibration", {})

    envelope = AircraftEnvelope(
        icao_type=row.get("icao_type", icao_type),
        name=row.get("name", ""),
        category=row.get("category", ""),

        v_stall_clean=_pick("v_stall_clean", "seed_v_stall_clean", 60.0),
        v_stall_flap=_pick("v_stall_flap", "seed_v_stall_flap", 50.0),
        v_rotate=_pick("v_rotate", "seed_v_rotate", 70.0),
        v_best_climb=_pick("v_best_climb", "seed_v_best_climb", 120.0),
        v_cruise=_pick("v_cruise", "seed_v_cruise", 200.0),
        v_approach=_pick("v_approach", "seed_v_approach", 80.0),
        v_land=_pick("v_land", "seed_v_land", 70.0),
        v_never_exceed=_pick("v_never_exceed", "seed_v_never_exceed", 250.0),

        takeoff_roll_ft=_pick("takeoff_roll_ft", "seed_takeoff_roll_ft", 2000.0),
        best_climb_fpm=_pick("best_climb_fpm", "seed_best_climb_fpm", 1000.0),
        service_ceiling=_pick("service_ceiling", "seed_service_ceiling", 15000.0),

        pid_heading=pid_gains.get("heading", {"kp": 0.003, "ki": 0.0, "kd": 0.002}),
        pid_altitude=pid_gains.get("altitude", {"kp": 0.005, "ki": 0.0, "kd": 0.002}),
        pid_airspeed=pid_gains.get("airspeed", {"kp": 0.015, "ki": 0.0, "kd": 0.004}),

        confidence=confidence,
        source=source,

        pitch_sensitivity=float(calibration.get("pitch_sensitivity", 0.0)),
        roll_sensitivity=float(calibration.get("roll_sensitivity", 0.0)),
        yaw_sensitivity=float(calibration.get("yaw_sensitivity", 0.0)),
        throttle_sensitivity=float(calibration.get("throttle_sensitivity", 0.0)),
        calibration_confidence=float(calibration.get("confidence", 0.0)),
        calibration_samples=int(calibration.get("samples", 0)),
    )

    print(f"[PEREGRINE] Aircraft loaded from SQLite: {envelope.name} ({envelope.icao_type})")
    print(f"  Source: {envelope.source} | Confidence: {envelope.confidence:.0%}")
    print(f"  V_rotate={envelope.v_rotate} V_cruise={envelope.v_cruise} "
          f"V_approach={envelope.v_approach} V_land={envelope.v_land}")
    print(f"  Takeoff roll={envelope.takeoff_roll_ft}ft "
          f"Best climb={envelope.best_climb_fpm}fpm "
          f"Ceiling={envelope.service_ceiling}ft")
    if envelope.calibration_confidence > 0:
        print(f"  Calibration: {envelope.calibration_confidence:.0%} "
              f"(pitch={envelope.pitch_sensitivity:.1f} roll={envelope.roll_sensitivity:.1f} "
              f"throttle={envelope.throttle_sensitivity:.1f} deg-or-kts/s per unit)")

    return envelope
