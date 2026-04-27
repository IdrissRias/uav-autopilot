"""Peregrine — Envelope Updater.

After each flight, merges new observations into the aircraft's
learned_envelope in local SQLite. Background sync pushes to Supabase.

Uses exponential moving average so recent flights have more weight,
but old measurements aren't lost.

Confidence increases with more samples:
  1 flight  → ~50% confidence
  3 flights → ~75% confidence
  10 flights → ~90% confidence
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from ..db import local_db

log = logging.getLogger(__name__)

# How much weight the newest observation gets (0.0-1.0).
# 0.3 = new observation is 30% of the updated value,
# old running average is 70%.
EMA_ALPHA = 0.3


def update_envelope(icao_type: str, new_observations: Dict[str, Any]) -> Dict[str, Any]:
    """Merge new flight observations into the aircraft's learned envelope.

    Args:
        icao_type: Aircraft ICAO type code (e.g. "SF50")
        new_observations: Output from FlightObserver.compile()

    Returns:
        The updated learned_envelope dict (after merge)
    """
    row = local_db.get_aircraft(icao_type)

    if not row:
        log.error(f"Aircraft {icao_type} not found in database")
        return {}

    existing: Dict[str, Any] = row.get("envelope") or {}
    flights_done = (row.get("flights_completed") or 0) + 1

    # Merge speeds
    existing_speeds = existing.get("speeds_kts", {})
    new_speeds = new_observations.get("speeds_kts", {})
    merged_speeds = _merge_section(existing_speeds, new_speeds, flights_done)

    # Merge performance
    existing_perf = existing.get("performance", {})
    new_perf = new_observations.get("performance", {})
    merged_perf = _merge_section(existing_perf, new_perf, flights_done)

    # Build updated envelope
    updated_envelope = {
        "speeds_kts": merged_speeds,
        "performance": merged_perf,
        "last_updated": time.time(),
        "total_flights": flights_done,
        "last_phase_timeline": new_observations.get("phase_timeline", []),
    }

    # Preserve any existing pid_tuned data
    if "pid_tuned" in existing:
        updated_envelope["pid_tuned"] = existing["pid_tuned"]

    # Merge calibration sensitivity data (EMA blend)
    new_cal = new_observations.get("calibration", {})
    existing_cal = existing.get("calibration", {})
    if new_cal.get("confidence", 0) > 0:
        if existing_cal.get("confidence", 0) > 0:
            # EMA blend existing + new
            updated_envelope["calibration"] = {
                "pitch_sensitivity": round(
                    EMA_ALPHA * new_cal.get("pitch_sensitivity", 0)
                    + (1 - EMA_ALPHA) * existing_cal.get("pitch_sensitivity", 0), 2),
                "roll_sensitivity": round(
                    EMA_ALPHA * new_cal.get("roll_sensitivity", 0)
                    + (1 - EMA_ALPHA) * existing_cal.get("roll_sensitivity", 0), 2),
                "yaw_sensitivity": round(
                    EMA_ALPHA * new_cal.get("yaw_sensitivity", 0)
                    + (1 - EMA_ALPHA) * existing_cal.get("yaw_sensitivity", 0), 2),
                "throttle_sensitivity": round(
                    EMA_ALPHA * new_cal.get("throttle_sensitivity", 0)
                    + (1 - EMA_ALPHA) * existing_cal.get("throttle_sensitivity", 0), 2),
                "confidence": round(min(0.95,
                    max(new_cal.get("confidence", 0), existing_cal.get("confidence", 0))), 3),
                "samples": existing_cal.get("samples", 0) + new_cal.get("samples", 0),
            }
        else:
            # First calibration observation
            updated_envelope["calibration"] = new_cal
    elif existing_cal:
        updated_envelope["calibration"] = existing_cal

    # Write to SQLite (sync queue auto-pushes to Supabase)
    try:
        local_db.update_aircraft_envelope(icao_type, updated_envelope, flights_done)
        local_db.increment_flight_count(icao_type)

        log.info(f"[ENVELOPE] Updated {icao_type} envelope (flight #{flights_done})")
        _log_changes(existing_speeds, merged_speeds, existing_perf, merged_perf)

    except Exception as e:
        log.error(f"Failed to update envelope for {icao_type}: {e}")

    return updated_envelope


def _merge_section(
    existing: Dict[str, Any],
    new: Dict[str, Any],
    total_flights: int,
) -> Dict[str, Any]:
    """Merge a section (speeds or performance) using EMA.

    For each key in `new`:
    - If it doesn't exist in `existing` → use new value directly
    - If it exists → EMA blend: value = alpha * new + (1-alpha) * old
    - Confidence grows with samples: min(0.95, samples / (samples + 3))
    """
    merged = dict(existing)  # start with everything we already have

    for key, new_entry in new.items():
        if not isinstance(new_entry, dict) or "value" not in new_entry:
            continue

        new_val = new_entry["value"]
        new_samples = new_entry.get("samples", 1)

        if key in existing and isinstance(existing[key], dict) and "value" in existing[key]:
            # Blend with existing
            old_val = existing[key]["value"]
            old_samples = existing[key].get("samples", 1)

            blended_val = EMA_ALPHA * new_val + (1 - EMA_ALPHA) * old_val
            total_samples = old_samples + new_samples

            # Confidence grows with total samples
            confidence = min(0.95, total_samples / (total_samples + 3))

            merged[key] = {
                "value": round(blended_val, 1),
                "confidence": round(confidence, 2),
                "samples": total_samples,
                "source": "flight_observation",
                "last_raw": round(new_val, 1),  # keep the raw latest for debugging
            }
        else:
            # First observation of this parameter
            merged[key] = {
                "value": round(new_val, 1),
                "confidence": round(min(0.6, new_samples / (new_samples + 3)), 2),
                "samples": new_samples,
                "source": "flight_observation",
            }

    return merged


def _log_changes(
    old_speeds: Dict, new_speeds: Dict,
    old_perf: Dict, new_perf: Dict,
) -> None:
    """Log what changed in this update."""
    for section_name, old, new in [("SPEED", old_speeds, new_speeds),
                                    ("PERF", old_perf, new_perf)]:
        for key in new:
            if not isinstance(new[key], dict):
                continue
            new_val = new[key].get("value")
            conf = new[key].get("confidence", 0)
            if key in old and isinstance(old[key], dict):
                old_val = old[key].get("value")
                delta = new_val - old_val if new_val and old_val else 0
                if abs(delta) > 0.5:
                    log.info(f"  [{section_name}] {key}: {old_val} → {new_val} "
                            f"(Δ{delta:+.1f}, conf {conf:.0%})")
            else:
                log.info(f"  [{section_name}] {key}: NEW = {new_val} (conf {conf:.0%})")
