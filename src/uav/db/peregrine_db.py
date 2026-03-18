"""Peregrine — Database operations for the autopilot.

All Supabase reads/writes go through here so the rest of the codebase
never touches the client directly.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .client import get_client

log = logging.getLogger(__name__)


# ── Aircraft ────────────────────────────────────────────────

def get_aircraft(icao_type: str) -> dict | None:
    """Fetch aircraft by ICAO type code (e.g. 'C172')."""
    resp = (
        get_client()
        .table("aircraft")
        .select("*")
        .eq("icao_type", icao_type)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def update_aircraft_envelope(icao_type: str, envelope_data: dict, version: int) -> None:
    """Update the learned envelope JSON for an aircraft."""
    get_client().table("aircraft").update({
        "envelope": envelope_data,
        "envelope_version": version,
    }).eq("icao_type", icao_type).execute()


def increment_flight_count(aircraft_id: str) -> None:
    """Bump flights_completed by 1."""
    aircraft = get_client().table("aircraft").select("flights_completed").eq("id", aircraft_id).limit(1).execute()
    if aircraft.data:
        count = (aircraft.data[0].get("flights_completed") or 0) + 1
        get_client().table("aircraft").update({"flights_completed": count}).eq("id", aircraft_id).execute()


# ── Airports & Runways ──────────────────────────────────────

def get_airport(icao_code: str) -> dict | None:
    """Fetch airport by ICAO code."""
    resp = (
        get_client()
        .table("airports")
        .select("*, runways(*)")
        .eq("icao_code", icao_code)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def get_runway(airport_icao: str, designator: str) -> dict | None:
    """Fetch a specific runway by airport ICAO and designator."""
    airport = get_airport(airport_icao)
    if not airport:
        return None
    for rwy in airport.get("runways", []):
        if designator in rwy.get("designator", ""):
            return rwy
    return None


def list_airports() -> list[dict]:
    """List all airports."""
    resp = get_client().table("airports").select("*").order("icao_code").execute()
    return resp.data or []


# ── Flights ─────────────────────────────────────────────────

def create_flight(
    aircraft_id: str | None = None,
    dep_airport_icao: str | None = None,
    dep_runway_designator: str | None = None,
    arr_airport_icao: str | None = None,
    arr_runway_designator: str | None = None,
    flight_mode: str = "normal",
    route_distance_nm: float | None = None,
    cruise_alt_target_ft: float | None = None,
) -> dict:
    """Create a new flight record. Returns the inserted row."""
    row: dict[str, Any] = {
        "flight_mode": flight_mode,
        "status": "in_progress",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    if aircraft_id:
        row["aircraft_id"] = aircraft_id
    if route_distance_nm is not None:
        row["route_distance_nm"] = route_distance_nm
    if cruise_alt_target_ft is not None:
        row["cruise_alt_target_ft"] = cruise_alt_target_ft

    # Resolve airport/runway IDs
    if dep_airport_icao:
        dep = get_airport(dep_airport_icao)
        if dep:
            row["dep_airport_id"] = dep["id"]
            if dep_runway_designator:
                for rwy in dep.get("runways", []):
                    if dep_runway_designator in rwy.get("designator", ""):
                        row["dep_runway_id"] = rwy["id"]
                        break

    if arr_airport_icao:
        arr = get_airport(arr_airport_icao)
        if arr:
            row["arr_airport_id"] = arr["id"]
            if arr_runway_designator:
                for rwy in arr.get("runways", []):
                    if arr_runway_designator in rwy.get("designator", ""):
                        row["arr_runway_id"] = rwy["id"]
                        break

    resp = get_client().table("flights").insert(row).execute()
    return resp.data[0]


def finalize_flight(
    flight_id: str,
    *,
    status: str = "completed",
    duration_s: float | None = None,
    score_total: float | None = None,
    score_accuracy: float | None = None,
    score_speed: float | None = None,
    score_time: float | None = None,
    score_stability: float | None = None,
    touchdown_speed_kts: float | None = None,
    touchdown_distance_ft: float | None = None,
    distance_from_dest_m: float | None = None,
    rollout_distance_ft: float | None = None,
    takeoff_roll_ft: float | None = None,
    rotate_speed_kts: float | None = None,
    cruise_alt_stddev_ft: float | None = None,
    cruise_alt_max_dev_ft: float | None = None,
    cruise_speed_avg_kts: float | None = None,
    phase_timeline: list | None = None,
    abort_reason: str | None = None,
) -> dict:
    """Update a flight with final results."""
    updates: dict[str, Any] = {
        "status": status,
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }

    # Only include non-None values
    optional = {
        "duration_s": duration_s,
        "score_total": score_total,
        "score_accuracy": score_accuracy,
        "score_speed": score_speed,
        "score_time": score_time,
        "score_stability": score_stability,
        "touchdown_speed_kts": touchdown_speed_kts,
        "touchdown_distance_ft": touchdown_distance_ft,
        "distance_from_dest_m": distance_from_dest_m,
        "rollout_distance_ft": rollout_distance_ft,
        "takeoff_roll_ft": takeoff_roll_ft,
        "rotate_speed_kts": rotate_speed_kts,
        "cruise_alt_stddev_ft": cruise_alt_stddev_ft,
        "cruise_alt_max_dev_ft": cruise_alt_max_dev_ft,
        "cruise_speed_avg_kts": cruise_speed_avg_kts,
        "phase_timeline": phase_timeline,
        "abort_reason": abort_reason,
    }
    updates.update({k: v for k, v in optional.items() if v is not None})

    resp = get_client().table("flights").update(updates).eq("id", flight_id).execute()
    return resp.data[0] if resp.data else updates


def get_flight_history(limit: int = 50) -> list[dict]:
    """Get recent flights, newest first."""
    resp = (
        get_client()
        .table("flights")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data or []


def get_personal_best(arr_airport_icao: str) -> dict | None:
    """Get the highest-scoring completed flight to a destination."""
    airport = get_airport(arr_airport_icao)
    if not airport:
        return None
    resp = (
        get_client()
        .table("flights")
        .select("*")
        .eq("arr_airport_id", airport["id"])
        .eq("status", "completed")
        .order("score_total", desc=True)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


# ── Flight Events ───────────────────────────────────────────

def log_event(
    flight_id: str,
    event_type: str,
    *,
    message: str | None = None,
    altitude_ft: float | None = None,
    agl_ft: float | None = None,
    airspeed_kts: float | None = None,
    heading_deg: float | None = None,
    vertical_speed_fpm: float | None = None,
    lat: float | None = None,
    lon: float | None = None,
    throttle: float | None = None,
    flap_pct: float | None = None,
    phase: str | None = None,
    payload: dict | None = None,
) -> None:
    """Insert a flight event. Fire-and-forget — errors are logged, not raised."""
    row: dict[str, Any] = {
        "flight_id": flight_id,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    optional = {
        "message": message,
        "altitude_ft": altitude_ft,
        "agl_ft": agl_ft,
        "airspeed_kts": airspeed_kts,
        "heading_deg": heading_deg,
        "vertical_speed_fpm": vertical_speed_fpm,
        "lat": lat,
        "lon": lon,
        "throttle": throttle,
        "flap_pct": flap_pct,
        "phase": phase,
        "payload": payload,
    }
    row.update({k: v for k, v in optional.items() if v is not None})

    try:
        get_client().table("flight_events").insert(row).execute()
    except Exception as e:
        log.warning("Failed to log event %s: %s", event_type, e)


def get_flight_events(flight_id: str) -> list[dict]:
    """Get all events for a flight, ordered by timestamp."""
    resp = (
        get_client()
        .table("flight_events")
        .select("*")
        .eq("flight_id", flight_id)
        .order("timestamp")
        .execute()
    )
    return resp.data or []


# ── Telemetry Snapshots ─────────────────────────────────────

def save_telemetry_snapshot(flight_id: str, snapshot: dict) -> None:
    """Insert a telemetry snapshot. Fire-and-forget."""
    snapshot["flight_id"] = flight_id
    snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
    try:
        get_client().table("telemetry_snapshots").insert(snapshot).execute()
    except Exception as e:
        log.warning("Failed to save telemetry snapshot: %s", e)


# ── Learn Sessions ──────────────────────────────────────────

def create_learn_session(aircraft_id: str, flight_id: str | None = None) -> dict:
    """Start a new learn session."""
    row = {
        "aircraft_id": aircraft_id,
        "status": "in_progress",
    }
    if flight_id:
        row["flight_id"] = flight_id
    resp = get_client().table("learn_sessions").insert(row).execute()
    return resp.data[0]


def update_learn_session(
    session_id: str,
    *,
    status: str | None = None,
    cards_completed: list | None = None,
    discoveries: list | None = None,
    confidence_after: float | None = None,
) -> None:
    """Update a learn session."""
    updates: dict[str, Any] = {}
    if status:
        updates["status"] = status
        if status in ("completed", "aborted"):
            updates["ended_at"] = datetime.now(timezone.utc).isoformat()
    if cards_completed is not None:
        updates["cards_completed"] = cards_completed
    if discoveries is not None:
        updates["discoveries"] = discoveries
    if confidence_after is not None:
        updates["confidence_after"] = confidence_after

    if updates:
        get_client().table("learn_sessions").update(updates).eq("id", session_id).execute()
