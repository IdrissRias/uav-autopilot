"""Peregrine — Database operations for the autopilot.

All reads/writes go through local SQLite first (zero latency).
Background sync pushes changes to Supabase.

This module re-exports from local_db so existing imports still work.
"""

from __future__ import annotations

# Re-export everything from local_db — existing code that does
# `from uav.db.peregrine_db import get_aircraft` will just work.
from .local_db import (
    get_aircraft,
    update_aircraft_envelope,
    update_aircraft_pid_gains,
    increment_flight_count,
    get_airport,
    get_runway,
    list_airports,
    create_flight,
    finalize_flight,
    get_flight_history,
    get_personal_best,
    log_event,
    get_flight_events,
    save_telemetry_snapshot,
    create_learn_session,
    update_learn_session,
    get_stats,
    get_connection,
    close,
)

__all__ = [
    "get_aircraft",
    "update_aircraft_envelope",
    "update_aircraft_pid_gains",
    "increment_flight_count",
    "get_airport",
    "get_runway",
    "list_airports",
    "create_flight",
    "finalize_flight",
    "get_flight_history",
    "get_personal_best",
    "log_event",
    "get_flight_events",
    "save_telemetry_snapshot",
    "create_learn_session",
    "update_learn_session",
    "get_stats",
    "get_connection",
    "close",
]
