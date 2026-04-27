"""Peregrine — Bidirectional sync between local SQLite and Supabase.

Sync strategy:
  - PUSH: After writes to SQLite, queued operations are pushed to Supabase
    in a background thread. If Supabase is unreachable, items stay in queue.
  - PULL: On startup, pull latest aircraft envelope/PID gains from Supabase
    and merge into SQLite (most recent updated_at wins).
  - Conflict resolution: timestamp-based — most recent write wins.
  - Offline-safe: If Supabase is unreachable, everything works from SQLite.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Columns that Supabase rejected with PGRST204 ("Could not find the 'X' column").
# Populated at runtime so we stop pushing them until the process restarts
# (schema may have been migrated by then). Keyed by table name.
_SCHEMA_DRIFT_COLS: Dict[str, set] = {}

# Matches: "Could not find the 'some_col' column of 'some_table' in the schema cache"
_PGRST204_COL_RE = re.compile(
    r"Could not find the ['\"]?(?P<col>\w+)['\"]? column", re.IGNORECASE
)

# Flag: is Supabase reachable?
_supabase_available = False
_sync_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _get_supabase_client():
    """Try to get Supabase client; return None if unavailable."""
    try:
        from .client import get_client
        client = get_client()
        # Quick health check — try a lightweight query
        client.table("aircraft").select("id").limit(1).execute()
        return client
    except Exception as e:
        log.info(f"Supabase not available: {e}")
        return None


def pull_from_supabase() -> Dict[str, Any]:
    """Pull latest data from Supabase into SQLite.

    Returns a summary of what was updated.
    """
    global _supabase_available

    client = _get_supabase_client()
    if not client:
        _supabase_available = False
        return {"status": "offline", "updated": []}

    _supabase_available = True
    updated = []

    try:
        from . import local_db

        # ── Pull airports & runways FIRST (reference data) ──
        # These use Supabase UUIDs so FKs work when pushing flights back.
        remote_airports = client.table("airports").select("*, runways(*)").execute()
        for remote_apt in (remote_airports.data or []):
            icao = remote_apt.get("icao_code")
            if not icao:
                continue
            local_apt = local_db.get_airport(icao)
            if not local_apt:
                _insert_airport_from_remote(remote_apt)
                updated.append(f"airport:{icao} (new)")
            else:
                # Ensure local uses Supabase UUID (for FK consistency on push)
                _align_airport_ids(remote_apt, local_apt)

        # ── Pull aircraft envelopes & PID gains ──
        remote_aircraft = client.table("aircraft").select("*").execute()
        for remote in (remote_aircraft.data or []):
            icao = remote.get("icao_type")
            if not icao:
                continue

            local = local_db.get_aircraft(icao)
            if not local:
                _insert_aircraft_from_remote(remote)
                updated.append(f"aircraft:{icao} (new)")
                continue

            # Align UUID with Supabase
            _align_aircraft_id(remote, local)

            # Compare updated_at — most recent wins
            remote_updated = remote.get("updated_at", "")
            local_updated = local.get("updated_at", "")

            if remote_updated > local_updated:
                _update_local_aircraft(icao, remote)
                updated.append(f"aircraft:{icao} (pulled)")
            elif local_updated > remote_updated:
                updated.append(f"aircraft:{icao} (local newer, will push)")

        log.info(f"Pull complete: {len(updated)} items updated")
        return {"status": "ok", "updated": updated}

    except Exception as e:
        log.warning(f"Pull failed: {e}")
        return {"status": "error", "error": str(e), "updated": updated}


def _align_aircraft_id(remote: dict, local: dict) -> None:
    """Ensure local aircraft row uses the Supabase UUID."""
    remote_id = remote.get("id", "")
    local_id = local.get("id", "")
    if remote_id and local_id != remote_id:
        from . import local_db
        conn = local_db.get_connection()
        conn.execute("PRAGMA foreign_keys=OFF")
        try:
            # Update FK references first
            conn.execute("UPDATE flights SET aircraft_id = ? WHERE aircraft_id = ?",
                          (remote_id, local_id))
            conn.execute("UPDATE learn_sessions SET aircraft_id = ? WHERE aircraft_id = ?",
                          (remote_id, local_id))
            conn.execute("UPDATE aircraft SET id = ? WHERE id = ?", (remote_id, local_id))
            conn.commit()
        finally:
            conn.execute("PRAGMA foreign_keys=ON")
        log.info(f"Aligned aircraft ID {local_id[:8]}→{remote_id[:8]}")


def _align_airport_ids(remote_apt: dict, local_apt: dict) -> None:
    """Ensure local airport + runway rows use Supabase UUIDs."""
    from . import local_db
    conn = local_db.get_connection()

    remote_id = remote_apt.get("id", "")
    local_id = local_apt.get("id", "")

    # Temporarily disable FK checks for ID swaps
    conn.execute("PRAGMA foreign_keys=OFF")

    try:
        if remote_id and local_id != remote_id:
            conn.execute("UPDATE runways SET airport_id = ? WHERE airport_id = ?",
                          (remote_id, local_id))
            conn.execute("UPDATE flights SET dep_airport_id = ? WHERE dep_airport_id = ?",
                          (remote_id, local_id))
            conn.execute("UPDATE flights SET arr_airport_id = ? WHERE arr_airport_id = ?",
                          (remote_id, local_id))
            conn.execute("UPDATE airports SET id = ? WHERE id = ?",
                          (remote_id, local_id))
            log.info(f"Aligned airport ID {local_apt['icao_code']}: {local_id[:8]}→{remote_id[:8]}")

        # Align runway IDs too
        remote_runways = {r["designator"]: r for r in remote_apt.get("runways", [])}
        for local_rwy in local_apt.get("runways", []):
            remote_rwy = remote_runways.get(local_rwy["designator"])
            if remote_rwy and remote_rwy.get("id") and local_rwy["id"] != remote_rwy["id"]:
                conn.execute("UPDATE flights SET dep_runway_id = ? WHERE dep_runway_id = ?",
                              (remote_rwy["id"], local_rwy["id"]))
                conn.execute("UPDATE flights SET arr_runway_id = ? WHERE arr_runway_id = ?",
                              (remote_rwy["id"], local_rwy["id"]))
                conn.execute("UPDATE runways SET id = ? WHERE id = ?",
                              (remote_rwy["id"], local_rwy["id"]))

        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


def _insert_aircraft_from_remote(remote: dict) -> None:
    """Insert a new aircraft from Supabase into SQLite."""
    from . import local_db
    conn = local_db.get_connection()
    conn.execute("""
        INSERT OR IGNORE INTO aircraft (id, icao_type, name, category,
            seed_v_stall_clean, seed_v_stall_flap, seed_v_rotate,
            seed_v_best_climb, seed_v_cruise, seed_v_approach,
            seed_v_land, seed_v_never_exceed,
            seed_takeoff_roll_ft, seed_best_climb_fpm, seed_service_ceiling,
            envelope, pid_gains, envelope_version, flights_completed,
            avg_confidence, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        remote.get("id", ""), remote["icao_type"], remote.get("name", ""),
        remote.get("category", "single_engine"),
        remote.get("seed_v_stall_clean"), remote.get("seed_v_stall_flap"),
        remote.get("seed_v_rotate"), remote.get("seed_v_best_climb"),
        remote.get("seed_v_cruise"), remote.get("seed_v_approach"),
        remote.get("seed_v_land"), remote.get("seed_v_never_exceed"),
        remote.get("seed_takeoff_roll_ft"), remote.get("seed_best_climb_fpm"),
        remote.get("seed_service_ceiling"),
        json.dumps(remote.get("envelope") or {}),
        json.dumps(remote.get("pid_gains") or {}),
        remote.get("envelope_version", 0), remote.get("flights_completed", 0),
        remote.get("avg_confidence", 0.0),
        remote.get("created_at", ""), remote.get("updated_at", ""),
    ))
    conn.commit()


def _update_local_aircraft(icao_type: str, remote: dict) -> None:
    """Update local aircraft with remote data (remote is newer)."""
    from . import local_db
    conn = local_db.get_connection()
    conn.execute("""
        UPDATE aircraft SET
            envelope = ?, pid_gains = ?, envelope_version = ?,
            flights_completed = ?, avg_confidence = ?, updated_at = ?
        WHERE icao_type = ?
    """, (
        json.dumps(remote.get("envelope") or {}),
        json.dumps(remote.get("pid_gains") or {}),
        remote.get("envelope_version", 0),
        remote.get("flights_completed", 0),
        remote.get("avg_confidence", 0.0),
        remote.get("updated_at", ""),
        icao_type,
    ))
    conn.commit()


def _insert_airport_from_remote(remote_apt: dict) -> None:
    """Insert airport and its runways from Supabase."""
    from . import local_db
    conn = local_db.get_connection()

    conn.execute("""
        INSERT OR IGNORE INTO airports (id, icao_code, name, lat, lon, elevation_ft,
            country, region, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        remote_apt.get("id", ""), remote_apt["icao_code"],
        remote_apt.get("name", ""), remote_apt["lat"], remote_apt["lon"],
        remote_apt["elevation_ft"], remote_apt.get("country"),
        remote_apt.get("region"), remote_apt.get("created_at", ""),
    ))

    for rwy in remote_apt.get("runways", []):
        conn.execute("""
            INSERT OR IGNORE INTO runways (id, airport_id, designator, heading_deg,
                length_ft, width_ft, surface, threshold_lat, threshold_lon,
                end_lat, end_lon, displaced_ft, threshold_elev_ft, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rwy.get("id", ""), remote_apt.get("id", ""),
            rwy["designator"], rwy["heading_deg"], rwy["length_ft"],
            rwy["width_ft"], rwy.get("surface", "asphalt"),
            rwy["threshold_lat"], rwy["threshold_lon"],
            rwy["end_lat"], rwy["end_lon"],
            rwy.get("displaced_ft", 0), rwy.get("threshold_elev_ft"),
            rwy.get("created_at", ""),
        ))

    conn.commit()


# ── Push to Supabase ─────────────────────────────────────────

def push_pending() -> Dict[str, Any]:
    """Push all pending sync operations to Supabase.

    Returns summary of what was pushed.
    """
    client = _get_supabase_client()
    if not client:
        return {"status": "offline", "pushed": 0, "failed": 0}

    from . import local_db
    pending = local_db.get_pending_syncs(limit=200)
    pushed = 0
    failed = 0

    for item in pending:
        try:
            _push_one(client, item)
            local_db.mark_synced(item["id"])
            pushed += 1
        except Exception as e:
            log.warning(f"Push failed for {item['table_name']}/{item['row_id']}: {e}")
            local_db.mark_sync_error(item["id"], str(e))
            failed += 1

    if pushed:
        log.info(f"Pushed {pushed} items to Supabase ({failed} failed)")
    return {"status": "ok", "pushed": pushed, "failed": failed}


def _push_one(client, sync_item: dict) -> None:
    """Push a single sync item to Supabase."""
    from . import local_db
    conn = local_db.get_connection()

    table = sync_item["table_name"]
    row_id = sync_item["row_id"]
    operation = sync_item["operation"]

    # Fetch the current local row
    if table == "aircraft":
        row = conn.execute("SELECT * FROM aircraft WHERE icao_type = ?", (row_id,)).fetchone()
    else:
        row = conn.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()

    if not row:
        log.warning(f"Sync: row not found in {table} for {row_id}")
        return

    data = dict(row)

    # Remove SQLite-only columns
    data.pop("synced_at", None)

    # Parse JSON strings back to objects for Supabase
    for key in ("envelope", "pid_gains", "phase_timeline", "envelope_updates",
                "cards_completed", "discoveries", "payload"):
        if key in data and isinstance(data[key], str):
            try:
                data[key] = json.loads(data[key])
            except (json.JSONDecodeError, TypeError):
                pass

    # Convert SQLite boolean ints back to bools
    if "is_personal_best" in data:
        data["is_personal_best"] = bool(data["is_personal_best"])
    if "gear_down" in data:
        data["gear_down"] = bool(data["gear_down"])

    # Strip None values that Supabase doesn't accept as empty
    data = {k: v for k, v in data.items() if v is not None}

    # Strip columns already known to be missing from Supabase (schema drift).
    drifted = _SCHEMA_DRIFT_COLS.get(table)
    if drifted:
        for col in drifted:
            data.pop(col, None)

    def _attempt(payload: Dict[str, Any]) -> None:
        if operation == "insert":
            try:
                client.table(table).insert(payload).execute()
            except Exception:
                # Might already exist — try upsert
                client.table(table).upsert(payload).execute()
        elif operation == "update":
            if table == "aircraft":
                client.table(table).update(payload).eq("icao_type", row_id).execute()
            else:
                client.table(table).update(payload).eq("id", row_id).execute()

    # Try the push, but if Supabase rejects on PGRST204 (column missing),
    # remember the drifted column, strip it, and retry. This keeps the flight
    # record (and its child rows) flowing even when local SQLite has columns
    # that haven't been migrated to Supabase yet.
    for _ in range(4):
        try:
            _attempt(data)
            return
        except Exception as e:
            msg = str(e)
            m = _PGRST204_COL_RE.search(msg)
            if not m or "PGRST204" not in msg:
                raise
            col = m.group("col")
            if col not in data:
                raise
            log.warning(f"Sync: Supabase missing column '{col}' on '{table}' — "
                        f"stripping and retrying (add via migration to persist).")
            _SCHEMA_DRIFT_COLS.setdefault(table, set()).add(col)
            data.pop(col, None)
    # Too many drift strips in a row — surface the last error
    raise RuntimeError(f"Sync gave up after repeated schema drift on {table}")


# ── Background Sync Thread ───────────────────────────────────

def start_background_sync(interval_s: float = 30.0) -> None:
    """Start a background thread that periodically pushes pending items."""
    global _sync_thread
    if _sync_thread and _sync_thread.is_alive():
        return

    _stop_event.clear()
    _sync_thread = threading.Thread(
        target=_sync_loop,
        args=(interval_s,),
        daemon=True,
        name="peregrine-sync",
    )
    _sync_thread.start()
    log.info(f"Background sync started (every {interval_s}s)")


def stop_background_sync() -> None:
    """Stop the background sync thread."""
    _stop_event.set()
    if _sync_thread:
        _sync_thread.join(timeout=5.0)


def _sync_loop(interval_s: float) -> None:
    """Background sync loop."""
    while not _stop_event.is_set():
        try:
            push_pending()
        except Exception as e:
            log.warning(f"Sync loop error: {e}")
        _stop_event.wait(timeout=interval_s)


# ── One-shot full sync ───────────────────────────────────────

def full_sync() -> Dict[str, Any]:
    """Run a complete pull + push cycle. Call on startup."""
    pull_result = pull_from_supabase()
    push_result = push_pending()
    return {
        "pull": pull_result,
        "push": push_result,
        "supabase_available": _supabase_available,
    }
