"""Peregrine — Local SQLite database.

Single-file database at ~/.peregrine/peregrine.db that mirrors the Supabase
schema. All autopilot reads/writes go here first (zero latency). Background
sync pushes changes to Supabase and pulls updates back.

Why SQLite:
  - Built into Python (no extra deps)
  - Zero config, no server process
  - Single file, easy to backup/inspect
  - Fast enough for 15 Hz reads (autopilot loop)
  - WAL mode for concurrent reads during writes
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

DB_DIR = Path.home() / ".peregrine"
DB_PATH = DB_DIR / "peregrine.db"

# Module-level connection (reused across calls)
_conn: Optional[sqlite3.Connection] = None


def _generate_uuid() -> str:
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    """Get or create the module-level SQLite connection."""
    global _conn
    if _conn is not None:
        return _conn

    DB_DIR.mkdir(parents=True, exist_ok=True)
    _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    _conn.row_factory = sqlite3.Row  # dict-like access
    _conn.execute("PRAGMA journal_mode=WAL")  # concurrent reads
    _conn.execute("PRAGMA foreign_keys=ON")
    _conn.execute("PRAGMA busy_timeout=5000")  # wait up to 5s on locks

    _ensure_schema(_conn)
    return _conn


def close():
    """Close the database connection."""
    global _conn
    if _conn:
        _conn.close()
        _conn = None


def _row_to_dict(row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
    """Convert a sqlite3.Row to a plain dict, parsing JSON columns."""
    if row is None:
        return None
    d = dict(row)
    # Parse known JSON columns
    for key in ("envelope", "pid_gains", "phase_timeline", "envelope_updates",
                "cards_completed", "discoveries", "payload"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


def _rows_to_dicts(rows: list) -> List[Dict[str, Any]]:
    return [_row_to_dict(r) for r in rows]


# ── Schema Creation ──────────────────────────────────────────

def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist. Idempotent."""
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    # Seed data if aircraft table is empty
    count = conn.execute("SELECT COUNT(*) FROM aircraft").fetchone()[0]
    if count == 0:
        _seed_data(conn)
        log.info("Seeded local database with initial data")


_SCHEMA_SQL = """
-- Aircraft
CREATE TABLE IF NOT EXISTS aircraft (
    id              TEXT PRIMARY KEY DEFAULT '',
    icao_type       TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'single_engine',

    seed_v_stall_clean    REAL,
    seed_v_stall_flap     REAL,
    seed_v_rotate         REAL,
    seed_v_best_climb     REAL,
    seed_v_cruise         REAL,
    seed_v_approach       REAL,
    seed_v_land           REAL,
    seed_v_never_exceed   REAL,

    seed_takeoff_roll_ft  REAL,
    seed_landing_roll_ft  REAL,                   -- seed braking / rollout distance
    seed_best_climb_fpm   REAL,
    seed_service_ceiling  REAL,

    envelope        TEXT DEFAULT '{}',
    pid_gains       TEXT DEFAULT '{}',

    envelope_version    INTEGER DEFAULT 0,
    flights_completed   INTEGER DEFAULT 0,
    avg_confidence      REAL DEFAULT 0.0,

    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);

-- Airports
CREATE TABLE IF NOT EXISTS airports (
    id              TEXT PRIMARY KEY DEFAULT '',
    icao_code       TEXT NOT NULL UNIQUE,
    name            TEXT NOT NULL,
    lat             REAL NOT NULL,
    lon             REAL NOT NULL,
    elevation_ft    REAL NOT NULL,
    country         TEXT,
    region          TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Runways
CREATE TABLE IF NOT EXISTS runways (
    id              TEXT PRIMARY KEY DEFAULT '',
    airport_id      TEXT NOT NULL REFERENCES airports(id) ON DELETE CASCADE,
    designator      TEXT NOT NULL,
    heading_deg     REAL NOT NULL,
    length_ft       REAL NOT NULL,
    width_ft        REAL NOT NULL,
    surface         TEXT DEFAULT 'asphalt',
    threshold_lat   REAL NOT NULL,
    threshold_lon   REAL NOT NULL,
    end_lat         REAL NOT NULL,
    end_lon         REAL NOT NULL,
    displaced_ft    REAL DEFAULT 0,
    threshold_elev_ft REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_runways_airport ON runways(airport_id);

-- User-defined runways / takeoff spots.
-- Users draw these on the app map: two points (takeoff end + far end) + a
-- width band. Ribbon geometry bolts its first/last segments exactly to the
-- line between start and end, giving the L1 follower a real centerline to
-- hug instead of drifting onto the grass.
--
-- Separate from the imported `runways` table because:
--   • Drone strips in a field have no ICAO/airport parent
--   • User may trust-weight their own drawings differently
--   • Delete-all-my-runways shouldn't touch X-Plane reference data
--
-- Heading and length are DERIVED from start/end coordinates — never stored
-- so they can't drift from the geometry.
CREATE TABLE IF NOT EXISTS user_runways (
    id              TEXT PRIMARY KEY DEFAULT '',
    name            TEXT NOT NULL,                -- "My field strip" / "KUBE 27 (custom)"
    icao            TEXT,                         -- optional link to airport (no FK — airport may not exist yet)
    lat_start       REAL NOT NULL,                -- takeoff-end threshold
    lon_start       REAL NOT NULL,
    lat_end         REAL NOT NULL,                -- far end
    lon_end         REAL NOT NULL,
    width_m         REAL NOT NULL DEFAULT 20.0,   -- on-runway band ±width/2 from centerline.
                                                  -- Note: width is ONLY the preflight "is this my runway?"
                                                  -- gate. The plane always targets the centerline during
                                                  -- the roll — L1 follower + speed-scaled rudder authority
                                                  -- self-centers even from an off-centerline start.
    surface         TEXT DEFAULT 'unknown',       -- asphalt / grass / dirt / water / unknown
    elevation_ft    REAL,                         -- optional; derived from X-Plane terrain if unset
    notes           TEXT,                         -- free-form user notes
    created_by      TEXT,                         -- user id, if known
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now')),
    synced_at       TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_runways_icao ON user_runways(icao);

-- Flights
CREATE TABLE IF NOT EXISTS flights (
    id              TEXT PRIMARY KEY DEFAULT '',
    aircraft_id     TEXT REFERENCES aircraft(id),
    dep_airport_id  TEXT REFERENCES airports(id),
    dep_runway_id   TEXT REFERENCES runways(id),
    arr_airport_id  TEXT REFERENCES airports(id),
    arr_runway_id   TEXT REFERENCES runways(id),
    started_at      TEXT,
    ended_at        TEXT,
    duration_s      REAL,
    route_distance_nm REAL,
    flight_mode     TEXT DEFAULT 'normal',
    score_total     REAL,
    score_accuracy  REAL,
    score_speed     REAL,
    score_time      REAL,
    score_stability REAL,
    score_bonus     REAL,
    touchdown_speed_kts    REAL,
    touchdown_distance_ft  REAL,
    touchdown_offset_m     REAL,
    distance_from_dest_m   REAL,
    rollout_distance_ft    REAL,
    takeoff_roll_ft        REAL,
    rotate_speed_kts       REAL,
    liftoff_speed_kts      REAL,
    cruise_alt_target_ft   REAL,
    cruise_alt_stddev_ft   REAL,
    cruise_alt_max_dev_ft  REAL,
    cruise_speed_avg_kts   REAL,
    phase_timeline  TEXT DEFAULT '[]',
    telemetry_log_path TEXT,
    envelope_updates TEXT DEFAULT '[]',
    status          TEXT DEFAULT 'in_progress',
    abort_reason    TEXT,
    is_personal_best INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now')),
    synced_at       TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_flights_aircraft ON flights(aircraft_id);
CREATE INDEX IF NOT EXISTS idx_flights_status ON flights(status);
CREATE INDEX IF NOT EXISTS idx_flights_created ON flights(created_at DESC);

-- Flight Events
CREATE TABLE IF NOT EXISTS flight_events (
    id          TEXT PRIMARY KEY DEFAULT '',
    flight_id   TEXT NOT NULL REFERENCES flights(id) ON DELETE CASCADE,
    event_type  TEXT NOT NULL,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    altitude_ft     REAL,
    agl_ft          REAL,
    airspeed_kts    REAL,
    heading_deg     REAL,
    vertical_speed_fpm REAL,
    lat             REAL,
    lon             REAL,
    throttle        REAL,
    flap_pct        REAL,
    phase           TEXT,
    payload     TEXT DEFAULT '{}',
    message     TEXT,
    created_at  TEXT DEFAULT (datetime('now')),
    synced_at   TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_flight ON flight_events(flight_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON flight_events(event_type);

-- Telemetry Snapshots
CREATE TABLE IF NOT EXISTS telemetry_snapshots (
    id          TEXT PRIMARY KEY DEFAULT '',
    flight_id   TEXT NOT NULL REFERENCES flights(id) ON DELETE CASCADE,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    tick_num    INTEGER,
    lat             REAL,
    lon             REAL,
    altitude_ft     REAL,
    agl_ft          REAL,
    heading_deg     REAL,
    airspeed_kts    REAL,
    groundspeed_kts REAL,
    vertical_speed_fpm REAL,
    pitch_deg       REAL,
    roll_deg        REAL,
    throttle        REAL,
    pitch_cmd       REAL,
    roll_cmd        REAL,
    yaw_cmd         REAL,
    flap_ratio      REAL,
    gear_down       INTEGER,
    brake_ratio     REAL,
    phase           TEXT,
    dist_to_dest_nm REAL,
    created_at      TEXT DEFAULT (datetime('now')),
    synced_at       TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_telemetry_flight ON telemetry_snapshots(flight_id);

-- Learn Sessions
CREATE TABLE IF NOT EXISTS learn_sessions (
    id              TEXT PRIMARY KEY DEFAULT '',
    aircraft_id     TEXT NOT NULL REFERENCES aircraft(id),
    flight_id       TEXT REFERENCES flights(id),
    status          TEXT DEFAULT 'in_progress',
    cards_completed TEXT DEFAULT '[]',
    cards_total     INTEGER DEFAULT 9,
    discoveries     TEXT DEFAULT '[]',
    confidence_before REAL,
    confidence_after  REAL,
    started_at      TEXT DEFAULT (datetime('now')),
    ended_at        TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    synced_at       TEXT DEFAULT NULL
);

-- Sync queue: tracks what needs to be pushed to Supabase
CREATE TABLE IF NOT EXISTS _sync_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name  TEXT NOT NULL,
    row_id      TEXT NOT NULL,
    operation   TEXT NOT NULL,   -- 'insert', 'update'
    created_at  TEXT DEFAULT (datetime('now')),
    synced_at   TEXT DEFAULT NULL,
    error       TEXT DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_sync_pending ON _sync_queue(synced_at) WHERE synced_at IS NULL;
"""


# ── Seed Data ────────────────────────────────────────────────

def _seed_data(conn: sqlite3.Connection) -> None:
    """Insert initial aircraft, airports, runways — mirrors 002_seed_data.sql."""
    now = _now_iso()

    # SF50
    sf50_id = _generate_uuid()
    conn.execute("""
        INSERT INTO aircraft (id, icao_type, name, category,
            seed_v_stall_clean, seed_v_stall_flap, seed_v_rotate,
            seed_v_best_climb, seed_v_cruise, seed_v_approach,
            seed_v_land, seed_v_never_exceed,
            seed_takeoff_roll_ft, seed_landing_roll_ft,
            seed_best_climb_fpm, seed_service_ceiling,
            pid_gains, created_at, updated_at)
        VALUES (?, 'SF50', 'Cirrus Vision SF50', 'light_jet',
            86, 67, 90, 160, 305, 83, 77, 250,
            2036, 1628, 1600, 31000, ?, ?, ?)
    """, (sf50_id, json.dumps({
        "heading": {"kp": 0.0035, "ki": 0.0001, "kd": 0.0022},
        "altitude": {"kp": 0.0050, "ki": 0.0000, "kd": 0.0020},
        "airspeed": {"kp": 0.0120, "ki": 0.0000, "kd": 0.0035},
    }), now, now))

    # KFAR
    kfar_id = _generate_uuid()
    conn.execute("""
        INSERT INTO airports (id, icao_code, name, lat, lon, elevation_ft, country, region, created_at)
        VALUES (?, 'KFAR', 'Hector International Airport', 46.92065, -96.81580, 902, 'US', 'ND', ?)
    """, (kfar_id, now))

    # KFAR runways
    for designator, heading, length, width, t_lat, t_lon, e_lat, e_lon, elev in [
        ('18/36', 180, 9946, 150, 46.93420, -96.81580, 46.92065, -96.81580, 900),
        ('9/27', 90, 6401, 150, 46.92065, -96.82580, 46.92065, -96.80580, 902),
        ('13/31', 130, 6100, 100, 46.92400, -96.82200, 46.91600, -96.81000, 900),
    ]:
        conn.execute("""
            INSERT INTO runways (id, airport_id, designator, heading_deg, length_ft, width_ft,
                surface, threshold_lat, threshold_lon, end_lat, end_lon, threshold_elev_ft, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'asphalt', ?, ?, ?, ?, ?, ?)
        """, (_generate_uuid(), kfar_id, designator, heading, length, width,
              t_lat, t_lon, e_lat, e_lon, elev, now))

    # KFFM
    kffm_id = _generate_uuid()
    conn.execute("""
        INSERT INTO airports (id, icao_code, name, lat, lon, elevation_ft, country, region, created_at)
        VALUES (?, 'KFFM', 'Fergus Falls Municipal Airport', 46.28440, -96.15640, 1178, 'US', 'MN', ?)
    """, (kffm_id, now))

    for designator, heading, length, width, t_lat, t_lon, e_lat, e_lon, elev in [
        ('13/31', 130, 4200, 75, 46.28800, -96.16200, 46.28100, -96.15100, 1178),
        ('17/35', 170, 3199, 75, 46.28900, -96.15640, 46.28000, -96.15640, 1178),
    ]:
        conn.execute("""
            INSERT INTO runways (id, airport_id, designator, heading_deg, length_ft, width_ft,
                surface, threshold_lat, threshold_lon, end_lat, end_lon, threshold_elev_ft, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'asphalt', ?, ?, ?, ?, ?, ?)
        """, (_generate_uuid(), kffm_id, designator, heading, length, width,
              t_lat, t_lon, e_lat, e_lon, elev, now))

    # 65MN
    mn65_id = _generate_uuid()
    conn.execute("""
        INSERT INTO airports (id, icao_code, name, lat, lon, elevation_ft, country, region, created_at)
        VALUES (?, '65MN', 'Carr Lake Airport', 46.0170, -96.3820, 1030, 'US', 'MN', ?)
    """, (mn65_id, now))

    conn.execute("""
        INSERT INTO runways (id, airport_id, designator, heading_deg, length_ft, width_ft,
            surface, threshold_lat, threshold_lon, end_lat, end_lon, threshold_elev_ft, created_at)
        VALUES (?, ?, '17/35', 170, 2640, 60, 'grass', 46.0200, -96.3820, 46.0140, -96.3820, 1030, ?)
    """, (_generate_uuid(), mn65_id, now))

    conn.commit()
    print(f"[PEREGRINE] Local database seeded: 1 aircraft, 3 airports, 6 runways")


# ── Aircraft Operations ──────────────────────────────────────

def get_aircraft(icao_type: str) -> Optional[Dict[str, Any]]:
    """Fetch aircraft by ICAO type code."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM aircraft WHERE icao_type = ?", (icao_type,)
    ).fetchone()
    return _row_to_dict(row)


def update_aircraft_envelope(icao_type: str, envelope_data: dict, version: int) -> None:
    """Update the learned envelope for an aircraft."""
    conn = get_connection()
    conn.execute(
        "UPDATE aircraft SET envelope = ?, envelope_version = ?, updated_at = ? WHERE icao_type = ?",
        (json.dumps(envelope_data), version, _now_iso(), icao_type)
    )
    conn.commit()
    _enqueue_sync(conn, "aircraft", icao_type, "update")


def update_aircraft_pid_gains(icao_type: str, pid_gains: dict) -> None:
    """Update PID gains for an aircraft."""
    conn = get_connection()
    conn.execute(
        "UPDATE aircraft SET pid_gains = ?, updated_at = ? WHERE icao_type = ?",
        (json.dumps(pid_gains), _now_iso(), icao_type)
    )
    conn.commit()
    _enqueue_sync(conn, "aircraft", icao_type, "update")


def increment_flight_count(icao_type: str) -> None:
    """Bump flights_completed by 1."""
    conn = get_connection()
    conn.execute(
        "UPDATE aircraft SET flights_completed = flights_completed + 1, updated_at = ? WHERE icao_type = ?",
        (_now_iso(), icao_type)
    )
    conn.commit()
    _enqueue_sync(conn, "aircraft", icao_type, "update")


# ── Airport & Runway Operations ──────────────────────────────

def get_airport(icao_code: str) -> Optional[Dict[str, Any]]:
    """Fetch airport with its runways."""
    conn = get_connection()
    airport = conn.execute(
        "SELECT * FROM airports WHERE icao_code = ?", (icao_code,)
    ).fetchone()
    if not airport:
        return None

    result = _row_to_dict(airport)
    runways = conn.execute(
        "SELECT * FROM runways WHERE airport_id = ?", (result["id"],)
    ).fetchall()
    result["runways"] = _rows_to_dicts(runways)
    return result


def get_runway(airport_icao: str, designator: str) -> Optional[Dict[str, Any]]:
    """Fetch a specific runway by airport ICAO and designator."""
    airport = get_airport(airport_icao)
    if not airport:
        return None
    for rwy in airport.get("runways", []):
        if designator in rwy.get("designator", ""):
            return rwy
    return None


def list_airports() -> List[Dict[str, Any]]:
    """List all airports."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM airports ORDER BY icao_code").fetchall()
    return _rows_to_dicts(rows)


# ── User Runways ─────────────────────────────────────────────
# CRUD for user-drawn takeoff/landing strips. These are the authoritative
# runway definition for the preflight check (is the plane on the runway?
# room to roll?) and the ribbon geometry (first/last segment colinear with
# the line between lat_start/lon_start → lat_end/lon_end).
#
# Heading and length are derived on read — stored data is start + end +
# width only, so geometry stays internally consistent.

def _derive_runway_geometry(runway: Dict[str, Any]) -> Dict[str, Any]:
    """Attach derived fields: heading_deg (start→end), length_m, length_ft.

    Called by every read path so callers can rely on these fields without
    us having to persist denormalized values that could drift from the
    stored endpoints.
    """
    from uav.nav.geo import bearing_deg, haversine_m
    lat1, lon1 = runway["lat_start"], runway["lon_start"]
    lat2, lon2 = runway["lat_end"], runway["lon_end"]
    runway["heading_deg"] = bearing_deg(lat1, lon1, lat2, lon2)
    runway["length_m"] = haversine_m(lat1, lon1, lat2, lon2)
    runway["length_ft"] = runway["length_m"] * 3.28084
    return runway


def create_user_runway(
    name: str,
    lat_start: float,
    lon_start: float,
    lat_end: float,
    lon_end: float,
    width_m: float = 20.0,
    icao: Optional[str] = None,
    surface: str = "unknown",
    elevation_ft: Optional[float] = None,
    notes: Optional[str] = None,
    created_by: Optional[str] = None,
) -> Dict[str, Any]:
    """Insert a new user-drawn runway, enqueue for Supabase sync, return the row."""
    conn = get_connection()
    runway_id = _generate_uuid()
    now = _now_iso()
    conn.execute(
        """INSERT INTO user_runways
           (id, name, icao, lat_start, lon_start, lat_end, lon_end,
            width_m, surface, elevation_ft, notes, created_by,
            created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (runway_id, name, icao, lat_start, lon_start, lat_end, lon_end,
         width_m, surface, elevation_ft, notes, created_by, now, now),
    )
    conn.commit()
    _enqueue_sync(conn, "user_runways", runway_id, "insert")
    return get_user_runway(runway_id) or {"id": runway_id}


def get_user_runway(runway_id: str) -> Optional[Dict[str, Any]]:
    """Fetch one user runway by id, with derived heading/length."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM user_runways WHERE id = ?", (runway_id,)
    ).fetchone()
    d = _row_to_dict(row)
    return _derive_runway_geometry(d) if d else None


def list_user_runways(icao: Optional[str] = None) -> List[Dict[str, Any]]:
    """List user runways, optionally filtered by ICAO.

    Returns each row with derived heading_deg, length_m, length_ft.
    """
    conn = get_connection()
    if icao:
        rows = conn.execute(
            "SELECT * FROM user_runways WHERE icao = ? ORDER BY name", (icao,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM user_runways ORDER BY created_at DESC"
        ).fetchall()
    return [_derive_runway_geometry(d) for d in _rows_to_dicts(rows) if d]


def update_user_runway(runway_id: str, **fields) -> Optional[Dict[str, Any]]:
    """Update allowed fields on a user runway. Unknown keys are ignored."""
    allowed = {
        "name", "icao", "lat_start", "lon_start", "lat_end", "lon_end",
        "width_m", "surface", "elevation_ft", "notes",
    }
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return get_user_runway(runway_id)

    updates["updated_at"] = _now_iso()
    conn = get_connection()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [runway_id]
    conn.execute(f"UPDATE user_runways SET {set_clause} WHERE id = ?", values)
    conn.commit()
    _enqueue_sync(conn, "user_runways", runway_id, "update")
    return get_user_runway(runway_id)


def delete_user_runway(runway_id: str) -> bool:
    """Delete a user runway. Returns True if a row was removed.

    Note: we don't enqueue a "delete" sync op because _push_one only
    understands insert/update — remote rows linger. Fine for now; users
    editing is the common case, deletion is rare. Can revisit if needed.
    """
    conn = get_connection()
    cur = conn.execute("DELETE FROM user_runways WHERE id = ?", (runway_id,))
    conn.commit()
    return cur.rowcount > 0


# ── Flight Operations ────────────────────────────────────────

def create_flight(
    aircraft_id: str | None = None,
    dep_airport_icao: str | None = None,
    dep_runway_designator: str | None = None,
    arr_airport_icao: str | None = None,
    arr_runway_designator: str | None = None,
    flight_mode: str = "normal",
    route_distance_nm: float | None = None,
    cruise_alt_target_ft: float | None = None,
) -> Dict[str, Any]:
    """Create a new flight record."""
    conn = get_connection()
    flight_id = _generate_uuid()
    now = _now_iso()

    dep_airport_id = dep_runway_id = arr_airport_id = arr_runway_id = None

    if dep_airport_icao:
        dep = get_airport(dep_airport_icao)
        if dep:
            dep_airport_id = dep["id"]
            if dep_runway_designator:
                for rwy in dep.get("runways", []):
                    if dep_runway_designator in rwy.get("designator", ""):
                        dep_runway_id = rwy["id"]
                        break

    if arr_airport_icao:
        arr = get_airport(arr_airport_icao)
        if arr:
            arr_airport_id = arr["id"]
            if arr_runway_designator:
                for rwy in arr.get("runways", []):
                    if arr_runway_designator in rwy.get("designator", ""):
                        arr_runway_id = rwy["id"]
                        break

    conn.execute("""
        INSERT INTO flights (id, aircraft_id, dep_airport_id, dep_runway_id,
            arr_airport_id, arr_runway_id, flight_mode, status, started_at,
            route_distance_nm, cruise_alt_target_ft, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'in_progress', ?, ?, ?, ?)
    """, (flight_id, aircraft_id, dep_airport_id, dep_runway_id,
          arr_airport_id, arr_runway_id, flight_mode, now,
          route_distance_nm, cruise_alt_target_ft, now))
    conn.commit()
    _enqueue_sync(conn, "flights", flight_id, "insert")

    return {"id": flight_id, "status": "in_progress", "started_at": now}


def finalize_flight(flight_id: str, **kwargs) -> Dict[str, Any]:
    """Update a flight with final results."""
    conn = get_connection()
    kwargs["ended_at"] = _now_iso()

    # Filter None values
    updates = {k: v for k, v in kwargs.items() if v is not None}

    # Serialize JSON fields
    for key in ("phase_timeline", "envelope_updates"):
        if key in updates and not isinstance(updates[key], str):
            updates[key] = json.dumps(updates[key])

    if not updates:
        return {}

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [flight_id]
    conn.execute(f"UPDATE flights SET {set_clause} WHERE id = ?", values)
    conn.commit()
    _enqueue_sync(conn, "flights", flight_id, "update")

    row = conn.execute("SELECT * FROM flights WHERE id = ?", (flight_id,)).fetchone()
    return _row_to_dict(row) or updates


def get_flight_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent flights, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM flights ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return _rows_to_dicts(rows)


def get_personal_best(arr_airport_icao: str) -> Optional[Dict[str, Any]]:
    """Get the highest-scoring completed flight to a destination."""
    airport = get_airport(arr_airport_icao)
    if not airport:
        return None
    conn = get_connection()
    row = conn.execute("""
        SELECT * FROM flights
        WHERE arr_airport_id = ? AND status = 'completed'
        ORDER BY score_total DESC LIMIT 1
    """, (airport["id"],)).fetchone()
    return _row_to_dict(row)


# ── Flight Events ────────────────────────────────────────────

def log_event(
    flight_id: str,
    event_type: str,
    **kwargs,
) -> None:
    """Insert a flight event. Fire-and-forget."""
    conn = get_connection()
    event_id = _generate_uuid()
    now = _now_iso()

    # Serialize payload
    if "payload" in kwargs and not isinstance(kwargs.get("payload"), str):
        kwargs["payload"] = json.dumps(kwargs["payload"])

    columns = ["id", "flight_id", "event_type", "timestamp", "created_at"]
    values = [event_id, flight_id, event_type, now, now]

    for key in ("message", "altitude_ft", "agl_ft", "airspeed_kts", "heading_deg",
                "vertical_speed_fpm", "lat", "lon", "throttle", "flap_pct",
                "phase", "payload"):
        if key in kwargs and kwargs[key] is not None:
            columns.append(key)
            values.append(kwargs[key])

    placeholders = ", ".join("?" * len(columns))
    col_str = ", ".join(columns)

    try:
        conn.execute(f"INSERT INTO flight_events ({col_str}) VALUES ({placeholders})", values)
        conn.commit()
        _enqueue_sync(conn, "flight_events", event_id, "insert")
    except Exception as e:
        log.warning("Failed to log event %s: %s", event_type, e)


def get_flight_events(flight_id: str) -> List[Dict[str, Any]]:
    """Get all events for a flight."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM flight_events WHERE flight_id = ? ORDER BY timestamp",
        (flight_id,)
    ).fetchall()
    return _rows_to_dicts(rows)


# ── Telemetry Snapshots ──────────────────────────────────────

def save_telemetry_snapshot(flight_id: str, snapshot: dict) -> None:
    """Insert a telemetry snapshot. Fire-and-forget."""
    conn = get_connection()
    snap_id = _generate_uuid()
    now = _now_iso()

    columns = ["id", "flight_id", "timestamp", "created_at"]
    values = [snap_id, flight_id, now, now]

    for key in ("tick_num", "lat", "lon", "altitude_ft", "agl_ft", "heading_deg",
                "airspeed_kts", "groundspeed_kts", "vertical_speed_fpm",
                "pitch_deg", "roll_deg", "throttle", "pitch_cmd", "roll_cmd",
                "yaw_cmd", "flap_ratio", "gear_down", "brake_ratio",
                "phase", "dist_to_dest_nm"):
        if key in snapshot and snapshot[key] is not None:
            columns.append(key)
            val = snapshot[key]
            if key == "gear_down":
                val = 1 if val else 0
            values.append(val)

    placeholders = ", ".join("?" * len(columns))
    col_str = ", ".join(columns)

    try:
        conn.execute(f"INSERT INTO telemetry_snapshots ({col_str}) VALUES ({placeholders})", values)
        conn.commit()
        # Sync every 5th snapshot to avoid overwhelming the queue
        tick = snapshot.get("tick_num", 0)
        if tick % 5 == 0:
            _enqueue_sync(conn, "telemetry_snapshots", snap_id, "insert")
    except Exception as e:
        log.warning("Failed to save telemetry snapshot: %s", e)


# ── Learn Sessions ───────────────────────────────────────────

def create_learn_session(aircraft_id: str, flight_id: str | None = None) -> Dict[str, Any]:
    """Start a new learn session."""
    conn = get_connection()
    session_id = _generate_uuid()
    now = _now_iso()
    conn.execute("""
        INSERT INTO learn_sessions (id, aircraft_id, flight_id, status, started_at, created_at)
        VALUES (?, ?, ?, 'in_progress', ?, ?)
    """, (session_id, aircraft_id, flight_id, now, now))
    conn.commit()
    _enqueue_sync(conn, "learn_sessions", session_id, "insert")
    return {"id": session_id, "status": "in_progress"}


def update_learn_session(session_id: str, **kwargs) -> None:
    """Update a learn session."""
    conn = get_connection()
    updates = {}
    for key in ("status", "cards_completed", "discoveries", "confidence_after"):
        if key in kwargs and kwargs[key] is not None:
            val = kwargs[key]
            if key in ("cards_completed", "discoveries") and not isinstance(val, str):
                val = json.dumps(val)
            updates[key] = val

    status = kwargs.get("status")
    if status in ("completed", "aborted"):
        updates["ended_at"] = _now_iso()

    if updates:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [session_id]
        conn.execute(f"UPDATE learn_sessions SET {set_clause} WHERE id = ?", values)
        conn.commit()
        _enqueue_sync(conn, "learn_sessions", session_id, "update")


# ── Sync Queue ───────────────────────────────────────────────

def _enqueue_sync(conn: sqlite3.Connection, table_name: str, row_id: str, operation: str) -> None:
    """Add a pending sync operation to the queue."""
    try:
        conn.execute(
            "INSERT INTO _sync_queue (table_name, row_id, operation, created_at) VALUES (?, ?, ?, ?)",
            (table_name, row_id, operation, _now_iso())
        )
        conn.commit()
    except Exception as e:
        log.warning("Failed to enqueue sync: %s", e)


def get_pending_syncs(limit: int = 100) -> List[Dict[str, Any]]:
    """Get pending sync operations."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM _sync_queue WHERE synced_at IS NULL ORDER BY created_at LIMIT ?",
        (limit,)
    ).fetchall()
    return _rows_to_dicts(rows)


def mark_synced(sync_id: int) -> None:
    """Mark a sync operation as completed."""
    conn = get_connection()
    conn.execute(
        "UPDATE _sync_queue SET synced_at = ? WHERE id = ?",
        (_now_iso(), sync_id)
    )
    conn.commit()


def mark_sync_error(sync_id: int, error: str) -> None:
    """Mark a sync operation as failed."""
    conn = get_connection()
    conn.execute(
        "UPDATE _sync_queue SET error = ? WHERE id = ?",
        (error, sync_id)
    )
    conn.commit()


# ── Utility ──────────────────────────────────────────────────

def get_stats() -> Dict[str, Any]:
    """Get database statistics."""
    conn = get_connection()
    stats = {}
    for table in ("aircraft", "airports", "runways", "flights", "flight_events",
                   "telemetry_snapshots", "learn_sessions"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        stats[table] = count

    pending = conn.execute(
        "SELECT COUNT(*) FROM _sync_queue WHERE synced_at IS NULL"
    ).fetchone()[0]
    stats["pending_syncs"] = pending

    return stats
