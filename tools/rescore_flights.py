"""One-off backfill: re-score historical flights against the aircraft's
REAL landing speed.

The scorer judged every landing against a hardcoded 65-kt reference
(Cessna-class) while the SF50's learned v_land is 98.4 — so an 88-kt
touchdown (10 kts under the aircraft's own book number) was called
"way too fast" and scored ~22. The scorer now reads the envelope
(commit b4da5a2); this script makes history consistent with it.

Only the SPEED component is recomputed — accuracy/time/stability
points were never wrong. Since score_speed is stored per flight, the
swap is exact:  new_total = old_total - old_speed_pts + new_speed_pts.

Updates local SQLite AND enqueues each flight for the Supabase sync
so the app's sparkline/history update on the autopilot's next push.

Usage:  PYTHONPATH=src python3 tools/rescore_flights.py [--dry-run]
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path.home() / ".peregrine" / "peregrine.db"


def _new_speed_pts(td_kts: float, v_land: float) -> float:
    """Mirror of FlightScorer's speed component with the envelope
    reference: full 30 pts at v_land+5, zero at v_land+35."""
    spd_over = max(0.0, td_kts - (v_land + 5.0))
    return max(0.0, 30.0 * (1.0 - spd_over / 30.0))


def main() -> None:
    dry = "--dry-run" in sys.argv
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # v_land per aircraft from the learned envelope (seed as fallback).
    v_land_by_aircraft: dict[str, float] = {}
    for a in conn.execute(
            "SELECT id, icao_type, seed_v_land, envelope FROM aircraft"):
        v_land = None
        try:
            env = json.loads(a["envelope"] or "{}")
            v_land = env.get("speeds_kts", {}).get("v_land", {}).get("value")
        except Exception:
            pass
        v_land = v_land or a["seed_v_land"]
        if v_land:
            v_land_by_aircraft[a["id"]] = float(v_land)
            print(f"aircraft {a['icao_type']}: v_land = {v_land:.1f} kts")

    rows = conn.execute("""
        SELECT id, aircraft_id, score_total, score_speed,
               touchdown_speed_kts
        FROM flights
        WHERE score_total IS NOT NULL
          AND touchdown_speed_kts IS NOT NULL
    """).fetchall()

    changed = 0
    for f in rows:
        v_land = v_land_by_aircraft.get(f["aircraft_id"])
        if v_land is None:
            continue
        # Junk guard: a stored touchdown speed below 30 kts is not a
        # landing measurement (aborted flights, zeroed rows) — awarding
        # such rows full speed points would corrupt history the other way.
        if float(f["touchdown_speed_kts"]) < 30.0:
            continue
        old_speed = float(f["score_speed"] or 0.0)
        new_speed = _new_speed_pts(float(f["touchdown_speed_kts"]), v_land)
        if abs(new_speed - old_speed) < 0.05:
            continue
        new_total = max(0.0, float(f["score_total"]) - old_speed + new_speed)
        print(f"  {f['id'][:8]}  td={f['touchdown_speed_kts']:5.1f}kts  "
              f"speed {old_speed:5.1f} → {new_speed:5.1f}  "
              f"total {f['score_total']:5.1f} → {new_total:5.1f}")
        if not dry:
            conn.execute(
                "UPDATE flights SET score_speed = ?, score_total = ?, "
                "synced_at = NULL WHERE id = ?",
                (round(new_speed, 1), round(new_total, 1), f["id"]),
            )
            conn.execute(
                "INSERT INTO _sync_queue (table_name, row_id, operation, "
                "created_at) VALUES ('flights', ?, 'update', datetime('now'))",
                (f["id"],),
            )
        changed += 1

    if not dry:
        conn.commit()
    print(f"\n{'DRY RUN — ' if dry else ''}{changed} of {len(rows)} "
          f"scored flights re-scored"
          + ("" if dry else " (updates queued for Supabase sync)"))
    conn.close()


if __name__ == "__main__":
    main()
