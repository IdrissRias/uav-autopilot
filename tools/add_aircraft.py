"""Add an aircraft profile to the local database.

The autopilot is airframe-agnostic: it plans from an envelope of speeds
and refines that envelope flight by flight via the observer. A new plane
therefore needs only sane SEED numbers to fly its first sortie; the real
metrics are LEARNED, not hand-tuned (the SF50's 98.4 kt v_land was
learned from twelve flights, not typed in).

This tool inserts a preset (published book numbers as seeds) so a new
airframe flies sensibly on flight one. After a handful of flights, the
observer's learned values override the seeds automatically (see
aircraft_loader._pick: learned wins when confidence > 0.5).

Usage:
    PYTHONPATH=src python3 tools/add_aircraft.py KINGAIR
    PYTHONPATH=src python3 tools/add_aircraft.py BARON58
    PYTHONPATH=src python3 tools/add_aircraft.py CITATIONX
    PYTHONPATH=src python3 tools/add_aircraft.py --list

Then point the autopilot at it:  set airframe.icao_type in
src/uav/config/default.yaml to the icao_type printed below, and restart.
"""
from __future__ import annotations

import sqlite3
import sys
import uuid
from pathlib import Path

DB_PATH = Path.home() / ".peregrine" / "peregrine.db"

# Published book numbers, used ONLY as seeds. Every value is refined by
# the observer once the plane has flown. Cruise seeds are deliberately
# the IAS we want to plan around at pattern altitude, not the high-alt
# TAS brochure figure.
PRESETS = {
    "KINGAIR": dict(
        icao_type="BE9L", name="Beechcraft King Air C90",
        category="twin_turboprop",
        v_stall_clean=78.0, v_stall_flap=72.0, v_rotate=95.0,
        v_best_climb=120.0, v_cruise=180.0, v_approach=110.0,
        v_land=100.0, v_never_exceed=208.0,
        takeoff_roll_ft=2000.0, landing_roll_ft=2250.0,
        best_climb_fpm=2000.0, service_ceiling=30000.0,
    ),
    "BARON58": dict(
        icao_type="BE58", name="Beechcraft Baron 58",
        category="twin_piston",
        v_stall_clean=84.0, v_stall_flap=74.0, v_rotate=85.0,
        v_best_climb=105.0, v_cruise=160.0, v_approach=100.0,
        v_land=92.0, v_never_exceed=223.0,
        takeoff_roll_ft=1400.0, landing_roll_ft=1400.0,
        best_climb_fpm=1700.0, service_ceiling=20688.0,
    ),
    "CITATIONX": dict(
        icao_type="C750", name="Cessna Citation X",
        category="light_jet",
        v_stall_clean=100.0, v_stall_flap=92.0, v_rotate=120.0,
        v_best_climb=180.0, v_cruise=250.0, v_approach=125.0,
        v_land=110.0, v_never_exceed=300.0,
        takeoff_roll_ft=5140.0, landing_roll_ft=3400.0,
        best_climb_fpm=3650.0, service_ceiling=51000.0,
    ),
}


def add(key: str) -> None:
    p = PRESETS[key]
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute(
        "SELECT id FROM aircraft WHERE icao_type = ?", (p["icao_type"],)
    ).fetchone()
    cols = ("seed_v_stall_clean", "seed_v_stall_flap", "seed_v_rotate",
            "seed_v_best_climb", "seed_v_cruise", "seed_v_approach",
            "seed_v_land", "seed_v_never_exceed", "seed_takeoff_roll_ft",
            "seed_landing_roll_ft", "seed_best_climb_fpm",
            "seed_service_ceiling")
    seed_map = dict(
        seed_v_stall_clean=p["v_stall_clean"], seed_v_stall_flap=p["v_stall_flap"],
        seed_v_rotate=p["v_rotate"], seed_v_best_climb=p["v_best_climb"],
        seed_v_cruise=p["v_cruise"], seed_v_approach=p["v_approach"],
        seed_v_land=p["v_land"], seed_v_never_exceed=p["v_never_exceed"],
        seed_takeoff_roll_ft=p["takeoff_roll_ft"],
        seed_landing_roll_ft=p["landing_roll_ft"],
        seed_best_climb_fpm=p["best_climb_fpm"],
        seed_service_ceiling=p["service_ceiling"],
    )
    if existing:
        sets = ", ".join(f"{c} = ?" for c in cols)
        conn.execute(
            f"UPDATE aircraft SET name = ?, category = ?, {sets}, "
            "updated_at = datetime('now') WHERE icao_type = ?",
            (p["name"], p["category"], *[seed_map[c] for c in cols],
             p["icao_type"]),
        )
        conn.execute(
            "INSERT INTO _sync_queue (table_name, row_id, operation, "
            "created_at) VALUES ('aircraft', ?, 'update', datetime('now'))",
            (existing[0],),
        )
        print(f"Updated {p['name']} ({p['icao_type']}) seeds.")
    else:
        new_id = str(uuid.uuid4())
        collist = ", ".join(cols)
        qs = ", ".join("?" for _ in cols)
        conn.execute(
            f"INSERT INTO aircraft (id, icao_type, name, category, {collist}) "
            f"VALUES (?, ?, ?, ?, {qs})",
            (new_id, p["icao_type"], p["name"], p["category"],
             *[seed_map[c] for c in cols]),
        )
        conn.execute(
            "INSERT INTO _sync_queue (table_name, row_id, operation, "
            "created_at) VALUES ('aircraft', ?, 'insert', datetime('now'))",
            (new_id,),
        )
        print(f"Added {p['name']} ({p['icao_type']}).")
    conn.commit()
    conn.close()
    print(f"  Seeds: rotate {p['v_rotate']:.0f}  cruise {p['v_cruise']:.0f}  "
          f"approach {p['v_approach']:.0f}  land {p['v_land']:.0f} kts")
    print(f"  To fly it: set airframe.icao_type = \"{p['icao_type']}\" in "
          f"src/uav/config/default.yaml, then restart the autopilot.")
    print("  Real metrics learn themselves over the first few flights.")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("--list", "-l", "-h", "--help"):
        print("Presets:")
        for k, v in PRESETS.items():
            print(f"  {k:12} -> {v['name']} ({v['icao_type']})")
        return
    key = sys.argv[1].upper()
    if key not in PRESETS:
        print(f"Unknown preset '{key}'. Options: {', '.join(PRESETS)}")
        sys.exit(1)
    add(key)


if __name__ == "__main__":
    main()
