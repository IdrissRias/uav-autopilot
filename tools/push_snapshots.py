"""Push a flight's telemetry snapshots straight to Supabase, bypassing
the sync queue. Args: <flight_id>. Idempotent-ish: upserts by id.

Stopgap for the live path not drawing when the sync queue is backlogged.
The durable fix lives in autopilot._push_snapshots_direct().
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from uav.db.sync import _get_supabase_client  # noqa: E402

fid = sys.argv[1]
conn = sqlite3.connect(Path.home() / ".peregrine" / "peregrine.db")
conn.row_factory = sqlite3.Row
rows = [dict(r) for r in conn.execute(
    "SELECT * FROM telemetry_snapshots WHERE flight_id = ? ORDER BY tick_num",
    (fid,)).fetchall()]
conn.close()

c = _get_supabase_client()
if c is None:
    print("no client"); sys.exit(1)
sample = c.table("telemetry_snapshots").select("*").limit(1).execute().data
allowed = set(sample[0].keys()) if sample else (set(rows[0].keys()) if rows else set())
payload = [{k: v for k, v in r.items() if k in allowed and v is not None}
           for r in rows]
if payload:
    # chunk to keep requests small
    for i in range(0, len(payload), 200):
        c.table("telemetry_snapshots").upsert(payload[i:i + 200]).execute()
print(f"pushed {len(payload)} snapshots for {fid[:8]}")
