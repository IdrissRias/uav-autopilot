#!/usr/bin/env python3
"""
Peregrine Daemon — always-on process that watches Supabase for commands
and starts/stops the autopilot accordingly.

The Flutter app writes to the aircraft table:
  - status = 'start_autopilot'  → daemon starts the autopilot process
  - status = 'stop_autopilot'   → daemon kills the autopilot process

Run this once on the machine connected to X-Plane:
    python3 peregrine_daemon.py

It will stay running and manage the autopilot lifecycle.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

POLL_INTERVAL = 3  # seconds
AUTOPILOT_CMD = [
    sys.executable, "-B", "-u", "-m", "uav.main", "--wait-for-fly"
]
AUTOPILOT_CWD = str(Path(__file__).parent)
LOG_FILE = "/tmp/peregrine_autopilot.log"

# Aircraft ID — auto-detect or set from env
AIRCRAFT_ID = os.environ.get("PEREGRINE_AIRCRAFT_ID", None)

_autopilot_proc = None
_supabase = None


def get_supabase():
    global _supabase
    if _supabase is None:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL", "")
        key = os.environ.get("SUPABASE_ANON_KEY", "")
        if not url or not key:
            print("[DAEMON] ERROR: SUPABASE_URL and SUPABASE_ANON_KEY must be set")
            sys.exit(1)
        _supabase = create_client(url, key)
    return _supabase


def find_aircraft_id():
    """Find the aircraft ID from the database."""
    global AIRCRAFT_ID
    if AIRCRAFT_ID:
        return AIRCRAFT_ID
    try:
        sb = get_supabase()
        rows = sb.table("aircraft").select("id, name").limit(1).execute()
        if rows.data:
            AIRCRAFT_ID = rows.data[0]["id"]
            print(f"[DAEMON] Using aircraft: {rows.data[0].get('name', AIRCRAFT_ID)}")
            return AIRCRAFT_ID
    except Exception as e:
        print(f"[DAEMON] Failed to find aircraft: {e}")
    return None


def is_autopilot_running():
    global _autopilot_proc
    if _autopilot_proc is None:
        return False
    poll = _autopilot_proc.poll()
    if poll is not None:
        _autopilot_proc = None
        return False
    return True


def start_autopilot():
    global _autopilot_proc
    if is_autopilot_running():
        print("[DAEMON] Autopilot already running")
        return True

    print(f"[DAEMON] Starting autopilot → {LOG_FILE}")
    try:
        log_f = open(LOG_FILE, "w")
        _autopilot_proc = subprocess.Popen(
            AUTOPILOT_CMD,
            cwd=AUTOPILOT_CWD,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(2)
        if _autopilot_proc.poll() is not None:
            print(f"[DAEMON] Autopilot crashed on start! Check {LOG_FILE}")
            _autopilot_proc = None
            return False
        print(f"[DAEMON] Autopilot started (PID {_autopilot_proc.pid})")
        return True
    except Exception as e:
        print(f"[DAEMON] Failed to start: {e}")
        return False


def stop_autopilot():
    global _autopilot_proc
    if not is_autopilot_running():
        print("[DAEMON] Autopilot not running")
        return

    print(f"[DAEMON] Stopping autopilot (PID {_autopilot_proc.pid})")
    try:
        os.killpg(os.getpgid(_autopilot_proc.pid), signal.SIGTERM)
        _autopilot_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(_autopilot_proc.pid), signal.SIGKILL)
        _autopilot_proc.wait(timeout=3)
    except Exception as e:
        print(f"[DAEMON] Kill error: {e}")
    _autopilot_proc = None
    print("[DAEMON] Autopilot stopped")


def update_status(status):
    """Update aircraft status in the database."""
    try:
        sb = get_supabase()
        sb.table("aircraft").update({
            "status": status,
            "last_heartbeat": "now()",
        }).eq("id", AIRCRAFT_ID).execute()
    except Exception as e:
        print(f"[DAEMON] Status update failed: {e}")


def poll_loop():
    aircraft_id = find_aircraft_id()
    if not aircraft_id:
        print("[DAEMON] No aircraft found. Set PEREGRINE_AIRCRAFT_ID env var.")
        sys.exit(1)

    print(f"[DAEMON] Watching aircraft {aircraft_id}")
    print(f"[DAEMON] Poll interval: {POLL_INTERVAL}s")
    print(f"[DAEMON] Autopilot log: {LOG_FILE}")
    print("[DAEMON] Ready. Waiting for commands from app...")

    while True:
        try:
            sb = get_supabase()
            row = sb.table("aircraft").select("status").eq("id", aircraft_id).single().execute()
            status = row.data.get("status", "") if row.data else ""

            running = is_autopilot_running()

            if status == "start_autopilot":
                if not running:
                    ok = start_autopilot()
                    update_status("ground" if ok else "offline")
                else:
                    # Already running — just ack
                    update_status("ground")

            elif status == "stop_autopilot":
                stop_autopilot()
                update_status("offline")

            elif status == "end_flight_requested":
                # Let the autopilot handle this — it's already running
                pass

            elif status == "fly_requested":
                # Autopilot needs to be running to handle this
                if not running:
                    print("[DAEMON] Fly requested but autopilot not running — starting it")
                    start_autopilot()
                # Don't change status — autopilot will pick up fly_requested

            # Heartbeat — let the app know daemon is alive
            if running:
                try:
                    sb.table("aircraft").update({
                        "last_heartbeat": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }).eq("id", aircraft_id).execute()
                except Exception:
                    pass

        except Exception as e:
            print(f"[DAEMON] Poll error: {e}")

        time.sleep(POLL_INTERVAL)


def _sigterm_handler(sig, frame):
    print("\n[DAEMON] Shutting down...")
    stop_autopilot()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)
    print("=" * 50)
    print("  PEREGRINE DAEMON")
    print("  Autopilot lifecycle manager")
    print("=" * 50)
    poll_loop()
