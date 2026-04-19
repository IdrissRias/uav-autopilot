"""Peregrine — Supabase Realtime Broadcast for live telemetry + commands.

The autopilot publishes telemetry at ~5Hz through Supabase Broadcast.
The app subscribes to receive live data and can send commands back.
No database writes — pure pub/sub relay through Supabase's infrastructure.

Channels:
  - peregrine:telemetry  → autopilot publishes, app subscribes
  - peregrine:commands    → app publishes, autopilot subscribes

Works over the internet (cellular from a plane, WiFi from phone).
If Supabase is unreachable, the autopilot keeps flying — broadcast
is fire-and-forget.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from dotenv import load_dotenv

log = logging.getLogger(__name__)

# Broadcast state
_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None
_client = None
_telemetry_channel = None
_commands_channel = None
_connected = False
_command_callback: Optional[Callable[[Dict[str, Any]], None]] = None


def _get_realtime_url() -> tuple[str, str]:
    """Get Supabase Realtime WebSocket URL and anon key."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_ANON_KEY", "")

    # Convert HTTP URL to WebSocket URL
    ws_url = url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = ws_url.rstrip("/") + "/realtime/v1"

    return ws_url, key


async def _run_broadcast(command_callback: Optional[Callable] = None) -> None:
    """Async loop that maintains the Realtime connection."""
    global _client, _telemetry_channel, _commands_channel, _connected

    from realtime import AsyncRealtimeClient

    ws_url, key = _get_realtime_url()
    if not ws_url or not key:
        log.warning("Supabase credentials not found — broadcast disabled")
        return

    # Bypass HTTP proxy for WebSocket connections (proxies break WSS)
    _saved_proxies = {}
    for pvar in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if pvar in os.environ:
            _saved_proxies[pvar] = os.environ.pop(pvar)

    try:
        _client = AsyncRealtimeClient(ws_url, token=key)
        await _client.connect()
        log.info("Connected to Supabase Realtime")

        # Telemetry channel — autopilot publishes here
        _telemetry_channel = _client.channel(
            "peregrine:telemetry",
            params={"config": {"broadcast": {"self": False}}}
        )
        # Also listen for command events on the telemetry channel
        # (Flutter app sends commands here since it's already subscribed)
        if command_callback:
            def _on_telemetry_command(payload):
                try:
                    print(f"[BROADCAST] Command via telemetry channel: {payload}", flush=True)
                    if isinstance(payload, dict):
                        data = payload.get("payload", payload)
                    else:
                        data = payload
                    print(f"[BROADCAST] Calling command_callback with: {data}", flush=True)
                    command_callback(data)
                    print(f"[BROADCAST] command_callback returned OK", flush=True)
                except Exception as e:
                    print(f"[BROADCAST] Command handler ERROR: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

            _telemetry_channel.on_broadcast("command", _on_telemetry_command)

        await _telemetry_channel.subscribe()
        log.info("Telemetry channel ready")

        # Commands channel — autopilot also listens here (legacy)
        _commands_channel = _client.channel(
            "peregrine:commands",
            params={"config": {"broadcast": {"self": False}}}
        )

        if command_callback:
            def _on_command(payload):
                try:
                    print(f"[BROADCAST] Raw command payload: {payload}")
                    # Try multiple payload structures
                    if isinstance(payload, dict):
                        data = payload.get("payload", payload)
                    else:
                        data = payload
                    command_callback(data)
                except Exception as e:
                    log.warning(f"Command handler error: {e}")
                    import traceback
                    traceback.print_exc()

            _commands_channel.on_broadcast("command", _on_command)

        await _commands_channel.subscribe()
        log.info("Commands channel ready (listening)")

        _connected = True

        # Restore proxy env vars now that WSS is connected
        os.environ.update(_saved_proxies)

        # Keep the connection alive
        while True:
            await asyncio.sleep(1)

    except Exception as e:
        log.warning(f"Broadcast connection failed: {e}")
        _connected = False
        # Restore proxy env vars on failure too
        os.environ.update(_saved_proxies)


def start(command_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
    """Start the broadcast system in a background thread.

    Args:
        command_callback: Called when a command arrives from the app.
            Receives a dict like {"action": "fly", "dest": "KFFM"}.
    """
    global _loop, _thread, _command_callback
    _command_callback = command_callback

    if _thread and _thread.is_alive():
        return

    def _run():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        try:
            _loop.run_until_complete(_run_broadcast(command_callback))
        except Exception as e:
            log.warning(f"Broadcast thread error: {e}")

    _thread = threading.Thread(target=_run, daemon=True, name="peregrine-broadcast")
    _thread.start()

    # Wait briefly for connection
    for _ in range(20):  # up to 2 seconds
        if _connected:
            break
        time.sleep(0.1)

    if _connected:
        print("[PEREGRINE] Broadcast: connected (telemetry + commands)")
    else:
        print("[PEREGRINE] Broadcast: offline (will retry)")


def stop() -> None:
    """Stop the broadcast system."""
    global _client, _connected
    _connected = False
    if _loop and _client:
        try:
            asyncio.run_coroutine_threadsafe(_client.close(), _loop)
        except Exception:
            pass


def is_connected() -> bool:
    """Check if broadcast is currently connected."""
    return _connected


def publish_telemetry(data: Dict[str, Any]) -> None:
    """Publish a telemetry frame. Fire-and-forget.

    Call this from the autopilot loop (every Nth tick).
    Non-blocking — queues the send on the async loop.
    """
    if not _connected or not _telemetry_channel or not _loop:
        return

    try:
        asyncio.run_coroutine_threadsafe(
            _telemetry_channel.send_broadcast("telemetry", data),
            _loop,
        )
    except Exception:
        pass  # fire-and-forget


def publish_status(data: Dict[str, Any]) -> None:
    """Publish a status update (phase changes, pre-flight info, etc.).

    Less frequent than telemetry — only on state changes.
    """
    if not _connected or not _telemetry_channel or not _loop:
        return

    try:
        asyncio.run_coroutine_threadsafe(
            _telemetry_channel.send_broadcast("status", data),
            _loop,
        )
    except Exception:
        pass


def publish_preflight(data: Dict[str, Any]) -> None:
    """Publish pre-flight status (runway detection, ready state, etc.)."""
    if not _connected or not _telemetry_channel or not _loop:
        return

    try:
        asyncio.run_coroutine_threadsafe(
            _telemetry_channel.send_broadcast("preflight", data),
            _loop,
        )
    except Exception:
        pass


# ── Heartbeat: writes to Supabase DB so the app knows the plane is online ──
_supabase_client = None
_heartbeat_lock = threading.Lock()


def _get_supabase_client():
    """Lazy-init a Supabase client for DB writes (heartbeat, status)."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    with _heartbeat_lock:
        if _supabase_client is not None:
            return _supabase_client
        try:
            from supabase import create_client
            env_path = Path(__file__).resolve().parents[3] / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            url = os.environ.get("SUPABASE_URL", "")
            key = os.environ.get("SUPABASE_ANON_KEY", "")
            if url and key:
                _supabase_client = create_client(url, key)
        except Exception as e:
            log.warning(f"Supabase client init failed: {e}")
    return _supabase_client


def publish_heartbeat(aircraft_id: str, data: Dict[str, Any]) -> None:
    """Write a heartbeat to the aircraft table in Supabase.

    Called every ~5s from the autopilot loop. The app checks last_heartbeat
    to determine if the plane is online (< 15s ago = online).

    Args:
        aircraft_id: UUID of the aircraft row.
        data: Dict of column→value pairs to write (caller builds it).
    """
    client = _get_supabase_client()
    if client is None:
        return

    try:
        import datetime, math
        data["last_heartbeat"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        # Strip None and NaN/Inf values — JSON doesn't support them
        def _safe(v):
            if v is None:
                return False
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return False
            return True
        clean = {k: v for k, v in data.items() if _safe(v)}
        client.table("aircraft").update(clean).eq("id", aircraft_id).execute()
    except Exception as e:
        print(f"[HEARTBEAT] Write failed: {e}", flush=True)


def poll_fly_command(aircraft_id: str) -> Optional[Dict[str, Any]]:
    """Check if the app wrote a fly_requested status to the DB.

    Returns destination dict if fly was requested, None otherwise.
    Clears the status after reading to prevent re-triggering.
    """
    client = _get_supabase_client()
    if client is None:
        return None

    try:
        row = (client.table("aircraft")
               .select("status, dest_icao")
               .eq("id", aircraft_id)
               .single()
               .execute())
        data = row.data
        status = data.get("status", "") if data else ""
        if status in ("fly_requested", "preview_requested"):
            dest_icao = data.get("dest_icao", "")
            # Clear the command so we don't re-trigger
            new_status = "flying" if status == "fly_requested" else "preflight"
            client.table("aircraft").update({"status": new_status}).eq("id", aircraft_id).execute()
            # Look up destination coordinates from airports table
            dest = {"icao": dest_icao, "name": "", "lat": 0.0, "lon": 0.0,
                    "_action": "fly" if status == "fly_requested" else "preview"}
            try:
                apt = (client.table("airports")
                       .select("name, lat, lon")
                       .eq("icao_code", dest_icao)
                       .single()
                       .execute())
                if apt.data:
                    dest["name"] = apt.data["name"]
                    dest["lat"] = apt.data["lat"]
                    dest["lon"] = apt.data["lon"]
            except Exception:
                pass
            return dest
    except Exception as e:
        log.debug(f"Fly command poll failed: {e}")

    return None


def poll_aircraft_status(aircraft_id: str) -> Optional[str]:
    """Read the current status field from the aircraft row.

    Returns the status string, or None on failure.
    Clears calibrate_requested after reading to prevent re-triggering.
    """
    client = _get_supabase_client()
    if client is None:
        return None

    try:
        row = (client.table("aircraft")
               .select("status")
               .eq("id", aircraft_id)
               .single()
               .execute())
        status = row.data.get("status") if row.data else None
        if status == "calibrate_requested":
            # Clear so we don't re-trigger
            client.table("aircraft").update({"status": "calibrating"}).eq("id", aircraft_id).execute()
        return status
    except Exception as e:
        log.debug(f"Status poll failed: {e}")
        return None


def poll_end_flight(aircraft_id: str) -> bool:
    """Check if the app wrote end_flight_requested status to the DB.

    Returns True if end flight was requested. Clears the status after reading.
    """
    client = _get_supabase_client()
    if client is None:
        return False

    try:
        row = (client.table("aircraft")
               .select("status")
               .eq("id", aircraft_id)
               .single()
               .execute())
        data = row.data
        if data and data.get("status") == "end_flight_requested":
            # Clear so we don't re-trigger
            client.table("aircraft").update({"status": "ending"}).eq("id", aircraft_id).execute()
            return True
    except Exception as e:
        log.debug(f"End flight poll failed: {e}")

    return False
