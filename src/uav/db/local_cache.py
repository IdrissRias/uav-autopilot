"""Peregrine — Local JSON cache for aircraft envelopes.

Fast local-first reads. Supabase syncs in the background after flights.
If Supabase is unreachable, the local cache (or YAML fallback) keeps us flying.

Cache file: ~/.peregrine/aircraft_cache.json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".peregrine"
CACHE_FILE = CACHE_DIR / "aircraft_cache.json"


def _ensure_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def read_cache(icao_type: str) -> Optional[Dict[str, Any]]:
    """Read aircraft envelope from local cache. Returns None if not cached."""
    if not CACHE_FILE.exists():
        return None
    try:
        data = json.loads(CACHE_FILE.read_text())
        return data.get(icao_type)
    except Exception as e:
        log.warning(f"Cache read failed: {e}")
        return None


def write_cache(icao_type: str, envelope_data: Dict[str, Any]) -> None:
    """Write/update aircraft envelope in local cache."""
    _ensure_dir()
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text())
        else:
            data = {}
        data[icao_type] = envelope_data
        CACHE_FILE.write_text(json.dumps(data, indent=2))
        log.info(f"Cache updated for {icao_type}")
    except Exception as e:
        log.warning(f"Cache write failed: {e}")


def envelope_to_dict(envelope) -> Dict[str, Any]:
    """Convert AircraftEnvelope dataclass to a dict for caching."""
    from dataclasses import asdict
    return asdict(envelope)


def dict_to_envelope(d: Dict[str, Any]):
    """Convert cached dict back to AircraftEnvelope."""
    from .aircraft_loader import AircraftEnvelope
    # Filter to only keys that AircraftEnvelope accepts
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(AircraftEnvelope)}
    filtered = {k: v for k, v in d.items() if k in valid_keys}
    return AircraftEnvelope(**filtered)
