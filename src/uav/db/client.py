"""Peregrine — Supabase client singleton."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client


def _load_env() -> None:
    """Load .env from project root (two levels above src/uav/db/)."""
    env_path = Path(__file__).resolve().parents[3] / ".env"
    if env_path.exists():
        load_dotenv(env_path)


@lru_cache(maxsize=1)
def get_client() -> Client:
    """Return a cached Supabase client. Reads from .env on first call."""
    _load_env()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env"
        )
    return create_client(url, key)
