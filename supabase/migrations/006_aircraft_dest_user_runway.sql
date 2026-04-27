-- ============================================================
-- PEREGRINE — Aircraft destination: user-drawn runway
-- ============================================================
-- Adds a nullable FK from aircraft → user_runways so the preflight screen
-- can target a user-drawn strip (e.g. a grass field with no ICAO) as the
-- flight destination.
--
-- Resolution rule for the autopilot (implement in ribbon builder):
--   1. If dest_user_runway_id IS NOT NULL → use lat_start/lon_start/lat_end/
--      lon_end from the referenced user_runways row as the runway geometry.
--   2. Otherwise fall back to dest_icao → airports/runways tables.
--
-- This is additive and safe: existing rows keep dest_icao, new rows can
-- populate either column (or both — user_runway wins by rule #1).
--
-- Run in Supabase SQL Editor after 005.
-- ============================================================

ALTER TABLE aircraft
    ADD COLUMN IF NOT EXISTS dest_user_runway_id UUID
    REFERENCES user_runways(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_aircraft_dest_user_runway
    ON aircraft(dest_user_runway_id);
