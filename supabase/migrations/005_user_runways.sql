-- ============================================================
-- PEREGRINE — User-defined runways
-- ============================================================
-- Users draw custom takeoff/landing strips on the app map. These are the
-- authoritative runway definition for:
--   • Preflight checks: is the plane on the runway? room to roll?
--   • Ribbon geometry: first/last segment colinear with the runway line.
--
-- Separate from the imported `runways` table because drone strips in a
-- field have no ICAO/airport parent. Heading and length are DERIVED on
-- read from lat_start/lon_start → lat_end/lon_end, never stored, so
-- geometry cannot drift from the stored endpoints.
--
-- Run in Supabase SQL Editor after 004.
-- ============================================================

CREATE TABLE IF NOT EXISTS user_runways (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    icao            TEXT,
    lat_start       DOUBLE PRECISION NOT NULL,
    lon_start       DOUBLE PRECISION NOT NULL,
    lat_end         DOUBLE PRECISION NOT NULL,
    lon_end         DOUBLE PRECISION NOT NULL,
    width_m         DOUBLE PRECISION NOT NULL DEFAULT 20.0,
    surface         TEXT DEFAULT 'unknown',
    elevation_ft    DOUBLE PRECISION,
    notes           TEXT,
    created_by      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_runways_icao ON user_runways(icao);
CREATE INDEX IF NOT EXISTS idx_user_runways_created_by ON user_runways(created_by);

-- RLS — the Flutter app uses the anon key, so we need open read/write for
-- now. Tighten when multi-user isolation is needed (match created_by to
-- auth.uid()).
ALTER TABLE user_runways ENABLE ROW LEVEL SECURITY;

CREATE POLICY "user_runways_read"  ON user_runways FOR SELECT USING (true);
CREATE POLICY "user_runways_insert" ON user_runways FOR INSERT WITH CHECK (true);
CREATE POLICY "user_runways_update" ON user_runways FOR UPDATE USING (true);
CREATE POLICY "user_runways_delete" ON user_runways FOR DELETE USING (true);
