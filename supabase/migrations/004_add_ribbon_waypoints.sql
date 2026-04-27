-- ============================================================
-- PEREGRINE — Add ribbon_waypoints column to flights table
-- ============================================================
-- The autopilot stores a downsampled copy of the V2 ribbon
-- (lat/lon/alt/speed/phase/heading for ~60 points) in this column
-- so the app can draw the planned route on the map. The column was
-- added to local SQLite via an in-place ALTER, but was never added
-- to Supabase — which caused every flight push to fail with
-- PGRST204 ("Could not find the 'ribbon_waypoints' column"),
-- cascading into FK failures on flight_events and
-- telemetry_snapshots, which is why the app's flight history
-- never refreshes after a landing.
--
-- Run in Supabase SQL Editor.
-- ============================================================

ALTER TABLE flights ADD COLUMN IF NOT EXISTS ribbon_waypoints jsonb;
