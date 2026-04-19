-- ============================================================
-- PEREGRINE — Add app-facing columns to aircraft table
-- ============================================================
-- Adds: tail_number, allowed_users, owner_id, heartbeat, status,
-- last position, and runway detection fields.
-- Run in Supabase SQL Editor.
-- ============================================================

-- New columns on aircraft
ALTER TABLE aircraft ADD COLUMN IF NOT EXISTS tail_number text;
ALTER TABLE aircraft ADD COLUMN IF NOT EXISTS owner_id uuid;
ALTER TABLE aircraft ADD COLUMN IF NOT EXISTS allowed_users uuid[] DEFAULT '{}';
ALTER TABLE aircraft ADD COLUMN IF NOT EXISTS status text DEFAULT 'offline';
ALTER TABLE aircraft ADD COLUMN IF NOT EXISTS last_heartbeat timestamptz;
ALTER TABLE aircraft ADD COLUMN IF NOT EXISTS last_lat double precision;
ALTER TABLE aircraft ADD COLUMN IF NOT EXISTS last_lon double precision;
ALTER TABLE aircraft ADD COLUMN IF NOT EXISTS last_heading double precision;
ALTER TABLE aircraft ADD COLUMN IF NOT EXISTS runway_icao text;
ALTER TABLE aircraft ADD COLUMN IF NOT EXISTS runway_designator text;

-- New precision fields on flights
ALTER TABLE flights ADD COLUMN IF NOT EXISTS pilot_id uuid;

-- Takeoff precision
ALTER TABLE flights ADD COLUMN IF NOT EXISTS takeoff_start_along_ft real;
ALTER TABLE flights ADD COLUMN IF NOT EXISTS takeoff_rotate_along_ft real;
ALTER TABLE flights ADD COLUMN IF NOT EXISTS takeoff_liftoff_along_ft real;
ALTER TABLE flights ADD COLUMN IF NOT EXISTS takeoff_runway_remaining_ft real;

-- Landing precision
ALTER TABLE flights ADD COLUMN IF NOT EXISTS landing_touchdown_along_ft real;
ALTER TABLE flights ADD COLUMN IF NOT EXISTS landing_touchdown_target_ft real;
ALTER TABLE flights ADD COLUMN IF NOT EXISTS landing_touchdown_error_ft real;
ALTER TABLE flights ADD COLUMN IF NOT EXISTS landing_lateral_offset_ft real;
ALTER TABLE flights ADD COLUMN IF NOT EXISTS landing_stop_along_ft real;
ALTER TABLE flights ADD COLUMN IF NOT EXISTS landing_rollout_ft real;
ALTER TABLE flights ADD COLUMN IF NOT EXISTS landing_runway_remaining_ft real;
ALTER TABLE flights ADD COLUMN IF NOT EXISTS landing_vs_fpm real;
ALTER TABLE flights ADD COLUMN IF NOT EXISTS landing_gear_down_agl_ft real;

-- Index for allowed_users array lookups
CREATE INDEX IF NOT EXISTS idx_aircraft_allowed_users ON aircraft USING gin(allowed_users);

-- Index for heartbeat queries
CREATE INDEX IF NOT EXISTS idx_aircraft_heartbeat ON aircraft(last_heartbeat);
