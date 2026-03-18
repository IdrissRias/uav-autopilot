-- ============================================================
-- PEREGRINE — UAV Autopilot Database Schema
-- ============================================================
-- Run this in Supabase SQL Editor (Dashboard → SQL → New query)
-- ============================================================

-- Enable UUID generation
create extension if not exists "uuid-ossp";

-- ============================================================
-- 1. AIRCRAFT — what the plane is
-- ============================================================
create table aircraft (
  id          uuid primary key default uuid_generate_v4(),
  icao_type   text not null,                        -- e.g. "C172"
  name        text not null,                        -- e.g. "Cessna 172 Skyhawk"
  category    text not null default 'single_engine', -- single_engine, twin, jet, turboprop

  -- Seed / manual speeds (kts) — starting point before learn mode
  seed_v_stall_clean    real,
  seed_v_stall_flap     real,
  seed_v_rotate         real,
  seed_v_best_climb     real,
  seed_v_cruise         real,
  seed_v_approach       real,
  seed_v_land           real,
  seed_v_never_exceed   real,

  -- Seed performance
  seed_takeoff_roll_ft  real,
  seed_best_climb_fpm   real,
  seed_service_ceiling  real,

  -- Learned envelope (full JSON blob from learn mode)
  envelope    jsonb default '{}'::jsonb,

  -- PID gains (per aircraft — tuned by learn mode or manual)
  pid_gains   jsonb default '{}'::jsonb,

  -- Metadata
  envelope_version    int default 0,
  flights_completed   int default 0,
  avg_confidence      real default 0.0,

  created_at  timestamptz default now(),
  updated_at  timestamptz default now()
);

-- ============================================================
-- 2. AIRPORTS — where you fly to and from
-- ============================================================
create table airports (
  id          uuid primary key default uuid_generate_v4(),
  icao_code   text not null unique,                 -- e.g. "KFAR"
  name        text not null,                        -- e.g. "Hector International"
  lat         double precision not null,
  lon         double precision not null,
  elevation_ft real not null,
  country     text,
  region      text,                                 -- state/province

  created_at  timestamptz default now()
);

-- ============================================================
-- 3. RUNWAYS — each airport has 1+ runways
-- ============================================================
create table runways (
  id              uuid primary key default uuid_generate_v4(),
  airport_id      uuid not null references airports(id) on delete cascade,

  designator      text not null,                    -- e.g. "31" or "13/31"
  heading_deg     real not null,                    -- magnetic heading
  length_ft       real not null,
  width_ft        real not null,
  surface         text default 'asphalt',           -- asphalt, concrete, grass, gravel

  -- Threshold coordinates (START of runway for this designator)
  threshold_lat   double precision not null,
  threshold_lon   double precision not null,

  -- Far end coordinates (END of runway — plane must stop before here)
  end_lat         double precision not null,
  end_lon         double precision not null,

  -- Displaced threshold (if landing zone starts further in)
  displaced_ft    real default 0,

  -- Elevation at threshold
  threshold_elev_ft real,

  created_at      timestamptz default now()
);

create index idx_runways_airport on runways(airport_id);

-- ============================================================
-- 4. FLIGHTS — every flight the system performs
-- ============================================================
create table flights (
  id              uuid primary key default uuid_generate_v4(),
  aircraft_id     uuid references aircraft(id),

  -- Route
  dep_airport_id  uuid references airports(id),
  dep_runway_id   uuid references runways(id),
  arr_airport_id  uuid references airports(id),
  arr_runway_id   uuid references runways(id),

  -- Timing
  started_at      timestamptz,
  ended_at        timestamptz,
  duration_s      real,

  -- Distance
  route_distance_nm real,

  -- Flight mode
  flight_mode     text default 'normal',            -- normal, learn, training_loop, free_fly

  -- Scoring (100 pt scale)
  score_total     real,
  score_accuracy  real,       -- /40
  score_speed     real,       -- /30
  score_time      real,       -- /20
  score_stability real,       -- /10
  score_bonus     real,       -- extra points (runway usage, etc.)

  -- Landing details
  touchdown_speed_kts    real,
  touchdown_distance_ft  real,  -- from runway threshold
  touchdown_offset_m     real,  -- lateral offset from centerline
  distance_from_dest_m   real,  -- haversine to destination point
  rollout_distance_ft    real,

  -- Takeoff details
  takeoff_roll_ft        real,
  rotate_speed_kts       real,
  liftoff_speed_kts      real,

  -- Cruise details
  cruise_alt_target_ft   real,
  cruise_alt_stddev_ft   real,
  cruise_alt_max_dev_ft  real,
  cruise_speed_avg_kts   real,

  -- Phase timeline (JSON array of {phase, started_at, alt_ft, speed_kts})
  phase_timeline  jsonb default '[]'::jsonb,

  -- Telemetry log file path (for replay)
  telemetry_log_path text,

  -- Envelope contributions from this flight
  envelope_updates jsonb default '[]'::jsonb,

  -- Status
  status          text default 'in_progress',       -- in_progress, completed, aborted, crashed
  abort_reason    text,

  -- Personal best flag
  is_personal_best boolean default false,

  created_at      timestamptz default now()
);

create index idx_flights_aircraft on flights(aircraft_id);
create index idx_flights_arr_airport on flights(arr_airport_id);
create index idx_flights_status on flights(status);
create index idx_flights_created on flights(created_at desc);

-- ============================================================
-- 5. FLIGHT_EVENTS — timestamped events during a flight
-- ============================================================
create table flight_events (
  id          uuid primary key default uuid_generate_v4(),
  flight_id   uuid not null references flights(id) on delete cascade,

  event_type  text not null,                        -- phase_change, v_rotate, wheels_up,
                                                    -- flap_change, approach_initiated, touchdown,
                                                    -- full_stop, rejected_takeoff, go_around,
                                                    -- stall_warning, telemetry_stale, envelope_learned

  timestamp   timestamptz not null default now(),

  -- Snapshot at time of event
  altitude_ft     real,
  agl_ft          real,
  airspeed_kts    real,
  heading_deg     real,
  vertical_speed_fpm real,
  lat             double precision,
  lon             double precision,
  throttle        real,
  flap_pct        real,
  phase           text,

  -- Event-specific payload
  payload     jsonb default '{}'::jsonb,            -- e.g. {"from": "CLIMB", "to": "CRUISE"}
  message     text,                                 -- human-readable event description

  created_at  timestamptz default now()
);

create index idx_events_flight on flight_events(flight_id);
create index idx_events_type on flight_events(event_type);
create index idx_events_timestamp on flight_events(timestamp);

-- ============================================================
-- 6. TELEMETRY_SNAPSHOTS — periodic telemetry for replay
-- ============================================================
-- Not every tick (too much data) — every 1s or on phase change
create table telemetry_snapshots (
  id          uuid primary key default uuid_generate_v4(),
  flight_id   uuid not null references flights(id) on delete cascade,

  timestamp   timestamptz not null default now(),
  tick_num    int,

  -- Position
  lat             double precision,
  lon             double precision,
  altitude_ft     real,
  agl_ft          real,
  heading_deg     real,

  -- Motion
  airspeed_kts    real,
  groundspeed_kts real,
  vertical_speed_fpm real,
  pitch_deg       real,
  roll_deg        real,

  -- Controls
  throttle        real,
  pitch_cmd       real,
  roll_cmd        real,
  yaw_cmd         real,
  flap_ratio      real,
  gear_down       boolean,
  brake_ratio     real,

  -- State
  phase           text,
  dist_to_dest_nm real,

  created_at      timestamptz default now()
);

create index idx_telemetry_flight on telemetry_snapshots(flight_id);
create index idx_telemetry_timestamp on telemetry_snapshots(flight_id, timestamp);

-- ============================================================
-- 7. LEARN_SESSIONS — envelope discovery sessions
-- ============================================================
create table learn_sessions (
  id              uuid primary key default uuid_generate_v4(),
  aircraft_id     uuid not null references aircraft(id),
  flight_id       uuid references flights(id),

  status          text default 'in_progress',       -- in_progress, completed, aborted

  -- Which test cards were run
  cards_completed jsonb default '[]'::jsonb,         -- [{card: 0, status: "complete", ...}]
  cards_total     int default 9,

  -- Discoveries made
  discoveries     jsonb default '[]'::jsonb,         -- [{param: "v_stall_clean", value: 48, ...}]

  -- Confidence before and after
  confidence_before real,
  confidence_after  real,

  started_at      timestamptz default now(),
  ended_at        timestamptz,

  created_at      timestamptz default now()
);

create index idx_learn_aircraft on learn_sessions(aircraft_id);

-- ============================================================
-- 8. UPDATED_AT TRIGGER (reusable)
-- ============================================================
create or replace function update_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create trigger set_updated_at
  before update on aircraft
  for each row execute function update_updated_at();

-- ============================================================
-- 9. ROW LEVEL SECURITY
-- ============================================================
-- For now, allow all operations with anon key (single-user system).
-- When we add auth later, we'll tighten these.

alter table aircraft enable row level security;
alter table airports enable row level security;
alter table runways enable row level security;
alter table flights enable row level security;
alter table flight_events enable row level security;
alter table telemetry_snapshots enable row level security;
alter table learn_sessions enable row level security;

-- Permissive policies for anon (single user, no auth yet)
create policy "anon_all" on aircraft for all using (true) with check (true);
create policy "anon_all" on airports for all using (true) with check (true);
create policy "anon_all" on runways for all using (true) with check (true);
create policy "anon_all" on flights for all using (true) with check (true);
create policy "anon_all" on flight_events for all using (true) with check (true);
create policy "anon_all" on telemetry_snapshots for all using (true) with check (true);
create policy "anon_all" on learn_sessions for all using (true) with check (true);
