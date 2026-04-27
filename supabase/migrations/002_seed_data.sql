-- ============================================================
-- PEREGRINE — Seed Data
-- ============================================================
-- Run AFTER 001_peregrine_schema.sql
-- ============================================================

-- ============================================================
-- AIRCRAFT: Cirrus Vision SF50 (our plane in X-Plane 12)
-- ============================================================
insert into aircraft (
  icao_type, name, category,
  seed_v_stall_clean, seed_v_stall_flap, seed_v_rotate,
  seed_v_best_climb, seed_v_cruise, seed_v_approach,
  seed_v_land, seed_v_never_exceed,
  seed_takeoff_roll_ft, seed_best_climb_fpm, seed_service_ceiling,
  pid_gains
) values (
  'SF50', 'Cirrus Vision SF50', 'light_jet',
  86, 67, 90,
  160, 305, 83,
  77, 250,
  2036, 1600, 31000,
  '{
    "heading":  {"kp": 0.0035, "ki": 0.0001, "kd": 0.0022},
    "altitude": {"kp": 0.0050, "ki": 0.0000, "kd": 0.0020},
    "airspeed": {"kp": 0.0120, "ki": 0.0000, "kd": 0.0035}
  }'::jsonb
);

-- ============================================================
-- AIRPORT: KFAR — Hector International (Fargo, ND)
-- Our departure airport
-- ============================================================
insert into airports (icao_code, name, lat, lon, elevation_ft, country, region)
values ('KFAR', 'Hector International Airport', 46.92065, -96.81580, 902, 'US', 'ND');

-- KFAR Runway 18/36 (9,946 ft, primary runway)
insert into runways (
  airport_id, designator, heading_deg, length_ft, width_ft, surface,
  threshold_lat, threshold_lon, end_lat, end_lon, threshold_elev_ft
) values (
  (select id from airports where icao_code = 'KFAR'),
  '18/36', 180, 9946, 150, 'asphalt',
  46.93420, -96.81580,   -- RWY 18 threshold (north end)
  46.92065, -96.81580,   -- RWY 36 threshold (south end)
  900
);

-- KFAR Runway 9/27 (6,401 ft)
insert into runways (
  airport_id, designator, heading_deg, length_ft, width_ft, surface,
  threshold_lat, threshold_lon, end_lat, end_lon, threshold_elev_ft
) values (
  (select id from airports where icao_code = 'KFAR'),
  '9/27', 90, 6401, 150, 'asphalt',
  46.92065, -96.82580,   -- RWY 27 threshold (west end)
  46.92065, -96.80580,   -- RWY 9 threshold (east end)
  902
);

-- KFAR Runway 13/31 (6,100 ft — our usual departure runway)
insert into runways (
  airport_id, designator, heading_deg, length_ft, width_ft, surface,
  threshold_lat, threshold_lon, end_lat, end_lon, threshold_elev_ft
) values (
  (select id from airports where icao_code = 'KFAR'),
  '13/31', 130, 6100, 100, 'asphalt',
  46.92400, -96.82200,   -- RWY 31 threshold (northwest end)
  46.91600, -96.81000,   -- RWY 13 threshold (southeast end)
  900
);

-- ============================================================
-- AIRPORT: KFFM — Fergus Falls Municipal (our destination)
-- ============================================================
insert into airports (icao_code, name, lat, lon, elevation_ft, country, region)
values ('KFFM', 'Fergus Falls Municipal Airport', 46.28440, -96.15640, 1178, 'US', 'MN');

-- KFFM Runway 13/31 (4,200 ft)
insert into runways (
  airport_id, designator, heading_deg, length_ft, width_ft, surface,
  threshold_lat, threshold_lon, end_lat, end_lon, threshold_elev_ft
) values (
  (select id from airports where icao_code = 'KFFM'),
  '13/31', 130, 4200, 75, 'asphalt',
  46.28800, -96.16200,   -- RWY 31 threshold (northwest end)
  46.28100, -96.15100,   -- RWY 13 threshold (southeast end)
  1178
);

-- KFFM Runway 17/35 (3,199 ft)
insert into runways (
  airport_id, designator, heading_deg, length_ft, width_ft, surface,
  threshold_lat, threshold_lon, end_lat, end_lon, threshold_elev_ft
) values (
  (select id from airports where icao_code = 'KFFM'),
  '17/35', 170, 3199, 75, 'asphalt',
  46.28900, -96.15640,   -- RWY 35 threshold (north end)
  46.28000, -96.15640,   -- RWY 17 threshold (south end)
  1178
);

-- ============================================================
-- AIRPORT: 65MN — our old short-hop test destination
-- ============================================================
insert into airports (icao_code, name, lat, lon, elevation_ft, country, region)
values ('65MN', 'Carr Lake Airport', 46.0170, -96.3820, 1030, 'US', 'MN');

insert into runways (
  airport_id, designator, heading_deg, length_ft, width_ft, surface,
  threshold_lat, threshold_lon, end_lat, end_lon, threshold_elev_ft
) values (
  (select id from airports where icao_code = '65MN'),
  '17/35', 170, 2640, 60, 'grass',
  46.0200, -96.3820,     -- north end
  46.0140, -96.3820,     -- south end
  1030
);
