from __future__ import annotations

import math
import time
from typing import Optional

from uav.sim.types import Telemetry, Targets


def _wrap_deg(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def _centerline_correction(lat: float, lon: float, ref_lat: float, ref_lon: float, rwy_hdg_deg: float) -> float:
    """Compute heading correction to track a runway centerline.

    Returns a heading offset in degrees (positive = steer right) that
    corrects for lateral deviation from the line defined by (ref_lat, ref_lon)
    along rwy_hdg_deg. Capped at ±8°.
    """
    import math as _m
    rwy_hdg_rad = _m.radians(rwy_hdg_deg)
    dlat = lat - ref_lat
    dlon = (lon - ref_lon) * _m.cos(_m.radians(ref_lat))
    dlat_m = dlat * 111320.0
    dlon_m = dlon * 111320.0
    # Cross-track: positive = right of centerline
    cl_east = _m.sin(rwy_hdg_rad)
    cl_north = _m.cos(rwy_hdg_rad)
    cross_track_m = dlon_m * cl_north - dlat_m * cl_east
    # 1° correction per 10m offset during ground ops (tighter than approach)
    return max(-8.0, min(8.0, cross_track_m / 10.0))


def _cross_track_m(lat: float, lon: float, ref_lat: float, ref_lon: float, rwy_hdg_deg: float) -> float:
    """Raw cross-track distance in meters (positive = right of centerline)."""
    rwy_hdg_rad = math.radians(rwy_hdg_deg)
    dlat = lat - ref_lat
    dlon = (lon - ref_lon) * math.cos(math.radians(ref_lat))
    dlat_m = dlat * 111320.0
    dlon_m = dlon * 111320.0
    return dlon_m * math.cos(rwy_hdg_rad) - dlat_m * math.sin(rwy_hdg_rad)


# ── Precision Approach Path ─────────────────────────────────────────
# A sequence of "gates" along the extended runway centerline.
# Each gate: (distance_nm_from_threshold, altitude_agl_ft, speed_kts)
# The plane must fly through each gate in order.

def _build_approach_path(
    threshold_lat: float,
    threshold_lon: float,
    rwy_hdg: float,
    rwy_elev_ft: float,
    entry_alt_ft: float,       # MSL altitude when approach starts
    v_approach: float,
    v_land: float,
) -> list:
    """Build precision approach gates along the extended runway centerline.

    Returns a list of dicts, each with:
      lat, lon, alt_ft (MSL), speed_kts, name, agl_ft
    Ordered from farthest to nearest (plane flies through them in order).
    """
    from uav.nav.geo import destination_point

    entry_agl = entry_alt_ft - rwy_elev_ft
    # Reciprocal heading (gates are BEHIND the threshold along approach)
    recip_hdg = (rwy_hdg + 180.0) % 360.0
    NM_TO_M = 1852.0

    # Build gates from threshold outward, then reverse so plane flies
    # from farthest to nearest.
    # Standard 3° glideslope: ~318ft per nm. We use ~320.
    # But we also need to handle high entries — add gates dynamically.
    gates = []

    # Fixed gate distances (nm from threshold)
    # Each gate: (nm, target_agl, speed)
    fixed_gates = [
        (0.3,    50.0,  v_land),          # flare handoff
        (0.5,   100.0,  v_land + 5),
        (1.0,   300.0,  v_approach),
        (1.5,   480.0,  v_approach + 5),
        (2.0,   640.0,  v_approach + 10),
        (3.0,   960.0,  v_approach + 15),
        (4.0,  1280.0,  v_approach + 20),
        (5.0,  1600.0,  v_approach + 25),
        (6.0,  1920.0,  v_approach + 30),
        (8.0,  2560.0,  v_approach + 40),
        (10.0, 3200.0,  v_approach + 50),
    ]

    for nm, agl, spd in fixed_gates:
        if agl > entry_agl + 200:
            continue  # skip gates above our entry altitude
        alt_msl = rwy_elev_ft + agl
        lat, lon = destination_point(
            threshold_lat, threshold_lon,
            nm * NM_TO_M, recip_hdg,
        )
        gates.append({
            "name": f"APP_{nm:.1f}NM",
            "lat": lat,
            "lon": lon,
            "alt_ft": alt_msl,
            "agl_ft": agl,
            "speed_kts": spd,
            "dist_nm": nm,
        })

    # Sort farthest first (plane flies through these in order)
    gates.sort(key=lambda g: -g["dist_nm"])

    return gates


class ReactiveFlightDirector:
    """
    Situation-aware flight director.

    Replaces ModeManager + per-mode classes with a single step() that decides
    what to do based on current flight state every tick.  No named transitions,
    no missed windows, no "stuck-in-mode" bugs.

    Interface is drop-in compatible with ModeManager:
      .name   — current phase label (string)
      .ctx    — shared context dict
      .step() — returns Targets
      .reset()— called after a flight reset
    """

    _MAX_TURN_DEG_PER_STEP = 1.5  # heading advance per loop tick — reduced for smoother turns

    def __init__(self, ctx: dict) -> None:
        self.ctx = ctx
        self._phase = "GROUND"
        self._st: dict = {}  # small persistent sub-state (heading ramps, timers)

    # ------------------------------------------------------------------
    # Public interface (matches ModeManager)
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._phase

    def reset(self) -> None:
        """Return to ground state after a flight reset."""
        self._phase = "GROUND"
        self._st.clear()
        self.ctx.pop("mode_state", None)
        self.ctx.pop("destination", None)
        # approach_initiated lives in _st which was just cleared — nothing extra needed

    def step(self, telemetry: Telemetry, stale: bool = False) -> Targets:
        if stale:
            self._phase = "ABORT"
            return Targets(
                heading_deg=telemetry.heading_deg,
                altitude_ft=telemetry.altitude_ft,
                airspeed_kts=0.0,
                throttle=0.0,
                brake_ratio=1.0,
                gear_down=True,
            )

        cfg = self.ctx
        airframe = cfg.get("airframe", {})
        speeds = airframe.get("speeds_kts", {})
        throttle_cfg = airframe.get("throttle", {})
        thresholds = airframe.get("thresholds", {})
        takeoff_cfg = cfg.get("takeoff", {})
        climb_cfg = cfg.get("climb", {})
        mode_cfg = cfg.get("mode", {})

        v_rotate = float(speeds.get("v_rotate", thresholds.get("takeoff_to_climb_speed_kts", 55.0)))
        v_climb = float(speeds.get("v_climb", 75.0))
        v_cruise = float(speeds.get("v_cruise", 105.0))
        v_approach = float(speeds.get("v_approach", 70.0))
        v_land = float(speeds.get("v_land", 60.0))
        v_stall = float(speeds.get("v_stall", 48.0))

        # Safety: approach/land speeds must be above stall with margin
        v_approach = max(v_approach, v_stall * 1.3)
        v_land = max(v_land, v_stall * 1.15)

        cruise_target_ft = float(cfg["targets"]["target_alt_ft"])
        agl_ft = (telemetry.agl_m * 3.28084) if not math.isnan(telemetry.agl_m) else 0.0

        # ── Auto-start guard ──────────────────────────────────────────────────
        # Don't begin the flight sequence unless requested.
        auto_start = bool(mode_cfg.get("auto_start", True))
        has_dest = cfg.get("destination") is not None
        demo = bool(mode_cfg.get("demo_sequence", False))
        if not auto_start or (not has_dest and not demo):
            self._phase = "GROUND"
            return self._ground_idle(telemetry)

        # ── Destination geometry ──────────────────────────────────────────────
        dist_nm: float = math.inf
        true_bearing: Optional[float] = None
        dest = cfg.get("destination")
        if dest and telemetry.has_position():
            try:
                from uav.nav.geo import haversine_m, bearing_deg
                dist_nm = haversine_m(
                    telemetry.lat_deg, telemetry.lon_deg,
                    float(dest["lat"]), float(dest["lon"]),
                ) / 1852.0
                true_bearing = bearing_deg(
                    telemetry.lat_deg, telemetry.lon_deg,
                    float(dest["lat"]), float(dest["lon"]),
                )
            except Exception:
                pass

        # ── Flight plan (GPS waypoints) ─────────────────────────────────────
        # If a flight plan exists, use it for waypoint-based navigation.
        # The active waypoint overrides bearing/distance/speed targets.
        flight_plan = cfg.get("flight_plan")
        nav_bearing: Optional[float] = true_bearing
        nav_dist_nm: float = dist_nm
        wp_phase: Optional[str] = None

        if flight_plan and not flight_plan.completed and telemetry.has_position():
            from uav.nav.geo import haversine_m as _hm2, bearing_deg as _brg2
            wp = flight_plan.active
            if wp:
                wp_dist_nm = _hm2(
                    telemetry.lat_deg, telemetry.lon_deg,
                    wp.lat, wp.lon,
                ) / 1852.0
                wp_bearing = _brg2(
                    telemetry.lat_deg, telemetry.lon_deg,
                    wp.lat, wp.lon,
                )
                wp_phase = wp.phase

                # Advance to next waypoint when close enough
                if wp.phase in ("CRUISE", "CLIMB"):
                    capture_nm = 0.3
                elif wp.phase == "APPROACH":
                    capture_nm = 1.0  # generous — start approach early
                else:
                    capture_nm = 0.15
                if wp.fly_over:
                    capture_nm = 0.08

                if wp_dist_nm < capture_nm:
                    # Mark if we just captured an APPROACH waypoint
                    if wp.phase == "APPROACH":
                        self._st["approach_wp_captured"] = True
                        print(f"[GPS] APPROACH waypoint captured: {wp.name}")
                    next_wp = flight_plan.advance()
                    if next_wp:
                        print(f"[GPS] Waypoint reached: {wp.name} → next: {next_wp.name}")
                        wp = next_wp
                        wp_dist_nm = _hm2(
                            telemetry.lat_deg, telemetry.lon_deg,
                            wp.lat, wp.lon,
                        ) / 1852.0
                        wp_bearing = _brg2(
                            telemetry.lat_deg, telemetry.lon_deg,
                            wp.lat, wp.lon,
                        )
                        wp_phase = wp.phase
                    else:
                        print(f"[GPS] Final waypoint reached: {wp.name}")

                # Use waypoint data for navigation
                nav_bearing = wp_bearing
                nav_dist_nm = wp_dist_nm

        # ── Approach trigger ────────────────────────────────────────────────
        # The plane must PHYSICALLY FLY TO the APPROACH_ENTRY waypoint before
        # approach can begin. This is the single, definitive gate:
        #
        #   approach_wp_captured = True  ← set when the plane flies within
        #     capture radius of the APPROACH_ENTRY waypoint in the flight plan.
        #
        # This means: no matter what wp_phase says, no matter what distance
        # we are from the airport, approach does NOT start until the plane
        # has actually arrived at the approach entry point on the map.
        #
        # Fallback for flights without a flight plan: distance-based.
        if self._phase == "CRUISE":
            cruise_ticks = self._st.get("cruise_ticks", 0) + 1
            self._st["cruise_ticks"] = cruise_ticks
            if cruise_ticks >= 30:
                self._st["has_cruised"] = True
        at_cruise = telemetry.altitude_ft >= cruise_target_ft - 250.0
        has_cruised = self._st.get("has_cruised", False)

        if not self._st.get("approach_initiated", False):
            approach_wp_captured = self._st.get("approach_wp_captured", False)

            if approach_wp_captured:
                # The plane physically reached the APPROACH_ENTRY waypoint.
                # THIS is the only way approach triggers with a flight plan.
                self._st["approach_initiated"] = True
                print(f"[APPROACH] ENTRY — approach waypoint captured: "
                      f"alt={telemetry.altitude_ft:.0f}ft dist={dist_nm:.1f}nm")
            elif flight_plan is None or flight_plan.completed:
                # Distance-based fallback (no flight plan)
                alt_to_lose = max(0.0, telemetry.altitude_ft - (dest.get("elevation_ft", 0.0) if dest else 0.0))
                approach_entry_nm = (alt_to_lose / 700.0) + 1.0
                approach_entry_nm = max(3.0, min(10.0, approach_entry_nm))
                if has_cruised and dist_nm <= approach_entry_nm:
                    self._st["approach_initiated"] = True
                    print(f"[APPROACH] ENTRY via distance: alt={telemetry.altitude_ft:.0f}ft "
                          f"dist={dist_nm:.1f}nm gate={approach_entry_nm:.1f}nm")

        near_dest = self._st.get("approach_initiated", False)

        # ── Phase selection ───────────────────────────────────────────────────
        on_ground = agl_ft < 5.0
        # LAND/FLARE starts at 50ft AGL — gives enough time to arrest descent.
        # Must also be near the destination and not screaming fast.
        flaring = agl_ft < 50.0 and near_dest and telemetry.airspeed_kts < v_approach + 30.0

        # Sticky liftoff: once we've entered CLIMB, don't bounce back to GROUND
        # unless we're actually landed (low speed). This prevents GROUND↔CLIMB
        # oscillation during liftoff when AGL fluctuates around the threshold.
        was_airborne = self._phase in ("CLIMB", "CRUISE", "APPROACH")
        if was_airborne and on_ground and telemetry.airspeed_kts > 50.0:
            on_ground = False  # still flying — ignore ground blip

        if flaring:
            phase = "LAND"
        elif on_ground:
            phase = "GROUND"
        elif not at_cruise and not near_dest:
            phase = "CLIMB"
        elif near_dest:
            phase = "APPROACH"
        else:
            phase = "CRUISE"

        # Capture runway heading on first ground tick so CLIMB and APPROACH can use it.
        if on_ground and "runway_hdg" not in self._st:
            self._st["runway_hdg"] = telemetry.heading_deg

        # Clear approach state whenever we leave APPROACH (e.g. go-around).
        prev_phase = self._phase
        self._phase = phase
        if prev_phase == "APPROACH" and phase not in ("APPROACH", "LAND"):
            self._st.pop("approach_hdg_locked", None)
            self._st.pop("approach_locked_hdg", None)
            self._st.pop("approach_initiated", None)
            self._st.pop("approach_tick", None)
            self._st.pop("approach_ceiling_ft", None)
            self._st.pop("approach_aligned", None)
            self._st.pop("approach_path", None)
            self._st.pop("approach_gate_idx", None)
            self._st.pop("approach_descent_fpm", None)
            self._st.pop("approach_wp_captured", None)

        # Reset approach tick counter on entry + flag PID reset
        if prev_phase != "APPROACH" and phase == "APPROACH":
            self._st["approach_tick"] = 0
            self._st["approach_reset_pid"] = True  # clear integral windup from cruise
            print(f"[APPROACH] Phase transition: {prev_phase} → APPROACH")

        # Reset PID on CLIMB→CRUISE to clear climb integral bias
        if prev_phase == "CLIMB" and phase == "CRUISE":
            self._st["cruise_reset_pid"] = True
            print(f"[CRUISE] Phase transition: CLIMB → CRUISE")

        # Reset PID on APPROACH→LAND to clear descent bias before flare
        if prev_phase == "APPROACH" and phase == "LAND":
            self._st["land_reset_pid"] = True
            print(f"[LAND] Phase transition: APPROACH → LAND")

        # ── Dispatch to phase implementation ─────────────────────────────────
        if phase == "GROUND":
            return self._ground(
                telemetry, v_rotate, v_climb, cruise_target_ft,
                takeoff_cfg, throttle_cfg, thresholds,
            )
        if phase == "CLIMB":
            return self._climb(
                telemetry, v_climb, v_stall, cruise_target_ft,
                climb_cfg, throttle_cfg,
            )
        if phase == "CRUISE":
            return self._cruise(telemetry, v_cruise, v_stall, cruise_target_ft, nav_bearing, agl_ft)
        if phase == "APPROACH":
            return self._approach(
                telemetry, v_approach, v_stall, agl_ft, dist_nm, nav_bearing, throttle_cfg, v_land,
            )
        # LAND
        return self._land(telemetry, v_land, agl_ft, throttle_cfg)

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _ground_idle(self, telemetry: Telemetry) -> Targets:
        """Hold on ground, engines idle — used when auto-start is disabled."""
        return Targets(
            heading_deg=telemetry.heading_deg,
            altitude_ft=telemetry.altitude_ft,
            airspeed_kts=0.0,
            throttle=0.0,
            brake_ratio=1.0,
            gear_down=True,
        )

    def _ground(
        self,
        telemetry: Telemetry,
        v_rotate: float,
        v_climb: float,
        cruise_target_ft: float,
        takeoff_cfg: dict,
        throttle_cfg: dict,
        thresholds: dict,
    ) -> Targets:
        """
        Takeoff roll only.

        Full throttle, track runway centerline, no pitch input.
        Once at Vr the phase naturally transitions to CLIMB.
        """
        if "runway_hdg" not in self._st:
            self._st["runway_hdg"] = telemetry.heading_deg
        if "takeoff_lat" not in self._st and telemetry.has_position():
            self._st["takeoff_lat"] = telemetry.lat_deg
            self._st["takeoff_lon"] = telemetry.lon_deg
        runway_hdg = self._st["runway_hdg"]

        # Centerline tracking: correct heading to stay on the runway line
        xtk_correction = 0.0
        if "takeoff_lat" in self._st and telemetry.has_position():
            xtk_correction = _centerline_correction(
                telemetry.lat_deg, telemetry.lon_deg,
                self._st["takeoff_lat"], self._st["takeoff_lon"],
                runway_hdg,
            )
        hold_hdg = (runway_hdg + xtk_correction) % 360.0

        throttle = float(takeoff_cfg.get("throttle", throttle_cfg.get("takeoff", 1.0)))

        # On ground: target = current altitude (no pitch input, just roll straight)
        target_alt = telemetry.altitude_ft

        # Yaw authority tapers from full at 0 kts down to 35% at 50 kts.
        base_yaw_limit = float(takeoff_cfg.get("yaw_limit", 0.35))
        taper = 1.0 - min(max(telemetry.airspeed_kts, 0.0), 50.0) / 50.0
        yaw_limit = max(0.25, base_yaw_limit * (0.35 + 0.65 * taper))

        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=target_alt,
            airspeed_kts=v_climb,
            throttle=throttle,
            brake_ratio=0.0,
            gear_down=True,
            flap_ratio=0.5,     # SF50 takeoff flaps = 50%
            roll_limit=0.0,     # no roll on ground
            pitch_limit=0.05,   # minimal pitch authority on ground
            yaw_hold=True,
            yaw_kp=float(takeoff_cfg.get("yaw_kp", 0.02)),
            yaw_ki=float(takeoff_cfg.get("yaw_ki", 0.00)),
            yaw_limit=yaw_limit,
            yaw_full_deg=float(takeoff_cfg.get("yaw_full_deg", 5.0)),
        )

    def _climb(
        self,
        telemetry: Telemetry,
        v_climb: float,
        v_stall: float,
        cruise_target_ft: float,
        climb_cfg: dict,
        throttle_cfg: dict,
    ) -> Targets:
        """
        Climb to cruise altitude.

        Fixed pitch + fixed throttle. No PID altitude tracking during climb.
        The controller sees climb_rate_fpm set, which bypasses the altitude PID
        and holds a constant pitch. Simple, predictable, no oscillation.
        """
        agl_m = telemetry.agl_m if not math.isnan(telemetry.agl_m) else 0.0
        agl_ft = agl_m * 3.28084

        # ── Heading ──────────────────────────────────────────────────────────
        # Hold runway heading below 300ft AGL, then turn toward destination.
        rwy_hdg = self._st.get("runway_hdg", telemetry.heading_deg)
        if agl_ft < 300.0:
            hold_hdg = rwy_hdg
        else:
            flight_plan = self.ctx.get("flight_plan")
            dest = self.ctx.get("destination")
            climb_bearing = None
            if flight_plan and not flight_plan.completed and flight_plan.active:
                try:
                    from uav.nav.geo import bearing_deg as _cb
                    climb_bearing = _cb(telemetry.lat_deg, telemetry.lon_deg,
                                        flight_plan.active.lat, flight_plan.active.lon)
                except Exception:
                    pass
            if climb_bearing is None and dest:
                try:
                    from uav.nav.geo import bearing_deg as _cb2
                    climb_bearing = _cb2(telemetry.lat_deg, telemetry.lon_deg,
                                         float(dest["lat"]), float(dest["lon"]))
                except Exception:
                    pass
            if climb_bearing is not None:
                prev_cmd = self._st.get("climb_cmd_hdg", rwy_hdg)
                error = _wrap_deg(climb_bearing - prev_cmd)
                advance = max(-self._MAX_TURN_DEG_PER_STEP, min(self._MAX_TURN_DEG_PER_STEP, error))
                hold_hdg = (prev_cmd + advance) % 360.0
                self._st["climb_cmd_hdg"] = hold_hdg
            else:
                hold_hdg = rwy_hdg

        # ── Roll gate ────────────────────────────────────────────────────────
        max_roll_cfg = float(climb_cfg.get("max_roll_cmd", 0.15))
        if agl_ft < 300.0:
            roll_lim = 0.0
        elif agl_ft < 600.0:
            roll_lim = max_roll_cfg * (agl_ft - 300.0) / 300.0
        else:
            roll_lim = max_roll_cfg

        # ── Yaw hold while roll is restricted ────────────────────────────────
        takeoff_cfg = self.ctx.get("takeoff", {})
        if agl_ft < 600.0:
            yaw_hold = True
            yaw_kp = float(takeoff_cfg.get("yaw_kp", 0.07))
            yaw_ki = float(takeoff_cfg.get("yaw_ki", 0.02))
            yaw_limit = float(takeoff_cfg.get("yaw_limit", 0.5))
            yaw_full_deg = float(takeoff_cfg.get("yaw_full_deg", 5.0))
        else:
            yaw_hold = False
            yaw_kp = yaw_ki = yaw_limit = yaw_full_deg = None

        # ── Fixed climb: constant throttle + constant pitch ──────────────────
        # Target altitude = cruise. climb_rate_fpm tells the controller to use
        # a fixed pitch instead of the altitude PID. No hunting, no oscillation.
        climb_throttle = float(climb_cfg.get("throttle_min", throttle_cfg.get("climb", 0.95)))
        climb_fpm = float(self.ctx.get("airframe", {}).get("rates_fpm", {}).get("climb", 800.0))

        gear_down = agl_ft <= 50.0  # retract gear early
        climb_flaps = 0.5 if agl_ft < 300.0 else 0.0

        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=cruise_target_ft,        # target = cruise altitude
            airspeed_kts=v_climb,
            climb_rate_fpm=climb_fpm,             # tells controller: fixed pitch climb
            throttle=climb_throttle,
            brake_ratio=0.0,
            gear_down=gear_down,
            flap_ratio=climb_flaps,
            roll_limit=roll_lim,
            pitch_limit=float(climb_cfg.get("max_pitch_cmd", 0.15)),
            pitch_protect_kts=v_stall + 15.0,
            pitch_protect_gain=0.03,
            yaw_hold=yaw_hold,
            yaw_kp=yaw_kp,
            yaw_ki=yaw_ki,
            yaw_limit=yaw_limit,
            yaw_full_deg=yaw_full_deg,
            reset_alt_pid=True,  # always reset PID during climb — we're not using it
        )

    def _cruise(
        self,
        telemetry: Telemetry,
        v_cruise: float,
        v_stall: float,
        cruise_target_ft: float,
        true_bearing: Optional[float],
        agl_ft: float,
    ) -> Targets:
        """
        Cruise at altitude toward destination.

        Heading is advanced at most _MAX_TURN_DEG_PER_STEP per tick so the
        heading PID always sees a small error and rolls stay proportional.

        Terrain avoidance: if AGL drops below 300ft during cruise (rising terrain),
        the altitude target is raised to maintain at least 300ft AGL clearance.
        """
        if true_bearing is not None:
            # Carry over climb heading if transitioning from CLIMB
            prev_cmd = self._st.get("cruise_cmd_hdg",
                        self._st.get("climb_cmd_hdg", telemetry.heading_deg))
            error = _wrap_deg(true_bearing - prev_cmd)
            advance = max(-self._MAX_TURN_DEG_PER_STEP, min(self._MAX_TURN_DEG_PER_STEP, error))
            cmd_hdg = (prev_cmd + advance) % 360.0
            self._st["cruise_cmd_hdg"] = cmd_hdg
        else:
            cmd_hdg = self._st.get("cruise_cmd_hdg", telemetry.heading_deg)

        # Terrain floor: if rising terrain brings AGL below 300ft, push target up.
        terrain_floor_ft = (telemetry.altitude_ft - agl_ft) + 300.0
        effective_target = max(cruise_target_ft, terrain_floor_ft)

        # Cruise: HIGH throttle (go fast), pitch holds altitude.
        # Like a real pilot: power set for max cruise, stick holds level.
        cruise_throttle = float(self.ctx.get("airframe", {}).get("throttle", {}).get("cruise", 0.85))

        # Reset PID on first cruise tick to clear climb integral windup
        do_reset = self._st.pop("cruise_reset_pid", False)

        return Targets(
            heading_deg=cmd_hdg,
            altitude_ft=effective_target,
            airspeed_kts=v_cruise,
            throttle=cruise_throttle,       # hardcoded power — go fast
            brake_ratio=0.0,
            gear_down=False,
            pitch_limit=None,              # PID has full authority to hold altitude
            roll_limit=0.12,
            pitch_protect_kts=v_stall + 20.0,
            pitch_protect_gain=0.03,
            reset_alt_pid=do_reset,
        )

    def _approach(
        self,
        telemetry: Telemetry,
        v_approach: float,
        v_stall: float,
        agl_ft: float,
        dist_nm: float,
        true_bearing: Optional[float],
        throttle_cfg: dict,
        v_land: float = 60.0,
    ) -> Targets:
        """
        Precision approach — follows a pre-computed path of gates along
        the extended runway centerline.

        Phase 1 — ALIGN:
           Turn toward the runway, slow down, hold altitude.
           Build the approach path once aligned.

        Phase 2 — PATH FOLLOW:
           Fly through each gate in order. Each gate has:
             - lat/lon (on the extended centerline)
             - target altitude (MSL)
             - target speed
           Between gates, interpolate the target altitude linearly.
           Cross-track correction keeps the plane on the centerline.
        """
        tick = self._st.get("approach_tick", 0)
        self._st["approach_tick"] = tick + 1
        speed = telemetry.airspeed_kts

        if "approach_ceiling_ft" not in self._st:
            self._st["approach_ceiling_ft"] = telemetry.altitude_ft
            print(f"[APPROACH] ENTER: alt={telemetry.altitude_ft:.0f}ft "
                  f"speed={speed:.0f}kts v_approach={v_approach:.0f}kts")

        # ── Runway data ───────────────────────────────────────────────
        dest_rwy = self.ctx.get("dest_runway")
        rwy_hdg = dest_rwy["heading"] if dest_rwy else None

        rwy_bearing = true_bearing
        rwy_dist_nm = dist_nm
        if dest_rwy and telemetry.has_position():
            try:
                from uav.nav.geo import haversine_m as _hm, bearing_deg as _brg
                rwy_dist_nm = _hm(
                    telemetry.lat_deg, telemetry.lon_deg,
                    dest_rwy["threshold_lat"], dest_rwy["threshold_lon"],
                ) / 1852.0
                rwy_bearing = _brg(
                    telemetry.lat_deg, telemetry.lon_deg,
                    dest_rwy["threshold_lat"], dest_rwy["threshold_lon"],
                )
            except Exception:
                pass

        # ── Cross-track error ─────────────────────────────────────────
        xtk_deg = 0.0
        xtk_m_raw = 0.0
        if dest_rwy and rwy_hdg is not None and telemetry.has_position():
            xtk_deg = _centerline_correction(
                telemetry.lat_deg, telemetry.lon_deg,
                dest_rwy["threshold_lat"], dest_rwy["threshold_lon"],
                rwy_hdg,
            )
            xtk_m_raw = _cross_track_m(
                telemetry.lat_deg, telemetry.lon_deg,
                dest_rwy["threshold_lat"], dest_rwy["threshold_lon"],
                rwy_hdg,
            )

        xtk_m = abs(xtk_m_raw)

        # ── ALIGNMENT GATE ────────────────────────────────────────────
        hdg_error = abs(_wrap_deg(telemetry.heading_deg - rwy_hdg)) if rwy_hdg is not None else 999.0
        speed_ok = speed < 150.0
        aligned = hdg_error < 20.0 and xtk_m < 500.0 and speed_ok

        if aligned and not self._st.get("approach_aligned", False):
            self._st["approach_aligned"] = True
            # Build the precision approach path
            if dest_rwy:
                rwy_elev = dest_rwy.get("elevation_ft", telemetry.altitude_ft - agl_ft)
                path = _build_approach_path(
                    dest_rwy["threshold_lat"], dest_rwy["threshold_lon"],
                    rwy_hdg, rwy_elev,
                    telemetry.altitude_ft, v_approach, v_land,
                )
                self._st["approach_path"] = path
                self._st["approach_gate_idx"] = 0
                gate_names = [g["name"] for g in path]
                print(f"[APPROACH] ALIGNED — built precision path: {gate_names}")
            print(f"[APPROACH] ALIGNED: hdg_err={hdg_error:.0f}° "
                  f"xtk={xtk_m:.0f}m spd={speed:.0f}kts — following path")

        is_aligned = self._st.get("approach_aligned", False)

        # ── PHASE 1: ALIGN — turn toward runway, slow down, HOLD ALTITUDE ──
        if not is_aligned:
            if rwy_bearing is not None:
                hold_hdg = rwy_bearing
            elif rwy_hdg is not None:
                hold_hdg = rwy_hdg
            else:
                hold_hdg = telemetry.heading_deg
            roll_limit = 0.20

            # CRITICAL: ALIGN must hold altitude — use enough throttle!
            # Old code used 0.0 throttle which caused -1200fpm sinkrate.
            # Use speed-based throttle: slow down gently, but maintain altitude.
            if speed > 210.0:
                approach_throttle = 0.0   # way too fast, idle to slow
                gear_down = False
                flap_ratio = 0.0
            elif speed > 190.0:
                approach_throttle = 0.10  # still fast, mostly idle
                gear_down = True
                flap_ratio = 0.0
            elif speed > 150.0:
                approach_throttle = 0.20  # moderate — need some power with drag
                gear_down = True
                flap_ratio = 0.3
            else:
                approach_throttle = 0.35  # slow enough — maintain altitude!
                gear_down = True
                flap_ratio = 0.5

            # Safety: if sinking during ALIGN, add power to arrest descent
            actual_vs_align = telemetry.vs_fpm if not math.isnan(telemetry.vs_fpm) else 0.0
            if actual_vs_align < -500.0:
                approach_throttle = max(approach_throttle, 0.70)
            elif actual_vs_align < -200.0:
                approach_throttle = max(approach_throttle, 0.50)

            if tick < 10 or tick % 45 == 0:
                actual_vs = telemetry.vs_fpm if not math.isnan(telemetry.vs_fpm) else 0.0
                print(f"[APPROACH] t={tick} ALIGN hdg_err={hdg_error:.0f}° "
                      f"xtk={xtk_m:.0f}m spd={speed:.0f}kts "
                      f"alt={telemetry.altitude_ft:.0f} vs={actual_vs:.0f}fpm "
                      f"dist={rwy_dist_nm:.1f}nm spd_ok={'Y' if speed_ok else 'N'} "
                      f"thr={approach_throttle:.2f}")

            do_reset = self._st.pop("approach_reset_pid", False)

            return Targets(
                heading_deg=hold_hdg,
                altitude_ft=self._st["approach_ceiling_ft"],
                airspeed_kts=v_approach,
                throttle=approach_throttle,
                brake_ratio=0.0,
                gear_down=gear_down,
                roll_limit=roll_limit,
                pitch_limit=0.10,
                pitch_protect_kts=v_stall + 10.0,
                pitch_protect_gain=0.04,
                flap_ratio=flap_ratio,
                reset_alt_pid=do_reset,
            )

        # ── PHASE 2: PRECISION PATH FOLLOW ───────────────────────────
        # Follow the pre-computed gates. At each tick:
        # 1. Find the active gate (next one ahead of us)
        # 2. Steer toward it (bearing + XTK correction)
        # 3. Compute target altitude by interpolating between gates
        # 4. Use VS PID to track the descent rate needed

        path = self._st.get("approach_path", [])
        gate_idx = self._st.get("approach_gate_idx", 0)
        rwy_elev_ft = dest_rwy.get("elevation_ft", 0.0) if dest_rwy else (telemetry.altitude_ft - agl_ft)

        # ── Advance through gates ─────────────────────────────────────
        # Check if we've passed the current gate (distance to it < capture radius)
        if path and gate_idx < len(path) and telemetry.has_position():
            from uav.nav.geo import haversine_m as _hm3, bearing_deg as _brg3
            gate = path[gate_idx]
            gate_dist_m = _hm3(
                telemetry.lat_deg, telemetry.lon_deg,
                gate["lat"], gate["lon"],
            )
            gate_dist_nm = gate_dist_m / 1852.0

            # Capture radius: tighter for close gates, looser for far ones
            if gate["agl_ft"] < 200:
                capture_nm = 0.10
            elif gate["agl_ft"] < 500:
                capture_nm = 0.15
            else:
                capture_nm = 0.25

            if gate_dist_nm < capture_nm:
                print(f"[APPROACH] GATE {gate['name']} reached "
                      f"(alt={telemetry.altitude_ft:.0f} target={gate['alt_ft']:.0f} "
                      f"err={telemetry.altitude_ft - gate['alt_ft']:+.0f}ft)")
                gate_idx += 1
                self._st["approach_gate_idx"] = gate_idx

        # ── Compute target altitude from gate path ────────────────────
        # Interpolate between the gate we just passed and the next one
        if path and gate_idx < len(path):
            gate = path[gate_idx]
            target_alt_ft = gate["alt_ft"]
            target_speed = gate["speed_kts"]

            # If we have a previous gate, interpolate altitude based on distance
            if gate_idx > 0:
                prev_gate = path[gate_idx - 1]
                from uav.nav.geo import haversine_m as _hm4
                total_seg_m = _hm4(prev_gate["lat"], prev_gate["lon"],
                                    gate["lat"], gate["lon"])
                dist_to_gate_m = _hm4(telemetry.lat_deg, telemetry.lon_deg,
                                       gate["lat"], gate["lon"]) if telemetry.has_position() else total_seg_m
                # Progress through segment (0 = at prev gate, 1 = at next gate)
                if total_seg_m > 0:
                    progress = max(0.0, min(1.0, 1.0 - (dist_to_gate_m / total_seg_m)))
                else:
                    progress = 1.0
                target_alt_ft = prev_gate["alt_ft"] + (gate["alt_ft"] - prev_gate["alt_ft"]) * progress
                target_speed = prev_gate["speed_kts"] + (gate["speed_kts"] - prev_gate["speed_kts"]) * progress
            else:
                # Before first gate — use first gate as target
                target_alt_ft = gate["alt_ft"]
                target_speed = gate["speed_kts"]
        else:
            # Past all gates — target 50ft above runway
            target_alt_ft = rwy_elev_ft + 50.0
            target_speed = v_approach

        # ── Descent rate from altitude error ──────────────────────────
        # How much we need to descend to match the path
        alt_error = telemetry.altitude_ft - target_alt_ft  # positive = too high
        alt_to_lose_ft = max(0.0, telemetry.altitude_ft - rwy_elev_ft - 50.0)
        horiz_dist_ft = max(rwy_dist_nm, 0.05) * 6076.0

        # Geometric descent rate (Pythagorean)
        glide_path_ft = math.sqrt(alt_to_lose_ft ** 2 + horiz_dist_ft ** 2)
        speed_ft_per_min = max(speed, 60.0) * 6076.0 / 60.0
        time_to_fly_min = glide_path_ft / speed_ft_per_min if speed_ft_per_min > 0 else 999.0
        geometric_descent = alt_to_lose_ft / time_to_fly_min if time_to_fly_min > 0 else 0.0

        # Path-following descent: base descent + correction for being off-path
        # If we're 100ft above the path → add 200fpm extra descent
        # If we're below the path → reduce descent (level off)
        path_correction_fpm = alt_error * 2.0  # 2 fpm per ft of error
        path_correction_fpm = max(-300.0, min(300.0, path_correction_fpm))

        required_descent_fpm = geometric_descent + path_correction_fpm

        # ── GPWS ENVELOPE ─────────────────────────────────────────────
        if agl_ft > 1000:
            gpws_max = 1200.0
        elif agl_ft > 500:
            gpws_max = 900.0
        elif agl_ft > 200:
            gpws_max = 700.0
        elif agl_ft > 100:
            gpws_max = 500.0
        elif agl_ft > 50:
            gpws_max = 350.0
        else:
            gpws_max = 200.0

        glide_angle_deg = math.degrees(math.atan2(alt_to_lose_ft, horiz_dist_ft)) if horiz_dist_ft > 0 else 0
        if glide_angle_deg > 6.0:
            gpws_max = min(gpws_max, 600.0)

        if alt_to_lose_ft <= 0:
            required_descent_fpm = 0.0
        elif required_descent_fpm < 0:
            # Below glidepath — allow climbing back up (negative descent = climb)
            # Cap the climb rate to 500fpm — don't pull up aggressively during approach
            required_descent_fpm = max(-500.0, required_descent_fpm)
        else:
            required_descent_fpm = min(gpws_max, required_descent_fpm)

        # Smooth transitions — max 80fpm change per tick
        prev_descent = self._st.get("approach_descent_fpm", required_descent_fpm)
        delta_descent = required_descent_fpm - prev_descent
        delta_descent = max(-80.0, min(80.0, delta_descent))
        descent_fpm = prev_descent + delta_descent
        self._st["approach_descent_fpm"] = descent_fpm

        # ── HEADING — steer toward active gate + XTK correction ───────
        # This is the KEY difference: we steer toward the NEXT GATE,
        # not just the runway heading. The gates ARE on the centerline,
        # so this naturally keeps us aligned.
        if path and gate_idx < len(path) and telemetry.has_position():
            from uav.nav.geo import bearing_deg as _brg5
            gate = path[gate_idx]
            gate_bearing = _brg5(
                telemetry.lat_deg, telemetry.lon_deg,
                gate["lat"], gate["lon"],
            )
            # Blend: 70% gate bearing + 30% XTK correction
            # At low altitude, increase XTK correction weight
            if agl_ft < 100:
                hold_hdg = (gate_bearing + xtk_deg * 0.8) % 360.0
                roll_limit = 0.04
            elif agl_ft < 300:
                hold_hdg = (gate_bearing + xtk_deg * 0.5) % 360.0
                roll_limit = 0.06
            elif agl_ft < 500:
                hold_hdg = gate_bearing % 360.0
                roll_limit = 0.08
            else:
                hold_hdg = gate_bearing % 360.0
                roll_limit = 0.10
        elif rwy_hdg is not None:
            hold_hdg = (rwy_hdg + xtk_deg) % 360.0
            roll_limit = 0.10
        else:
            hold_hdg = rwy_bearing if rwy_bearing is not None else telemetry.heading_deg
            roll_limit = 0.12

        # ── SPEED + ENERGY MANAGEMENT ────────────────────────────────
        target_speed = min(target_speed, v_approach + 30)
        speed_err = speed - target_speed
        if speed_err > 20:
            approach_throttle = 0.0
        elif speed_err > 5:
            approach_throttle = 0.10
        elif speed_err > -5:
            approach_throttle = 0.30
        elif speed_err > -15:
            approach_throttle = 0.50
        else:
            approach_throttle = 0.70

        # Energy correction: if we're BELOW the glidepath, add throttle
        # to give the plane energy to climb back. This is what a real pilot
        # does — "low, add power".
        if alt_error < -100:
            # Significantly below path — need power to get back up
            approach_throttle = max(approach_throttle, 0.80)
        elif alt_error < -50:
            # Moderately below — bump throttle
            approach_throttle = max(approach_throttle, 0.60)

        # Gear + flaps
        if speed > 200.0:
            gear_down = False
            flap_ratio = 0.0
        elif speed > 160.0:
            gear_down = True
            flap_ratio = 0.0
        elif speed > 130.0:
            gear_down = True
            flap_ratio = 0.3
        elif speed > v_approach + 10:
            gear_down = True
            flap_ratio = 0.7
        else:
            gear_down = True
            flap_ratio = 1.0

        # ── Logging ───────────────────────────────────────────────────
        actual_vs = telemetry.vs_fpm if not math.isnan(telemetry.vs_fpm) else 0.0
        gate_name = path[gate_idx]["name"] if path and gate_idx < len(path) else "DONE"
        if tick < 10 or tick % 30 == 0:
            print(f"[APPROACH] t={tick} PATH gate={gate_name} "
                  f"alt={telemetry.altitude_ft:.0f} target={target_alt_ft:.0f} "
                  f"err={alt_error:+.0f}ft agl={agl_ft:.0f} "
                  f"vs_target={-descent_fpm:.0f} actual_vs={actual_vs:.0f}fpm "
                  f"spd={speed:.0f}/{target_speed:.0f}kts "
                  f"xtk={xtk_m_raw:+.0f}m dist={rwy_dist_nm:.1f}nm "
                  f"thr={approach_throttle:.2f}")

        do_reset = self._st.pop("approach_reset_pid", False)

        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=telemetry.altitude_ft,
            airspeed_kts=target_speed,
            climb_rate_fpm=-descent_fpm,
            throttle=approach_throttle,
            brake_ratio=0.0,
            gear_down=gear_down,
            roll_limit=roll_limit,
            pitch_limit=0.10,
            pitch_protect_kts=v_stall + 10.0,
            pitch_protect_gain=0.04,
            flap_ratio=flap_ratio,
            reset_alt_pid=do_reset,
        )

    def _land(
        self,
        telemetry: Telemetry,
        v_land: float,
        agl_ft: float,
        throttle_cfg: dict,
    ) -> Targets:
        """
        Flare and rollout using VS control — NOT altitude PID.

        The flare uses an exponential decay on descent rate:
          vs_target = vs_touchdown + (vs_entry - vs_touchdown) * (agl / agl_entry)^0.5

        This smoothly reduces the descent rate from whatever the approach
        was doing (~500fpm) to a gentle touchdown (~100fpm).

        The key insight: control DESCENT RATE, not altitude. The altitude
        PID doesn't know how fast the plane is falling. The VS PID does.
        """
        tick = self._st.get("land_tick", 0)
        self._st["land_tick"] = tick + 1

        # ── Lock flare entry state on first tick ─────────────────────
        if "flare_entry_agl" not in self._st:
            actual_vs = telemetry.vs_fpm if not math.isnan(telemetry.vs_fpm) else -500.0
            self._st["flare_entry_agl"] = max(agl_ft, 10.0)
            self._st["flare_entry_vs"] = min(actual_vs, -100.0)  # must be descending
            self._st["flare_entry_time"] = telemetry.timestamp
            print(f"[LAND] FLARE ENTRY: agl={agl_ft:.0f}ft vs={actual_vs:.0f}fpm")

        entry_agl = self._st["flare_entry_agl"]
        entry_vs = self._st["flare_entry_vs"]   # negative, e.g. -500
        entry_time = self._st["flare_entry_time"]

        # ── Runway heading + XTK correction ──────────────────────────
        dest_rwy = self.ctx.get("dest_runway")
        if dest_rwy:
            base_hdg = dest_rwy["heading"]
            xtk_correction = 0.0
            if telemetry.has_position():
                xtk_correction = _centerline_correction(
                    telemetry.lat_deg, telemetry.lon_deg,
                    dest_rwy["threshold_lat"], dest_rwy["threshold_lon"],
                    base_hdg,
                )
            # Keep XTK correction but gentle — wings almost level
            hold_hdg = (base_hdg + xtk_correction * 0.3) % 360.0
        else:
            hold_hdg = self._st.get("runway_hdg", telemetry.heading_deg)

        on_ground = agl_ft < 3.0
        idle_throttle = float(throttle_cfg.get("idle", 0.0))

        if on_ground:
            # ── ROLLOUT — on the ground ──────────────────────────────
            # Hold heading, brakes on, zero throttle
            if tick < 5 or tick % 30 == 0:
                print(f"[LAND] ROLLOUT agl={agl_ft:.0f} spd={telemetry.airspeed_kts:.0f}kts")
            return Targets(
                heading_deg=hold_hdg,
                altitude_ft=telemetry.altitude_ft,
                airspeed_kts=v_land,
                throttle=idle_throttle,
                brake_ratio=1.0,
                gear_down=True,
                pitch_limit=0.05,
                flap_ratio=1.0,
                yaw_hold=True,
                yaw_kp=0.02,
                yaw_limit=0.5,
            )

        # ── FLARE — exponential descent rate decay ───────────────────
        #
        # Target touchdown VS: -100 fpm (~1.7 fps, gentle landing)
        # Entry VS: whatever approach was doing (e.g. -500fpm)
        #
        # Decay based on AGL ratio (height-based, not time-based):
        #   vs_target = vs_td + (vs_entry - vs_td) * (agl / agl_entry)^0.5
        #
        # At entry (agl=entry_agl): vs_target = vs_entry (no change)
        # At 1/4 height:            vs_target = vs_td + 0.5*(vs_entry-vs_td)
        # At ground (agl=0):        vs_target = vs_td (-100fpm)
        #
        # The sqrt (^0.5) makes the initial pitch-up aggressive and
        # the final approach gentle — exactly what a flare should feel like.

        vs_touchdown = -100.0  # target: -100fpm at touchdown
        agl_ratio = max(0.0, min(1.0, agl_ft / entry_agl))
        vs_target = vs_touchdown + (entry_vs - vs_touchdown) * math.sqrt(agl_ratio)

        # Hard safety floor: NEVER descend faster than AGL allows
        # This is the last line of defense against crashing
        if agl_ft < 10.0:
            vs_target = max(vs_target, -150.0)   # very gentle near ground
        elif agl_ft < 30.0:
            vs_target = max(vs_target, -300.0)

        # Throttle: idle, but add a touch of power if sinking too fast
        actual_vs = telemetry.vs_fpm if not math.isnan(telemetry.vs_fpm) else vs_target
        if actual_vs < vs_target - 200.0:
            # Sinking way too fast — add power to arrest
            land_throttle = 0.30
        elif actual_vs < vs_target - 100.0:
            land_throttle = 0.15
        else:
            land_throttle = idle_throttle

        if tick < 10 or tick % 15 == 0:
            print(f"[LAND] FLARE agl={agl_ft:.0f} vs_target={vs_target:.0f} "
                  f"actual_vs={actual_vs:.0f}fpm ratio={agl_ratio:.2f} "
                  f"thr={land_throttle:.2f} spd={telemetry.airspeed_kts:.0f}kts")

        do_reset = self._st.pop("land_reset_pid", False)

        return Targets(
            heading_deg=hold_hdg,
            altitude_ft=telemetry.altitude_ft,  # doesn't matter — VS drives
            airspeed_kts=v_land,
            climb_rate_fpm=vs_target,            # VS PID tracks this
            throttle=land_throttle,
            brake_ratio=0.0,
            gear_down=True,
            roll_limit=0.03,                     # nearly wings level
            pitch_limit=0.08,                    # gentle nose-up allowed
            pitch_protect_kts=v_land - 10.0,     # stall protection
            pitch_protect_gain=0.04,
            flap_ratio=1.0,
            reset_alt_pid=do_reset,
        )
