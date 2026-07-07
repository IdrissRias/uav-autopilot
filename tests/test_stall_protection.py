"""Low-and-slow is the corner of the energy matrix where throttle is
the ONLY fix. Flight 20260706_135230 mushed 113 → 68 kts at idle
because the plane was above the slope and the alt-priority law refused
power. These tests pin the three guards that prevent it:

  1. stall floor forces throttle when slow, overriding alt priority
  2. pitch never dives for speed while at/below target altitude
  3. descent speed target becomes a ceiling (v_approach) above slope
"""
from __future__ import annotations

import unittest

from uav.core.control.pid import PID
from uav.core.control.simple_fixedwing import SimpleFixedWingController
from uav.core.flight_engine import FlightEngine
from uav.nav.flight_plan_v2 import plan_path
from uav.sim.types import Targets, Telemetry


def _ctl():
    return SimpleFixedWingController(
        PID(0.01, 0, 0), PID(0.001, 0.0001, 0.002), PID(0.02, 0.002, 0.0),
        cruise_throttle=0.55,
    )


class TestStallFloor(unittest.TestCase):
    def test_power_forced_when_slow_even_above_slope(self):
        # The crash scenario: above slope (alt error negative), coupled
        # mode, speed collapsing. Throttle must come in anyway.
        ctl = _ctl()
        tg = Targets(heading_deg=90.0, altitude_ft=2000.0, airspeed_kts=106.0,
                     throttle=None, throttle_for_alt=True, throttle_base=0.12,
                     stall_floor_kts=98.4)
        t = Telemetry(airspeed_kts=88.0, altitude_ft=2400.0, pitch_deg=5.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=200.0, agl_m=350.0)
        act = ctl.compute(t, tg, 0.05)
        self.assertGreaterEqual(
            act.throttle, 0.9,
            "10+ kts below the stall floor → near-full power, no slew delay."
        )

    def test_floor_overrides_explicit_idle(self):
        # DECELERATE commands idle explicitly; the floor still wins.
        ctl = _ctl()
        tg = Targets(heading_deg=90.0, altitude_ft=5700.0, airspeed_kts=133.0,
                     throttle=0.0, stall_floor_kts=98.4)
        t = Telemetry(airspeed_kts=90.0, altitude_ft=5700.0, pitch_deg=2.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=0.0, agl_m=1300.0)
        act = ctl.compute(t, tg, 0.05)
        self.assertGreaterEqual(act.throttle, 0.8)

    def test_no_floor_during_flare(self):
        # FLARE has stall_floor_kts=None — slow there is by design.
        ctl = _ctl()
        tg = Targets(heading_deg=90.0, altitude_ft=1243.0, airspeed_kts=98.4,
                     throttle=0.0, stall_floor_kts=None, vs_target_fpm=-200.0)
        t = Telemetry(airspeed_kts=95.0, altitude_ft=1260.0, pitch_deg=4.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=-300.0, agl_m=6.0)
        act = ctl.compute(t, tg, 0.05)
        self.assertEqual(act.throttle, 0.0,
                         "Flare keeps idle; no stall floor interference.")


class TestNoDiveForSpeed(unittest.TestCase):
    def test_pitch_capped_when_slow_and_below_alt(self):
        # TRANSITION balloon cause: 20 kts slow, 50 ft LOW → the pitch
        # law dove -0.31 to buy speed with altitude it didn't have.
        ctl = _ctl()
        tg = Targets(heading_deg=90.0, altitude_ft=5712.0, airspeed_kts=132.8,
                     throttle=None, throttle_for_alt=True)
        t = Telemetry(airspeed_kts=112.0, altitude_ft=5664.0, pitch_deg=5.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=0.0, agl_m=1380.0)
        act = ctl.compute(t, tg, 0.05)
        self.assertGreaterEqual(
            act.pitch, -0.05,
            "Below target alt, nose-down for speed is capped — throttle's job."
        )

    def test_dive_for_speed_allowed_when_high(self):
        ctl = _ctl()
        tg = Targets(heading_deg=90.0, altitude_ft=5712.0, airspeed_kts=132.8,
                     throttle=None, throttle_for_alt=True)
        t = Telemetry(airspeed_kts=112.0, altitude_ft=5900.0, pitch_deg=5.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=0.0, agl_m=1440.0)
        act = ctl.compute(t, tg, 0.05)
        self.assertLess(act.pitch, -0.1,
                        "With altitude to spare, diving for speed is fine.")


class TestBleedMode(unittest.TestCase):
    def test_holds_drag_attitude_at_target_speed(self):
        # The P-law relaxes to zero at target speed → best-glide, no
        # drag, slope never captured. Bleed mode holds the nose up.
        ctl = _ctl()
        tg = Targets(heading_deg=90.0, altitude_ft=4000.0, airspeed_kts=106.0,
                     throttle=None, throttle_for_alt=True, throttle_base=0.12,
                     bleed_mode=True)
        t = Telemetry(airspeed_kts=112.0, altitude_ft=4400.0, pitch_deg=2.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=-850.0, agl_m=900.0)
        act = ctl.compute(t, tg, 0.05)
        self.assertGreaterEqual(act.pitch, 0.12,
                                "Bleeding: hold a real nose-up drag attitude.")

    def test_bleed_never_climbs(self):
        # Pulling into a zoom re-banks the energy as altitude (the
        # 20-kt zoom-stall). Climbing → pitch capped near level.
        ctl = _ctl()
        tg = Targets(heading_deg=90.0, altitude_ft=4000.0, airspeed_kts=106.0,
                     throttle=None, throttle_for_alt=True, throttle_base=0.12,
                     bleed_mode=True)
        t = Telemetry(airspeed_kts=170.0, altitude_ft=4400.0, pitch_deg=10.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=1500.0, agl_m=900.0)
        act = ctl.compute(t, tg, 0.05)
        self.assertLessEqual(act.pitch, 0.04,
                             "Climbing while bleeding is forbidden.")


class TestDescentSpeedCeiling(unittest.TestCase):
    def setUp(self):
        self.ribbon = plan_path(
            dep_lat=45.42, dep_lon=-91.77, dep_alt_ft=1100.0,
            dep_heading=10.0,
            dest_lat=47.40, dest_lon=-94.77, dest_alt_ft=1380.0,
            dest_rwy_heading=270.0,
            dest_threshold_lat=47.40, dest_threshold_lon=-94.77,
            cruise_alt_ft=5000.0, v_stall=98.4, v_cruise_kts=132.7,
        )
        self.engine = FlightEngine({})
        self.kf = next(k for k in self.ribbon.keyframes if k.name == "DESCENT")

    def _resolve(self, alt_ft: float):
        self.engine._prev_targets = Targets(
            heading_deg=270.0, altitude_ft=alt_ft, airspeed_kts=120.0,
            flap_ratio=0.0, gear_down=False,
        )
        t = Telemetry(airspeed_kts=120.0, altitude_ft=alt_ft, pitch_deg=-3.0,
                      roll_deg=0.0, heading_deg=270.0, timestamp=0.0,
                      lat_deg=47.40, lon_deg=-94.72,
                      vs_fpm=-800.0, agl_m=(alt_ft - 1380.0) / 3.28084)
        return self.engine._resolve(self.kf, t, self.ribbon)

    def test_speed_target_drops_when_above_slope(self):
        on_slope = self._resolve(3000.0).altitude_ft
        out = self._resolve(on_slope + 300.0)
        self.assertAlmostEqual(
            out.airspeed_kts, self.ribbon.geometry.v_approach, delta=0.1,
        )

    def test_scheduled_speed_when_on_slope(self):
        on_slope = self._resolve(3000.0).altitude_ft
        out = self._resolve(on_slope)
        self.assertAlmostEqual(out.airspeed_kts, self.kf.target_speed_kts,
                               delta=0.1)

    def test_stall_floor_set_in_flight_phases(self):
        on_slope = self._resolve(3000.0).altitude_ft
        out = self._resolve(on_slope)
        self.assertAlmostEqual(out.stall_floor_kts,
                               self.ribbon.geometry.v_land, delta=0.1)

    def test_gear_leads_the_descent(self):
        # Clean mush (~850 fpm) cannot fly the 1000 ft/nm slope: gear
        # (drag without lift) deploys as soon as the descent begins and
        # speed permits, so the slope is actually capturable.
        on_slope = self._resolve(3000.0).altitude_ft
        out = self._resolve(on_slope + 300.0)  # above slope, 120 kts
        self.assertTrue(out.gear_down,
                        "Gear must lead the descent (drag ladder).")

    def test_bleed_mode_flag_set_above_slope(self):
        on_slope = self._resolve(3000.0).altitude_ft
        self.assertTrue(self._resolve(on_slope + 300.0).bleed_mode)
        self.assertFalse(self._resolve(on_slope).bleed_mode)


if __name__ == "__main__":
    unittest.main()
