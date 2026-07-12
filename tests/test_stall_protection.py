"""Power is an ALTITUDE servo, nothing else (Idriss doctrine, 2026-07-11).

  - ABOVE the commanded line → power is 0, always. A slow airplane up
    there is sinking, and sinking toward the path is the goal — there is
    nothing to rescue. There is no stall FLOOR and no power JUMP: the old
    deficit-scaled slam WAS the abrupt surge on final we chased out.
  - BELOW the line → power walks up gradually, in proportion to the sag.
    A low-and-slow airplane sinks below the line and the walk answers,
    bit by bit — never a step, never a fixed setpoint.
  - Explicit idle (FLARE/ROLLOUT) is absolute; nothing adds power back.

Plus: pitch never dives for speed while at/below target altitude. Rejoining
the glideslope (pull up when low / nose down when high) is the controller's
pitch step-and-check, not a vs_target convergence term (Idriss, 2026-07-12).
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
    def test_power_walks_up_when_below_line(self):
        # BELOW the commanded line: power comes up gradually, in
        # proportion to the sag — bit by bit, never a step.
        ctl = _ctl()
        ctl._prev_throttle = 0.20
        ctl._thr_walk = 0.20
        tg = Targets(heading_deg=90.0, altitude_ft=2600.0, airspeed_kts=106.0,
                     throttle=None, stall_floor_kts=98.4)
        t = Telemetry(airspeed_kts=88.0, altitude_ft=2200.0, pitch_deg=5.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=-400.0, agl_m=350.0)  # 400 ft below the line
        prev = None
        for _ in range(60):
            act = ctl.compute(t, tg, 0.05)
            if prev is not None:
                self.assertGreaterEqual(act.throttle, prev - 1e-9,
                                        "Below the line, power only rises.")
                self.assertLess(act.throttle - prev, 0.06,
                                "Gradual — no step inputs, ever.")
            prev = act.throttle
        self.assertGreater(act.throttle, 0.30,
                           "3 s below the line → power has walked up.")

    def test_no_power_when_slow_but_high(self):
        # ABOVE the line, slow is not a danger — it is a sink toward the
        # path, which is the goal. Power → idle regardless of speed.
        ctl = _ctl()
        ctl._prev_throttle = 0.45
        ctl._thr_walk = 0.45
        tg = Targets(heading_deg=90.0, altitude_ft=2000.0, airspeed_kts=106.0,
                     throttle=None, stall_floor_kts=98.4)
        t = Telemetry(airspeed_kts=90.0, altitude_ft=2400.0, pitch_deg=5.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=-300.0, agl_m=350.0)  # 400 ft above target
        for _ in range(60):
            act = ctl.compute(t, tg, 0.05)
        self.assertLess(act.throttle, 0.05,
                        "400 ft above target: power is zero, let it sink.")

    def test_no_power_when_high_even_stalling_on_short_final(self):
        # The corrected doctrine (Idriss, 2026-07-11): being near the
        # ground does NOT buy power when we are ABOVE the line. A stall
        # up here means we are going down — exactly what we want.
        ctl = _ctl()
        ctl._prev_throttle = 0.40
        ctl._thr_walk = 0.40
        tg = Targets(heading_deg=90.0, altitude_ft=1500.0, airspeed_kts=103.0,
                     throttle=None, stall_floor_kts=98.4)
        t = Telemetry(airspeed_kts=72.0, altitude_ft=1750.0, pitch_deg=3.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=-800.0, agl_m=120.0)  # ~400 ft AGL, above line
        for _ in range(60):
            act = ctl.compute(t, tg, 0.05)
        self.assertLess(act.throttle, 0.05,
                        "Above the line, short final or not: no rescue power.")

    def test_explicit_idle_is_absolute(self):
        # DECELERATE/FLARE command idle explicitly; nothing overrides it
        # anymore — the floor that used to is gone.
        ctl = _ctl()
        tg = Targets(heading_deg=90.0, altitude_ft=5700.0, airspeed_kts=133.0,
                     throttle=0.0, stall_floor_kts=98.4)
        t = Telemetry(airspeed_kts=90.0, altitude_ft=5700.0, pitch_deg=2.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=0.0, agl_m=1300.0)
        for _ in range(60):
            act = ctl.compute(t, tg, 0.05)
        self.assertEqual(act.throttle, 0.0,
                         "Explicit idle stays idle, slow or not.")

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
        # A few ticks: the stick slew (2.0/s) means one 50 ms tick
        # can only move 0.1 — smoothness is the point.
        for _ in range(5):
            act = ctl.compute(t, tg, 0.05)
        self.assertLess(act.pitch, -0.1,
                        "With altitude to spare, diving for speed is fine.")


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

    def test_vs_target_is_slope_baseline_not_convergence(self):
        # ARCHITECTURE (Idriss, 2026-07-12): vs_target is now the slope
        # BASELINE sink only — the sink that flies PARALLEL to the line —
        # and no longer carries an altitude-convergence term. Rejoining the
        # line (steepen when high / pull up when low) moved into the
        # controller's glideslope PITCH step-and-check, verified below in
        # TestGlideslopePitchStepCheck. So the commanded sink is the SAME
        # whether we are on the slope or 300 ft above it — the convergence
        # is no longer baked into this number.
        on_slope = self._resolve(3000.0).altitude_ft
        vs_on = self._resolve(on_slope).vs_target_fpm
        vs_high = self._resolve(on_slope + 300.0).vs_target_fpm
        self.assertAlmostEqual(
            vs_high, vs_on, delta=1.0,
            msg="vs_target is the slope baseline; convergence is the "
                "controller's pitch step-and-check now, not this term.")

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


class TestEnergyLawEnvelopeFloors(unittest.TestCase):
    """The one energy law + three result-based envelope floors (Idriss,
    2026-07-12). Inputs are free; only stall, terrain-sink, and overspeed
    override at the edges."""

    def _one(self, *, actual_alt=2000.0, target_alt=2000.0, V=120.0,
             vs_fpm=-500.0, agl_ft=2000.0, v_max=250.0, ticks=80):
        # Run enough ticks for the throttle spool-walk to settle on the floor's
        # target (a single tick only nudges toward it).
        from uav.core.control.tecs_controller import TECSController
        ctl = TECSController()
        ctl._prev_throttle = 0.4
        tel = Telemetry(airspeed_kts=V, altitude_ft=actual_alt, pitch_deg=0.0,
                        roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                        lat_deg=0.0, lon_deg=0.0, agl_m=agl_ft * 0.3048,
                        vs_fpm=vs_fpm, groundspeed_kts=V)
        tg = Targets(heading_deg=90.0, altitude_ft=target_alt, airspeed_kts=130.0,
                     throttle=None, v_max_kts=v_max, gear_down=True, flap_ratio=1.0)
        act = None
        for _ in range(ticks):
            act = ctl.compute(tel, tg, 0.1)
        return act

    def test_stall_floor_powers_up_when_slow(self):
        # Below the protected speed (100 kt): full power, no nose-up.
        act = self._one(V=90.0)
        self.assertAlmostEqual(act.throttle, 1.0, places=3)
        self.assertLessEqual(act.pitch, 0.05, "must not command nose-up near stall")

    def test_on_speed_approach_does_not_trip_stall_floor(self):
        # A normal on-speed approach (114 kt) 480 ft high must NOT slam full
        # throttle — the bug that fought the plane up above the glideslope. With
        # the floor at 30 kts and throttle walking, power must not be pinned.
        act = self._one(V=114.0, actual_alt=1720.0, target_alt=1244.0, agl_ft=480.0)
        self.assertLess(act.throttle, 0.7,
                        "on-speed and 480 ft high must not command full power")

    def test_overspeed_cuts_throttle(self):
        # Above v_max: throttle cut, nose not pushed down.
        act = self._one(V=260.0, v_max=250.0)
        self.assertAlmostEqual(act.throttle, 0.0, places=3)

    def test_terrain_floor_arrests_sink_low(self):
        # Fast sink close to the ground: elevator commands nose-up.
        act = self._one(agl_ft=120.0, vs_fpm=-1600.0, V=120.0)
        self.assertGreater(act.pitch, 0.1, "sink floor must pull the nose up")

    def test_holds_altitude_band_no_runaway(self):
        # On target, healthy speed: throttle stays sane, elevator near neutral.
        act = self._one(actual_alt=2000.0, target_alt=2000.0, vs_fpm=0.0)
        self.assertTrue(0.0 <= act.throttle <= 1.0)
        self.assertTrue(-1.0 <= act.pitch <= 1.0)

    def test_elevator_cannot_slam_full_to_full(self):
        # The servo rate-limit: consecutive elevator commands can't jump the full
        # range in a tick (the PIO was reversing ±2.0 per sample). Drive a big
        # pitch-demand swing and check the per-tick change stays bounded.
        from uav.core.control.tecs_controller import TECSController, ELEVATOR_SLEW_PER_S
        ctl = TECSController()
        dt = 0.04
        prev = 0.0
        worst = 0.0
        for i in range(60):
            # alternate a wild demand to try to provoke a slam
            tgt_alt = 2000.0 if i % 2 == 0 else 3000.0
            tel = Telemetry(airspeed_kts=120.0, altitude_ft=2500.0,
                            pitch_deg=(15.0 if i % 2 else -15.0), roll_deg=0.0,
                            heading_deg=90.0, timestamp=0.0, agl_m=600.0,
                            vs_fpm=0.0, groundspeed_kts=120.0)
            tg = Targets(heading_deg=90.0, altitude_ft=tgt_alt, airspeed_kts=130.0,
                         throttle=None, on_glideslope=True, gear_down=True, flap_ratio=0.5)
            act = ctl.compute(tel, tg, dt)
            worst = max(worst, abs(act.pitch - prev))
            prev = act.pitch
        cap = ELEVATOR_SLEW_PER_S * dt + 1e-6
        self.assertLessEqual(worst, cap,
                             f"elevator jumped {worst:.3f}/tick > slew cap {cap:.3f}")

    def test_stall_speed_protection_powers_up_never_dives(self):
        # Below the protected speed (100 kt): FULL power, nose capped at level
        # (no pull-up into a deeper stall), and crucially NEVER commanded
        # nose-DOWN — diving a low, slow airplane is what plummeted it.
        from uav.core.control.tecs_controller import TECSController
        ctl = TECSController()
        thr, pitch = ctl._envelope_floors(0.3, 4.0, V_kts=90.0, vs_fpm=-300.0,
                                          agl_ft=800.0, vmax_kts=250.0)
        self.assertAlmostEqual(thr, 1.0, places=3, msg="must add full power")
        self.assertLessEqual(pitch, 0.0, "cap nose-up, don't deepen the stall")
        self.assertGreaterEqual(pitch, 0.0, "must NOT command nose-down (no dive)")


class TestGlideslopeAttitudeLaw(unittest.TestCase):
    """DESCENT / APPROACH: above 80 ft AGL a feedback loop walks pitch inside a
    flat window [-5, +1.5] to track the target (check→adjust→hold→loop); below
    80 ft, the gentle nose-up flare hold (Idriss, 2026-07-12)."""

    def _law(self, *, actual_alt, target_alt, agl_ft, ticks=12):
        from uav.core.control.tecs_controller import TECSController
        ctl = TECSController()
        tel = Telemetry(airspeed_kts=120.0, altitude_ft=actual_alt, pitch_deg=0.0,
                        roll_deg=0.0, heading_deg=270.0, timestamp=0.0,
                        agl_m=agl_ft * 0.3048, vs_fpm=-600.0, groundspeed_kts=120.0)
        tg = Targets(heading_deg=270.0, altitude_ft=target_alt, airspeed_kts=120.0,
                     throttle=None, on_glideslope=True, gear_down=True, flap_ratio=0.5)
        thr = theta = None
        for _ in range(ticks):     # dt=1.0 so each tick is one feedback check
            thr, theta = ctl._glideslope_law(tel, tg, agl_ft, 1.0)
        return thr, theta

    def test_above_target_noses_down_within_window(self):
        # 400 ft high: pitch walks DOWN to the -5 floor (room to come down),
        # never below it; throttle idles.
        thr, theta = self._law(actual_alt=2400.0, target_alt=2000.0, agl_ft=1000.0)
        self.assertLess(theta, 0.0, "high → nose down to descend")
        self.assertGreaterEqual(theta, -5.0, "must not exceed the -5 window floor")
        self.assertLess(thr, 0.22, "above the band → reduce power")

    def test_below_target_noses_up_and_powers(self):
        # Below target: pitch walks UP to the +1.5 cap; throttle adds power.
        thr, theta = self._law(actual_alt=1900.0, target_alt=2000.0, agl_ft=1000.0)
        self.assertGreater(theta, 0.0, "low → nose up")
        self.assertLessEqual(theta, 1.5, "must not exceed the +1.5 window cap")
        self.assertGreater(thr, 0.22, "below target → add power")

    def test_on_target_holds(self):
        # On target (within the deadband): pitch holds its seed, doesn't drift.
        thr, theta = self._law(actual_alt=2000.0, target_alt=2000.0, agl_ft=1000.0)
        self.assertGreaterEqual(theta, -5.0)
        self.assertLessEqual(theta, 1.5)

    def test_flare_region_idle_and_nose_up(self):
        # Below 10 ft AGL: committed to land — IDLE power and the nose stays up.
        thr, theta = self._law(actual_alt=1005.0, target_alt=1000.0, agl_ft=5.0)
        self.assertGreaterEqual(theta, 0.0, "near the ground the nose stays up")
        self.assertAlmostEqual(thr, 0.0, places=3, msg="idle to land")


if __name__ == "__main__":
    unittest.main()


class TestModeFlipAndBleed(unittest.TestCase):
    def test_pid_reset_on_coupling_flip(self):
        # Classic PIDs sit unused (integrals frozen) while coupled laws
        # fly; re-entering classic mode with stale state produced the
        # DECELERATE zoom and the BASE_LEG full-power climb.
        ctl = _ctl()
        coupled = Targets(heading_deg=90.0, altitude_ft=3000.0,
                          airspeed_kts=120.0, throttle=None,
                          throttle_for_alt=True)
        classic = Targets(heading_deg=90.0, altitude_ft=3000.0,
                          airspeed_kts=120.0, throttle=None,
                          throttle_for_alt=False)
        t = Telemetry(airspeed_kts=120.0, altitude_ft=3000.0, pitch_deg=0.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=0.0, agl_m=500.0)
        ctl.compute(t, coupled, 0.05)
        ctl.altitude_pid._integral = 1.0   # wound while unused
        ctl.airspeed_pid._integral = 1.0
        ctl.compute(t, classic, 0.05)      # flip → reset
        self.assertEqual(ctl.altitude_pid._integral, 0.0)
        self.assertEqual(ctl.airspeed_pid._integral, 0.0)

    def test_bleed_never_climbs_anywhere(self):
        # INBOUND traded 30 kts for +300 ft on the loop flight: pitching
        # up to bleed overspeed while already climbing just re-banks the
        # energy. Global rule now, not descent-only.
        ctl = _ctl()
        tg = Targets(heading_deg=90.0, altitude_ft=2741.0,
                     airspeed_kts=129.6, throttle=None,
                     throttle_for_alt=True)
        t = Telemetry(airspeed_kts=165.0, altitude_ft=2741.0, pitch_deg=5.0,
                      roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
                      vs_fpm=800.0, agl_m=460.0)
        act = ctl.compute(t, tg, 0.05)
        self.assertLessEqual(act.pitch, 0.04,
                             "Overspeed + climbing → no more nose-up.")
