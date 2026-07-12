"""Configure-early flap control: all drag comes out at the top of the
descent; only SPEED stages the notches. The VS-tracking pitch counters
the deployment lift the same tick, so nothing deploys low and nothing
balloons unanswered.

  speed > flap_safe            → nothing new deploys
  flap_safe ≥ speed > v_app+10 → up to HALF flaps
  speed ≤ v_app+10             → full scheduled setting
  once out, flaps stay out (monotonic) except hard overspeed retract
"""
from __future__ import annotations

import unittest

from uav.core.flight_engine import FlightEngine
from uav.nav.flight_plan_v2 import plan_path
from uav.sim.types import Telemetry, Targets


def _telem(alt_ft: float, lat: float, lon: float, spd: float = 110.0):
    return Telemetry(
        airspeed_kts=spd, altitude_ft=alt_ft, pitch_deg=-3.0, roll_deg=0.0,
        heading_deg=270.0, timestamp=0.0, lat_deg=lat, lon_deg=lon,
        agl_m=(alt_ft - 1380.0) / 3.28084, vs_fpm=-800.0,
    )


class TestConfigureEarlyFlaps(unittest.TestCase):
    def setUp(self):
        self.ribbon = plan_path(
            dep_lat=45.42, dep_lon=-91.77, dep_alt_ft=1100.0,
            dep_heading=10.0,
            dest_lat=47.40, dest_lon=-94.77, dest_alt_ft=1380.0,
            dest_rwy_heading=270.0,
            dest_threshold_lat=47.40, dest_threshold_lon=-94.77,
            cruise_alt_ft=5000.0, v_stall=98.4,
        )
        self.engine = FlightEngine({})
        self.kf = next(k for k in self.ribbon.keyframes
                       if k.name == "DESCENT")
        self.g = self.ribbon.geometry
        # Plane ~2 nm east of threshold, on the approach side.
        self.lat, self.lon = 47.40, -94.72

    def _resolve(self, spd: float, prev_flap: float = 0.0,
                 alt_ft: float = 3400.0):
        self.engine._prev_targets = Targets(
            heading_deg=270.0, altitude_ft=alt_ft, airspeed_kts=spd,
            flap_ratio=prev_flap, gear_down=False,
        )
        t = _telem(alt_ft, self.lat, self.lon, spd=spd)
        return self.engine._resolve(self.kf, t, self.ribbon)

    def test_half_flaps_the_moment_speed_is_legal(self):
        # Descent entry: below flap_safe but above full-flap speed.
        out = self._resolve(spd=self.g.flap_safe_kts - 3.0)
        self.assertEqual(out.flap_ratio, 0.5,
                         "Half flaps deploy immediately at descent entry.")

    def test_half_flaps_is_the_cap_on_the_approach(self):
        # Config now uses HALF flaps on the glideslope (keyframe flap_ratio=0.5),
        # so even below full-flap speed it holds at half, never full. (Idriss,
        # 2026-07-12: the airframe floated on full flaps at approach speed.)
        out = self._resolve(spd=self.g.v_approach + 5.0, prev_flap=0.5)
        self.assertEqual(out.flap_ratio, 0.5,
                         "Half flaps is the configured cap on the approach.")

    def test_nothing_new_deploys_when_fast(self):
        out = self._resolve(spd=self.g.flap_safe_kts + 10.0)
        self.assertEqual(out.flap_ratio, 0.0,
                         "Above flap_safe nothing new deploys.")

    def test_flaps_monotonic_never_retract_in_descent(self):
        # Speed rose back above the full-flap gate: keep what's out.
        out = self._resolve(spd=self.g.v_approach + 15.0, prev_flap=1.0)
        self.assertEqual(out.flap_ratio, 1.0,
                         "Deployed flaps stay out (no low config churn).")

    def test_hard_overspeed_still_retracts(self):
        out = self._resolve(spd=self.g.flap_safe_kts * 1.08 + 5.0,
                            prev_flap=0.5)
        self.assertEqual(out.flap_ratio, 0.0,
                         "Structural overspeed pulls flaps in, always.")

    def test_gear_leads_with_the_flaps(self):
        out = self._resolve(spd=self.g.flap_safe_kts - 3.0)
        self.assertTrue(out.gear_down,
                        "Gear is part of the top-of-descent configuration.")

    def test_descent_pitch_is_vs_tracked(self):
        # The counter for the deployment balloon: pitch flies a sink
        # rate, so lift spikes are cancelled the same tick.
        out = self._resolve(spd=115.0)
        self.assertIsNotNone(out.vs_target_fpm,
                             "Glideslope descent commands a sink rate.")
        self.assertLess(out.vs_target_fpm, -200.0)

    def test_flare_keeps_half_flaps(self):
        # Half-flap config throughout the approach and flare (no lift spike from
        # a 0.5 -> 1.0 deployment at the flare).
        kf_flare = next(k for k in self.ribbon.keyframes
                        if k.name == "FLARE")
        self.engine._prev_targets = Targets(
            heading_deg=270.0, altitude_ft=1420.0, airspeed_kts=100.0,
            flap_ratio=0.5, gear_down=True,
        )
        t = _telem(1410.0, 47.40, -94.7705, spd=100.0)
        out = self.engine._resolve(kf_flare, t, self.ribbon)
        self.assertEqual(out.flap_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
