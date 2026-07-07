"""Landing commitment: above the gate the plane negotiates, below it,
it executes. Modeled on autoland practice — flare latch at ~50 ft radio
altitude with throttle retard, gated by stabilized-approach criteria,
sink-rate ratchet so nothing near the ground ever commands up, and an
emergency arrest so committed never means careless.
"""
from __future__ import annotations

import unittest

from uav.core.flight_engine import FlightEngine
from uav.nav.flight_plan_v2 import plan_path
from uav.sim.types import Targets, Telemetry


class TestLandingCommit(unittest.TestCase):
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
        self.g = self.ribbon.geometry
        self.kf_app = next(k for k in self.ribbon.keyframes
                           if k.name == "APPROACH")

    def _resolve(self, agl_ft: float, spd: float = 100.0,
                 vs: float = -600.0, hdg: float = 270.0,
                 lon_off: float = 0.004):
        """Resolve APPROACH near the threshold. lon_off ~0.004° ≈ 0.16 nm."""
        alt_ft = 1380.0 + agl_ft
        self.engine._prev_targets = Targets(
            heading_deg=hdg, altitude_ft=alt_ft, airspeed_kts=spd,
            flap_ratio=1.0, gear_down=True,
        )
        t = Telemetry(
            airspeed_kts=spd, altitude_ft=alt_ft, pitch_deg=2.0,
            roll_deg=0.0, heading_deg=hdg, timestamp=0.0,
            lat_deg=47.40, lon_deg=-94.77 + lon_off,
            agl_m=agl_ft / 3.28084, vs_fpm=vs,
        )
        return self.engine._resolve(self.kf_app, t, self.ribbon)

    def test_latches_when_stable_low_and_close(self):
        out = self._resolve(agl_ft=40.0)
        self.assertTrue(self.engine._land_committed)
        self.assertEqual(out.throttle, 0.0, "Retard: idle, locked.")
        self.assertIsNone(out.stall_floor_kts, "No power surges after commit.")
        self.assertLessEqual(out.roll_limit, 0.10, "Wing-strike bank cap.")

    def test_no_latch_when_high(self):
        self._resolve(agl_ft=200.0)
        self.assertFalse(self.engine._land_committed)

    def test_no_latch_when_misaligned(self):
        self._resolve(agl_ft=40.0, hdg=230.0)  # 40° off runway
        self.assertFalse(self.engine._land_committed,
                         "Unstable at the gate → no commit (go-around rule).")

    def test_no_latch_when_sinking_fast(self):
        self._resolve(agl_ft=40.0, vs=-1400.0)
        self.assertFalse(self.engine._land_committed,
                         "Sink > 1000 fpm at the gate is not stabilized.")

    def test_no_latch_when_hot(self):
        self._resolve(agl_ft=40.0, spd=self.g.v_land + 40.0)
        self.assertFalse(self.engine._land_committed)

    def test_ratchet_never_resteepens(self):
        self._resolve(agl_ft=40.0)          # latch (curve ≈ -520)
        vs_20 = self._resolve(agl_ft=20.0).vs_target_fpm   # shallower (-320)
        vs_blip = self._resolve(agl_ft=35.0).vs_target_fpm  # AGL blips UP
        self.assertGreaterEqual(
            vs_blip, vs_20,
            "An AGL blip must not re-steepen the sink target — one way down."
        )

    def test_nothing_commands_up(self):
        self._resolve(agl_ft=40.0)
        out = self._resolve(agl_ft=5.0)
        self.assertLess(out.vs_target_fpm, 0.0,
                        "Committed VS target is always a descent.")

    def test_emergency_arrest_bypasses_ratchet(self):
        self._resolve(agl_ft=40.0)
        out = self._resolve(agl_ft=25.0, vs=-1300.0)
        self.assertGreaterEqual(out.vs_target_fpm, -200.0,
                                "Dangerous sink near the ground → full arrest.")

    def test_reset_clears_commitment(self):
        self._resolve(agl_ft=40.0)
        self.assertTrue(self.engine._land_committed)
        self.engine.reset()
        self.assertFalse(self.engine._land_committed)


if __name__ == "__main__":
    unittest.main()
