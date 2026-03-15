import time
import unittest

from uav.core.mode_manager import ModeManager
from uav.sim.types import Telemetry


class TestModeTransitions(unittest.TestCase):
    def setUp(self) -> None:
        self.ctx = {
            "controller": {"cruise_throttle": 0.5},
            "targets": {
                "target_alt_ft": 3000.0,
                "target_hdg_deg": 90.0,
                "target_airspeed_kts": 90.0,
            },
            "mode": {"auto_start": True, "has_destination": True},
            "takeoff": {},
            "climb": {},
            "airframe": {
                "speeds_kts": {"v_rotate": 55.0},
                "thresholds": {
                    "rotate_altitude_gain_ft": 200.0,
                    "climb_to_cruise_altitude_tolerance_ft": 100.0,
                    "land_transition_alt_ft": 200.0,
                },
            },
        }

    def test_start_mode(self):
        mgr = ModeManager(start_mode="CRUISE", ctx=self.ctx, auto_start=True, has_destination=True)
        tel = Telemetry(
            airspeed_kts=10.0,
            altitude_ft=2000.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            heading_deg=80.0,
            timestamp=time.time(),
        )
        mgr.step(tel, stale=False)
        self.assertEqual(mgr.name, "TAKEOFF")

    def test_abort_on_stale(self):
        mgr = ModeManager(start_mode="CRUISE", ctx=self.ctx, auto_start=True, has_destination=True)
        tel = Telemetry(
            airspeed_kts=100.0,
            altitude_ft=3000.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            heading_deg=80.0,
            timestamp=time.time(),
        )
        mgr.step(tel, stale=True)
        self.assertEqual(mgr.name, "ABORT")

    def test_climb_to_cruise(self):
        mgr = ModeManager(start_mode="CRUISE", ctx=self.ctx, auto_start=True, has_destination=True)
        tel = Telemetry(
            airspeed_kts=60.0,
            altitude_ft=2900.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            heading_deg=80.0,
            timestamp=time.time(),
        )
        mgr.step(tel, stale=False)
        self.assertEqual(mgr.name, "CLIMB")
        tel.altitude_ft = 3000.0
        mgr.step(tel, stale=False)
        self.assertEqual(mgr.name, "CRUISE")

    def test_cruise_to_approach_to_land(self):
        ctx = dict(self.ctx)
        ctx["mode"] = {"auto_start": True, "has_destination": True, "landing_requested": True}
        ctx["airframe"] = {
            "speeds_kts": {"v_rotate": 55.0},
            "thresholds": {
                "rotate_altitude_gain_ft": 200.0,
                "climb_to_cruise_altitude_tolerance_ft": 100.0,
                "land_transition_alt_ft": 200.0,
            },
        }
        mgr = ModeManager(start_mode="CRUISE", ctx=ctx, auto_start=True, has_destination=True)
        tel = Telemetry(
            airspeed_kts=100.0,
            altitude_ft=3000.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            heading_deg=80.0,
            timestamp=time.time(),
        )
        mgr.step(tel, stale=False)
        self.assertEqual(mgr.name, "CRUISE")
        mgr.step(tel, stale=False)
        self.assertEqual(mgr.name, "APPROACH")
        tel.altitude_ft = 150.0
        mgr.step(tel, stale=False)
        self.assertEqual(mgr.name, "LAND")


if __name__ == "__main__":
    unittest.main()
