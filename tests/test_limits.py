import unittest

from uav.core.safety.limits import SafetyLimits
from uav.sim.types import Actuators


class TestSafetyLimits(unittest.TestCase):
    def test_clamp(self):
        limits = SafetyLimits(
            throttle_min=0.0,
            throttle_max=1.0,
            max_roll=0.35,
            max_pitch=0.25,
            max_yaw=0.2,
            brake_min=0.0,
            brake_max=1.0,
        )
        act = Actuators(throttle=1.5, roll=1.0, pitch=-1.0, yaw=-1.0, brake_ratio=2.0)
        clamped = limits.clamp(act)
        self.assertEqual(clamped.throttle, 1.0)
        self.assertEqual(clamped.roll, 0.35)
        self.assertEqual(clamped.pitch, -0.25)
        self.assertEqual(clamped.yaw, -0.2)
        self.assertEqual(clamped.brake_ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
