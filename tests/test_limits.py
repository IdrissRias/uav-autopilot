import unittest

from uav.core.safety.limits import SafetyLimits
from uav.sim.types import Actuators


class TestSafetyLimits(unittest.TestCase):
    """SafetyLimits now does hardware-only clamping:
      * surfaces ∈ [-1, +1]
      * throttle, brake, flap ∈ [0, 1]
    No policy knobs — ignores any kwargs for API compat."""

    def test_hardware_clamps(self):
        limits = SafetyLimits()  # no config; any kwargs would be ignored
        act = Actuators(
            throttle=1.5, roll=1.5, pitch=-1.5, yaw=-1.5,
            brake_ratio=2.0, flap_ratio=1.7,
        )
        clamped = limits.clamp(act)
        self.assertEqual(clamped.throttle, 1.0)
        self.assertEqual(clamped.roll, 1.0)
        self.assertEqual(clamped.pitch, -1.0)
        self.assertEqual(clamped.yaw, -1.0)
        self.assertEqual(clamped.brake_ratio, 1.0)
        self.assertEqual(clamped.flap_ratio, 1.0)

    def test_kwargs_ignored(self):
        # Old call sites may still pass policy kwargs; should not raise.
        limits = SafetyLimits(throttle_max=0.5, max_roll=0.35)
        act = Actuators(throttle=0.9, roll=0.9, pitch=0.0, yaw=0.0, brake_ratio=0.0)
        clamped = limits.clamp(act)
        self.assertEqual(clamped.throttle, 0.9)  # NOT capped at 0.5
        self.assertEqual(clamped.roll, 0.9)       # NOT capped at 0.35

    def test_passthrough_in_range(self):
        limits = SafetyLimits()
        act = Actuators(throttle=0.7, roll=0.3, pitch=-0.2, yaw=0.1, brake_ratio=0.4)
        clamped = limits.clamp(act)
        self.assertAlmostEqual(clamped.throttle, 0.7)
        self.assertAlmostEqual(clamped.roll, 0.3)
        self.assertAlmostEqual(clamped.pitch, -0.2)
        self.assertAlmostEqual(clamped.yaw, 0.1)
        self.assertAlmostEqual(clamped.brake_ratio, 0.4)


if __name__ == "__main__":
    unittest.main()
