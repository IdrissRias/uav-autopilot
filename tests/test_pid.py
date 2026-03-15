import unittest

from uav.core.control.pid import PID


class TestPID(unittest.TestCase):
    def test_pid_response_sign(self):
        pid = PID(1.0, 0.0, 0.0)
        out = pid.update(10.0, 0.1)
        self.assertGreater(out, 0.0)

    def test_pid_integral_accumulates(self):
        pid = PID(0.0, 1.0, 0.0)
        out1 = pid.update(1.0, 0.5)
        out2 = pid.update(1.0, 0.5)
        self.assertGreater(out2, out1)


if __name__ == "__main__":
    unittest.main()
