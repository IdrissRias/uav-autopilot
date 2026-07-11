"""
Attitude inner loop — the cascade every real autopilot uses.

TECS (and the lateral guidance) produce ANGLE demands: "fly 6° nose-up",
"bank 15° left". This loop turns an angle demand into a surface deflection
through two nested loops, which is what makes it smooth and stable:

    OUTER (P):   angle error      → a rate demand   (deg/s)
    INNER (P+D): rate error       → surface command (-1..1)

A bare "angle error → surface" law rings, because between the stick and the
angle sit an integration plus airframe lag. Closing the rate loop first gives
the outer loop a well-behaved thing to command. Gains scale with airspeed so
the response is the same slow or fast (a surface bites harder at speed).
Reference: PX4 / ArduPlane attitude controllers.
"""
from __future__ import annotations


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class AxisAttitude:
    """One axis (pitch or roll): hold a commanded ANGLE via angle→rate→surface."""

    def __init__(
        self,
        kp_angle: float,      # (deg/s) rate demanded per deg of angle error
        kp_rate: float,       # surface per (deg/s) of rate error
        kd_rate: float,       # surface damping per (deg/s^2) of rate change
        rate_limit: float,    # max commanded rate, deg/s
        surface_limit: float = 1.0,
        ref_speed_kts: float = 120.0,   # speed the gains were set at
    ) -> None:
        self.kp_angle = kp_angle
        self.kp_rate = kp_rate
        self.kd_rate = kd_rate
        self.rate_limit = rate_limit
        self.surface_limit = surface_limit
        self.ref_speed = ref_speed_kts
        self._prev_angle: float | None = None
        self._rate_filt = 0.0

    def reset(self) -> None:
        self._prev_angle = None
        self._rate_filt = 0.0

    def update(self, angle_dmd: float, angle: float, airspeed_kts: float,
               dt: float) -> float:
        if dt <= 0:
            return 0.0
        # Measured angular rate (filtered — the damping must arrive on time).
        if self._prev_angle is not None:
            raw = (angle - self._prev_angle) / dt
            self._rate_filt = 0.5 * raw + 0.5 * self._rate_filt
        self._prev_angle = angle
        rate = self._rate_filt

        # OUTER: angle error → rate demand (bounded).
        rate_dmd = _clamp((angle_dmd - angle) * self.kp_angle,
                          -self.rate_limit, self.rate_limit)

        # INNER: rate error → surface, with rate damping.
        rate_err = rate_dmd - rate
        surface = self.kp_rate * rate_err - self.kd_rate * rate

        # Airspeed scaling: a control surface is weaker slow, stronger fast, so
        # divide by (V/Vref)^2 to keep the closed-loop response constant. Clamp
        # so we never demand infinite deflection at a crawl.
        q_ratio = max(0.35, min(2.5, (max(airspeed_kts, 20.0) / self.ref_speed) ** 2))
        surface /= q_ratio
        return _clamp(surface, -self.surface_limit, self.surface_limit)


def default_pitch_axis() -> AxisAttitude:
    # 2.0 deg/s per deg error, capped ±8 deg/s; elevator 0.06 per deg/s error.
    return AxisAttitude(kp_angle=2.0, kp_rate=0.06, kd_rate=0.010,
                        rate_limit=8.0, surface_limit=1.0)


def default_roll_axis() -> AxisAttitude:
    # Snappier than pitch (ailerons are strong): 3 deg/s per deg, ±20 deg/s.
    return AxisAttitude(kp_angle=3.0, kp_rate=0.030, kd_rate=0.006,
                        rate_limit=20.0, surface_limit=1.0)
