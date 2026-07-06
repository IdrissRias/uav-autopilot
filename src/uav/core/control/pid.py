from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    integral_limit: float = 1.0
    derivative_filter: float = 0.3  # low-pass alpha: 0=ignore new, 1=no filter

    def __post_init__(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement: float | None = None
        self._filtered_derivative = 0.0
        self._initialized = False

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_measurement: float | None = None
        self._filtered_derivative = 0.0
        self._initialized = False

    def update(self, error: float, dt: float, measurement: float | None = None) -> float:
        """When `measurement` is given, the derivative acts on the (negated)
        measurement instead of the error. Same value in steady tracking, but
        immune to derivative kick: a target step (keyframe transition) spikes
        d(error)/dt while d(measurement)/dt stays smooth."""
        if dt <= 0.0:
            return 0.0

        if not self._initialized:
            self._prev_error = error
            self._prev_measurement = measurement
            self._initialized = True

        self._integral += error * dt
        if self._integral > self.integral_limit:
            self._integral = self.integral_limit
        elif self._integral < -self.integral_limit:
            self._integral = -self.integral_limit

        if measurement is not None and self._prev_measurement is not None:
            raw_derivative = -(measurement - self._prev_measurement) / dt
        else:
            raw_derivative = (error - self._prev_error) / dt
        # Low-pass filter on derivative to suppress noise spikes.
        # alpha=0.3 means 30% new sample + 70% previous → smooth but responsive.
        alpha = self.derivative_filter
        self._filtered_derivative = (
            alpha * raw_derivative + (1.0 - alpha) * self._filtered_derivative
        )
        self._prev_error = error
        self._prev_measurement = measurement

        return (
            self.kp * error
            + self.ki * self._integral
            + self.kd * self._filtered_derivative
        )
