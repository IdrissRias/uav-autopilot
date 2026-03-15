from __future__ import annotations

import time


class Timebase:
    def __init__(self, rate_hz: float) -> None:
        self.rate_hz = rate_hz
        self._last = time.time()

    def tick(self) -> float:
        now = time.time()
        dt = max(now - self._last, 1e-3)
        self._last = now
        return dt

    def sleep_to_rate(self, loop_start: float) -> None:
        period = 1.0 / self.rate_hz
        remaining = period - (time.time() - loop_start)
        if remaining > 0:
            time.sleep(remaining)
