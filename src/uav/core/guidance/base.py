from __future__ import annotations

from abc import ABC, abstractmethod
from uav.sim.types import Telemetry, Targets


class Guidance(ABC):
    @abstractmethod
    def compute(self, telemetry: Telemetry, desired: Targets) -> Targets:
        raise NotImplementedError
