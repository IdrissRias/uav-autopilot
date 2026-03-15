from __future__ import annotations

from abc import ABC, abstractmethod
from uav.sim.types import Telemetry, Actuators
from uav.core.guidance.base import Targets


class Controller(ABC):
    @abstractmethod
    def compute(self, telemetry: Telemetry, targets: Targets, dt: float) -> Actuators:
        raise NotImplementedError
