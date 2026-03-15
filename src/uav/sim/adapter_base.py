from __future__ import annotations

from abc import ABC, abstractmethod
from .types import Telemetry, Actuators


class SimAdapter(ABC):
    @abstractmethod
    def read_telemetry(self) -> Telemetry:
        raise NotImplementedError

    @abstractmethod
    def write_actuators(self, act: Actuators) -> None:
        raise NotImplementedError

    # Optional: not all adapters can do this.
    def reset_flight(self) -> bool:
        return False
