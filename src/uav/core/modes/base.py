from __future__ import annotations

from abc import ABC, abstractmethod
from uav.sim.types import Telemetry, Targets


class Mode(ABC):
    name: str

    @abstractmethod
    def enter(self, ctx: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def step(self, ctx: dict, telemetry: Telemetry) -> Targets:
        raise NotImplementedError

    @abstractmethod
    def exit(self, ctx: dict) -> None:
        raise NotImplementedError
