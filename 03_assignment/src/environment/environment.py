from abc import ABC, abstractmethod
from typing import Optional


class Environment(ABC):

    @abstractmethod
    def step(self, u):
        pass

    @abstractmethod
    def reset(self, x: Optional = None):
        pass

    @abstractmethod
    def render(self):
        pass

    @property
    @abstractmethod
    def nx(self) -> int:
        pass

    @property
    @abstractmethod
    def nu(self) -> int:
        pass
