from abc import ABC, abstractmethod


class Environment(ABC):

    @abstractmethod
    def step(self, u):
        pass

    @abstractmethod
    def reset(self, x):
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
