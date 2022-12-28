from abc import ABC, abstractmethod
from typing import Optional

from numpy.random import random


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

    @abstractmethod
    def c2du(self, u):
        pass

    def sample_random_start_episodes(self, episode_length: int) -> list:
        start_episodes: list = []
        for i in range(episode_length):
            state = random(self.nx)
            start_episodes.append(state)
        return start_episodes

    @property
    @abstractmethod
    def nx(self) -> int:
        pass

    @property
    @abstractmethod
    def nu(self) -> int:
        pass
