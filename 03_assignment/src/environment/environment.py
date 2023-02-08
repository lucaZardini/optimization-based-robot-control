from abc import ABC, abstractmethod
from typing import Optional, List

import numpy


class Environment(ABC):

    @abstractmethod
    def step(self, u, x: Optional = None):
        pass

    @abstractmethod
    def reset(self, x: Optional = None):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def sample_random_start_episodes(self, episode_length: int) -> list:
        pass

    @abstractmethod
    def sample_random_discrete_action(self, start: int, end: int) -> numpy.ndarray:
        pass

    @abstractmethod
    def d2cu(self, u):
        pass

    @property
    @abstractmethod
    def nx(self) -> int:
        pass

    @property
    @abstractmethod
    def nu(self) -> int:
        pass

    @property
    @abstractmethod
    def setup_state(self):
        pass

    @staticmethod
    def fixed_episodes_to_evaluate_model() -> List[numpy.ndarray]:
        pass
