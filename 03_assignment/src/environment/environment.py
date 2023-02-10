from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

import numpy
import numpy as np


class Environment(ABC):

    @abstractmethod
    def step(self, u, x: Optional = None) -> Tuple[np.ndarray, float]:
        """
        Perform the step function.
        :param u: the action
        :param x: the current state
        :return: the next state and the cost
        """
        pass

    @abstractmethod
    def reset(self, x: Optional = None):
        """
        Reset the environment to state x. If not provided, choose random state.
        :param x: the state to initialize the environment
        """
        pass

    @abstractmethod
    def render(self):
        """
        Display the environment in the Geppetto displayer.
        """
        pass

    @abstractmethod
    def sample_random_start_episodes(self, episode_length: int) -> List[np.ndarray]:
        """
        Sample a specific number of random start episodes
        :param episode_length: the number of episodes to sample
        """
        pass

    @abstractmethod
    def sample_random_discrete_action(self, start: int, end: int) -> numpy.ndarray:
        """
        Sample a random discrete action
        :param start: start discrete value
        :param end: end discrete value
        :return: the sampled action
        """
        pass

    @abstractmethod
    def d2cu(self, u: np.ndarray) -> np.ndarray:
        """
        Convert action from discrete to continuous
        :param u: discrete action
        :return: continuous action
        """
        pass

    @property
    @abstractmethod
    def nx(self) -> int:
        """
        Number of states
        """
        pass

    @property
    @abstractmethod
    def nu(self) -> int:
        """
        Number of actions
        """
        pass

    @property
    @abstractmethod
    def setup_state(self) -> np.ndarray:
        """
        Goal state
        """
        pass

    @staticmethod
    def fixed_episodes_to_evaluate_model() -> List[numpy.ndarray]:
        """
        Generate fixed episode to choose the best model.
        :return: list of fixed episodes.
        """
        pass
