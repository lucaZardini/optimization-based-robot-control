from __future__ import absolute_import, annotations

import random
from typing import List

import numpy as np


class ExperienceReplay:

    def __init__(self, buffer_size: int, batch_size: int):
        """
        The experience replay buffer
        :param buffer_size: the size of the buffer
        :param batch_size: the batch size
        """
        self.buffer: List[Transition] = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    @property
    def size(self) -> int:
        """
        Return the size of the buffer
        """
        return len(self.buffer)

    def append(self, transition: Transition):
        """
        Add new transition in the buffer
        :param transition: the new transition
        """
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample_random_minibatch(self):
        """
        Sample a random minibatch
        :return: the minibatch
        """
        if len(self.buffer) < self.batch_size:
            raise IndexError(f"The size of the buffer ({len(self.buffer)}) is lower than the batch size ({self.batch_size})")
        return random.sample(self.buffer, self.batch_size)

    def setup(self):
        """
        Initialize buffer
        """
        self.buffer = []

    def save_buffer(self, filename: str):
        """
        Save buffer to file
        :param filename: the filename
        """
        np.save(filename, self.buffer, allow_pickle=True)

    def load_buffer(self, filename: str):
        """
        Load buffer
        :param filename: the filename
        """
        self.buffer = np.load(filename, allow_pickle=True)


class Transition:

    def __init__(self, state: np.array, action: np.array, cost: float, next_state: np.array):
        """
        Single transition containing the state, action, cost and next state.
        :param state: state
        :param action: action
        :param cost: cost
        :param next_state: next_state
        """
        self.state = state
        self.action = action
        self.cost = cost
        self.next_state = next_state

    def get_state_vector(self) -> np.ndarray:
        """
        Return the state vector
        """
        return np.array([[x] for x in self.state])

    def get_next_state_vector(self) -> np.ndarray:
        """
        Return the next state vector
        """
        return np.array([[x] for x in self.next_state])

    def get_state_and_control_vector(self) -> np.ndarray:
        """
        Return state and constrol vector
        """
        state_vector = self.get_state_vector()
        return np.append(state_vector, [[x] for x in self.action], axis=0)
