from __future__ import absolute_import, annotations

import random
from typing import List

import numpy as np


class ExperienceReplay:

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer: List[Transition] = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    @property
    def size(self) -> int:
        return len(self.buffer)

    def append(self, transition: Transition):
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample_random_minibatch(self):
        if len(self.buffer) < self.batch_size:
            raise IndexError(f"The size of the buffer ({len(self.buffer)}) is lower than the batch size ({self.batch_size})")
        return random.sample(self.buffer, self.batch_size)


class Transition:

    def __init__(self, state: np.array, action: np.array, cost: int, next_state: np.array):
        self.state = state
        self.action = action
        self.cost = cost
        self.next_state = next_state

    def get_state_vector(self) -> np.ndarray:
        return np.array([[x] for x in self.state])

    def get_next_state_vector(self) -> np.ndarray:
        return np.array([[x] for x in self.next_state])

    def get_state_and_control_vector(self) -> np.ndarray:
        state_vector = self.get_state_vector()
        return np.append(state_vector, [[x] for x in self.action], axis=0)
