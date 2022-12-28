from __future__ import absolute_import, annotations

import random
from typing import List


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

    def __init__(self, state, action, cost, next_state):
        self.state = state
        self.action = action
        self.cost = cost
        self.next_state = next_state
