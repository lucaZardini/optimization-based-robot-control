import random


class ExperienceReplay:

    def __init__(self, buffer_size: int, batch_size: int):
        self.buffer: list = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def append(self, transition):  # TODO: typing
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample_random_minibatch(self):
        if len(self.buffer) < self.batch_size:
            raise IndexError(f"The size of the buffer ({len(self.buffer)}) is lower than the batch size ({self.batch_size})")
        return random.sample(self.buffer, self.batch_size)
