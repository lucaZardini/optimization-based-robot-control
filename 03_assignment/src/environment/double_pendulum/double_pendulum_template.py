from math import pi
from typing import Optional

from environment.double_pendulum.robot_wrapper import RobotWrapper
from environment.double_pendulum.simulator import RobotSimulator
from environment.environment import Environment
from example_robot_data.robots_loader import load
import numpy as np
import pinocchio as pin
from numpy.random import random


class DoublePendulum(Environment):

    def __init__(self, robot_model: str = 'double_pendulum'):
        r = load(robot_model)
        self.robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
        self.simu = RobotSimulator(self.robot, use_viewer=True)
        self.underact = 0
        self.goal = np.concatenate((np.array([0., 0]), np.zeros(self.robot.nv)))
        self.dt = 0.005
        self.dx = np.zeros(2 * self.robot.nv)

    ''' System dynamics '''

    def cost(self) -> np.ndarray:  # TODO: capire come fare il costo
        return np.abs(np.sum(self.dx))

    def step(self, u, x=None):
        u = np.array([self.d2cu(u[0]), 0.])
        nq = self.robot.nq
        nv = self.robot.nv
        model = self.robot.model
        data = self.robot.data
        q = x[:nq]
        v = x[nq:]
        ddq = pin.aba(model, data, q, v, u)
        self.dx[nv:] = ddq
        v_mean = v + 0.5 * self.dt * ddq
        self.dx[:nv] = v_mean
        state = x + self.dt * self.dx
        cost = self.cost()
        return np.copy(state), cost

    def reset(self, x: Optional = None):
        if x is not None:
            self.step(np.zeros(2), x)
        else:
            self.step(np.copy(np.zeros(2), np.random.random(2)))

    @property
    def nx(self):
        return self.robot.nq + self.robot.nv  # state size

    @property
    def nu(self):
        return self.robot.na  # control size TODO

    def render(self):
        self.simu.display(self.dx[:self.robot.nq])

    def d2cu(self, iu):
        iu = np.clip(iu, 0, self.nu - 1) - (self.nu - 1) / 2
        return iu * 10/self.nu

    def sample_random_start_episodes(self, episode_length: int) -> list:
        start_episodes: list = []
        for i in range(episode_length):
            if i % 10 == 0:
                start_episodes.append(np.array([pi, pi, 0., 0.]))
            else:
                state = random(self.nx)
                start_episodes.append(state)
        return start_episodes

    def sample_random_discrete_action(self, start: int, end: int) -> list:
        return [np.random.randint(start, end), 0]

    @property
    def setup_state(self):
        return np.array([pi, pi, 0., 0.])
