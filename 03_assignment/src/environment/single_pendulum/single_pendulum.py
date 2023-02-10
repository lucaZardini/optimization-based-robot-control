from math import pi
from typing import Optional, List

from environment.environment import Environment
from environment.single_pendulum.pendulum_template import Pendulum
import numpy as np
from numpy import random
import time


class SinglePendulum(Environment):
    '''
    Discrete Pendulum environment. Joint angle, velocity and torque are discretized
    with the specified steps. Joint velocity and torque are saturated.
    Gaussian noise can be added in the dynamics.
    Cost is -1 if the goal state has been reached, zero otherwise.
    '''

    def __init__(self, nu=11, uMax=2, dt=0.05, ndt=1, noise_stddev=0):
        """
        :param nu: the number of points joint torques
        """
        self.pendulum = Pendulum(1, noise_stddev)
        self.pendulum.DT = dt
        self.pendulum.NDT = ndt
        self._nu = nu  # Number of discretization steps for joint torque
        self.uMax = uMax  # Max torque (u in [-umax,umax])
        self.dt = dt  # time step
        self.DU = 2 * uMax / nu  # discretization resolution for joint torque

    @property
    def nu(self) -> int:
        return 11

    @property
    def nx(self) -> int:
        return 2

    @property
    def setup_state(self):
        return np.array([pi, 0.])

    def c2du(self, u):
        u = np.clip(u, -self.uMax + 1e-3, self.uMax - 1e-3)
        return int(np.floor((u + self.uMax) / self.DU))

    def d2cu(self, iu):
        iu = np.clip(iu, 0, self._nu - 1) - (self._nu - 1) / 2
        return iu * self.DU

    def reset(self, x: Optional = None):
        if x is None:
            # Initialize to basso e zero velocità.
            self.x = np.copy(np.random.random(2))
        else:
            self.x = np.copy(x)
        return self.x

    def step(self, u,  x=None): # convert the state from discrete to continuous
        u = self.d2cu(u)
        self.x, cost = self.pendulum.dynamics(self.x, u)
        return np.copy(self.x), cost

    def render(self):
        q = self.x[0]
        self.pendulum.display(np.array([q, ]))
        time.sleep(self.pendulum.DT)

    def dynamics(self, x, iu):
        u = self.d2cu(iu)
        self.xc, _ = self.pendulum.dynamics(x, u)
        return self.xc

    def sample_random_start_episodes(self, episode_length: int) -> list:
        start_episodes: list = []
        for i in range(episode_length):
            joint_angle = random.uniform(low=-pi, high=pi, size=1)
            joint_velocity = random.uniform(low=-self.pendulum.vmax, high=self.pendulum.vmax, size=1)
            state = np.array([joint_angle[0], joint_velocity[0]])
            start_episodes.append(state)
        return start_episodes

    def sample_random_discrete_action(self, start: int, end: int) -> np.ndarray:
        return np.random.randint(start, end)

    @staticmethod
    def fixed_episodes_to_evaluate_model() -> List[np.ndarray]:
        return [
            np.array([pi, 0]),
            np.array([pi/2, 0]),
            np.array([-pi/2, 0]),
            np.array([-pi/4, 0.5]),
            np.array([pi/4, +0.5]),
            np.array([pi*3/4, 1]),
            np.array([-pi*3/4, -1])
        ]
