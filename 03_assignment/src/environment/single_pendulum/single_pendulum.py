from math import pi
from typing import Optional, List, Tuple


from environment.environment import Environment
from environment.single_pendulum.pendulum_template import Pendulum
import numpy as np
from numpy import random
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mlp_colors


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

    @staticmethod
    def weight_path() -> str:
        return "weight_models/single_pendulum/"

    # Functions used to create value and policy table. They are not used in other parts of code.
    def disc_state_from_2d_to_1d(self, disc_state: Tuple[int, int], discrete_number: int) -> int:
        return disc_state[0] + disc_state[1] * discrete_number

    def disc_state_2d_to_cont_state(self, disc_state: Tuple[int, int], discrete_number: int) -> np.ndarray:
        return np.array([self.d2c_angle(disc_state[0], discrete_number), self.d2c_velocity(disc_state[1], discrete_number)])

    def d2c_angle(self, angle_idx: int, discrete_number: int) -> float:
        discr_resolution = 2 * np.pi / discrete_number
        angle_idx = np.clip(angle_idx, 0, discrete_number - 1)
        return angle_idx * discr_resolution - np.pi + 0.5 * discr_resolution

    def d2c_velocity(self, vel_idx: int, discrete_number: int) -> float:
        discr_resolution = 2 * 8 / discrete_number
        vel_idx = np.clip(vel_idx, 0, discrete_number - 1) - (discrete_number - 1) / 2
        return vel_idx * discr_resolution

    def plot_v_table(self, v_table: np.ndarray, discrete_number: int):
        plt.figure()
        angles, vels = np.meshgrid(
            [self.d2c_angle(i, discrete_number) for i in range(discrete_number)],
            [self.d2c_velocity(i, discrete_number) for i in range(discrete_number)]
        )
        plt.pcolormesh(
            angles, vels, v_table.reshape((discrete_number, discrete_number)), cmap=plt.cm.get_cmap("Blues")
        )
        plt.colorbar(label="Cost to go (value)")
        plt.title("Value table")
        plt.xlabel("Joint angle [rad]")
        plt.ylabel("Joint velocity [rad/s]")
        plt.savefig("value_table.png")

    def plot_pi_table(self, pi_table: np.ndarray, discrete_number: int):
        plt.figure()
        angles, vels = np.meshgrid(
            [self.d2c_angle(i, discrete_number) for i in range(discrete_number)],
            [self.d2c_velocity(i, discrete_number) for i in range(discrete_number)]
        )
        # Make a discrete color map
        cmap = plt.cm.get_cmap("RdBu")
        bounds = [self.d2cu(dis_torque) for dis_torque in range(11)]
        norm = mlp_colors.BoundaryNorm(bounds, cmap.N)
        plt.pcolormesh(
            angles, vels, pi_table.reshape((discrete_number, discrete_number)),
            cmap=cmap, norm=norm
        )
        plt.colorbar(label="Torque [Nm]")
        plt.title("Policy table")
        plt.xlabel("Joint angle [rad]")
        plt.ylabel("Joint velocity [rad/s]")
        plt.savefig("policy_table.png")
        plt.show()
