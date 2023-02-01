from typing import Optional

from environment.environment import Environment
from environment.single_pendulum.pendulum_template import Pendulum
import numpy as np
import time


class SinglePendulum(Environment):
    '''
    Discrete Pendulum environment. Joint angle, velocity and torque are discretized
    with the specified steps. Joint velocity and torque are saturated.
    Gaussian noise can be added in the dynamics.
    Cost is -1 if the goal state has been reached, zero otherwise.
    '''

    def __init__(self, nq=51, nv=21, nu=11, vMax=5, uMax=5, dt=0.2, ndt=1, noise_stddev=0):
        """
        :param nq: the number of points joint angles
        :param nv: the number of points joint velocities
        :param nu: the number of points joint torques
        """
        self.pendulum = Pendulum(1, noise_stddev)
        self.pendulum.DT = dt
        self.pendulum.NDT = ndt
        # self.nq = nq  # Number of discretization steps for joint angle
        # self.nv = nv  # Number of discretization steps for joint velocity
        # self.vMax = vMax  # Max velocity (v in [-vmax,vmax])
        self._nu = nu  # Number of discretization steps for joint torque
        self.uMax = uMax  # Max torque (u in [-umax,umax])
        self.dt = dt  # time step
        # self.DQ = 2 * pi / nq  # discretization resolution for joint angle
        # self.DV = 2 * vMax / nv  # discretization resolution for joint velocity
        self.DU = 2 * uMax / nu  # discretization resolution for joint torque

    @property
    def nu(self) -> int:
        return 11

    @property
    def nx(self) -> int:
        return 2

    @property
    def goal(self):  # TODO
        return [0., 0.]

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

    def step(self, u,  x=None):
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

    # def plot_V_table(self, V):
    #     ''' Plot the given Value table V '''
    #     import matplotlib.pyplot as plt
    #     Q, DQ = np.meshgrid([self.d2cq(i) for i in range(self.nq)],
    #                         [self.d2cv(i) for i in range(self.nv)])
    #     plt.pcolormesh(Q, DQ, V.reshape((self.nv, self.nq)), cmap=plt.cm.get_cmap('Blues'))
    #     plt.colorbar()
    #     plt.title('V table')
    #     plt.xlabel("q")
    #     plt.ylabel("dq")
    #     plt.show()
    #
    # def plot_policy(self, pi):
    #     ''' Plot the given policy table pi '''
    #     import matplotlib.pyplot as plt
    #     Q, DQ = np.meshgrid([self.d2cq(i) for i in range(self.nq)],
    #                         [self.d2cv(i) for i in range(self.nv)])
    #     plt.pcolormesh(Q, DQ, pi.reshape((self.nv, self.nq)), cmap=plt.cm.get_cmap('RdBu'))
    #     plt.colorbar()
    #     plt.title('Policy')
    #     plt.xlabel("q")
    #     plt.ylabel("dq")
    #     plt.show()
    #
    # def plot_Q_table(self, Q):
    #     ''' Plot the given Q table '''
    #     import matplotlib.pyplot as plt
    #     X, U = np.meshgrid(range(Q.shape[0]), range(Q.shape[1]))
    #     plt.pcolormesh(X, U, Q.T, cmap=plt.cm.get_cmap('Blues'))
    #     plt.colorbar()
    #     plt.title('Q table')
    #     plt.xlabel("x")
    #     plt.ylabel("u")
    #     plt.show()
