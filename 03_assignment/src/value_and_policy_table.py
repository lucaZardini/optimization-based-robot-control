import argparse

import tensorflow as tf
import numpy as np
from environment.single_pendulum.single_pendulum import SinglePendulum
from model.model import DQNModel, DQNManager, DQNType


class SinglePendulumValuePolicyTables:

    def __init__(self, weight_path: str):
        self.discrete_number = 80
        self.env = SinglePendulum()
        self.model = self._get_model(weight_path)

    def _get_model(self, weight_path) -> DQNModel:
        model = DQNManager.load_model(DQNType.DISCRETE, self.env.nx, self.env.nu, weight_path)
        return model

    def compute_value_and_policy_table(self):
        v_table = np.zeros(self.discrete_number ** 2)
        pi_table = np.zeros(self.discrete_number ** 2)

        states_idx = []
        states = []
        for angle in range(self.discrete_number):
            for velocity in range(self.discrete_number):
                pair = angle, velocity
                states_idx.append(self.env.disc_state_from_2d_to_1d(pair, self.discrete_number))
                states.append(self.env.disc_state_2d_to_cont_state(pair, self.discrete_number))

        states_tf = tf.convert_to_tensor(states)
        q_table = self.model.model(states_tf, training=False)

        for state_idx, q_values in zip(states_idx, q_table):
            v_table[state_idx] = np.min(q_values)
            u_best = np.where(q_values == v_table[state_idx])[0]

            if u_best[0] > self.env.c2du(0.0):
                pi_table[state_idx] = self.env.d2cu(u_best[-1])
            elif u_best[-1] < self.env.c2du(0.0):
                pi_table[state_idx] = self.env.d2cu(u_best[0])
            else:
                pi_table[state_idx] = self.env.d2cu(u_best[int(u_best.shape[0] / 2)])

        return v_table, pi_table

    def plot_v_and_pi_tables(self, v_table: np.ndarray, pi_table: np.ndarray):
        self.env.plot_v_table(v_table, self.discrete_number)
        self.env.plot_pi_table(pi_table, self.discrete_number)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Plot training info")
    arg_parser.add_argument("--weight-path", type=str, required=True, help="The path to the parameters file")
    args = arg_parser.parse_args()

    sp_value_policy = SinglePendulumValuePolicyTables(args.weight_path)
    value_table, policy_table = sp_value_policy.compute_value_and_policy_table()
    sp_value_policy.plot_v_and_pi_tables(value_table, policy_table)
