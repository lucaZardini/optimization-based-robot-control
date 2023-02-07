import math
from typing import List
import datetime
import time
import numpy as np
from environment.environment_type import EnvironmentType, EnvironmentManager
from simulate.simulate import Simulator
from model.model import DQNType, DQNManager
from model.optimizer import OptimizerType, OptimizerManager
from train.trainer import Trainer
import matplotlib.pyplot as plt
from environment.single_pendulum.single_pendulum import SinglePendulum


import logging
logger = logging.Logger("manager")
logger.addHandler(logging.StreamHandler())


class Manager:

    def __init__(self, discount: float, learning_rate: int, optimizer_type: OptimizerType, critic_type: DQNType,
                 target_type: DQNType, env_type: EnvironmentType, batch_size: int, update_target_params: int,
                 epsilon_start: float, epsilon_decay: float, epsilon_min: float, buffer_size: int,
                 max_iterations: int, episodes: int, experience_to_learn: int, plot_charts: bool = True,
                 save_params: bool = True):
        """
        This class is the entry point of the project. It dispatches every possible functionality implemented.

        :param discount: the discount factor
        :param learning_rate: the learning rate
        :param optimizer_type: the type of optimizer used
        :param critic_type: the type of critic model
        :param target_type: the type of target model
        :param env_type: the type of environment to interact with
        :param batch_size: the batch size
        :param update_target_params: every which steps update the target parameters
        :param epsilon_start: the start value of the epsilon param
        :param epsilon_decay: decay value of the epsilon
        :param epsilon_min: minimum value of epsilon
        :param buffer_size: the buffer size of the experience replay
        :param max_iterations: max iterations
        :param episodes: the number of episodes during training
        :param experience_to_learn: the number of experience collected to start learning
        """
        self.environment = EnvironmentManager.get_environment(env_type)
        self.critic = DQNManager.get_model(critic_type, self.environment.nx, self.environment.nu)
        self.target = DQNManager.get_model(target_type, self.environment.nx, self.environment.nu)
        self.optimizer_manager = OptimizerManager(qvalue_learning_rate=learning_rate)
        self.optimizer = self.optimizer_manager.get_optimizer(optimizer_type)
        self.trainer = Trainer(self.critic, self.target, self.optimizer, self.environment, discount, batch_size,
                               update_target_params, epsilon_start, epsilon_decay, epsilon_min, buffer_size,
                               max_iterations, episodes, experience_to_learn)
        self.plot_charts = plot_charts
        self.save_params = save_params

    def train(self, filename: str):
        """
        Perform the training procedure of the models.
        """
        logger.info(f"Starting to train the model [{self.critic.type.value}] and store the weights in [{filename}]")
        parameters = self.trainer.train(filename)
        if self.save_params:
            np.save("weight_models/single_pendulum/parameters.npy", parameters, allow_pickle=True)
        training_time = sum([time_episode for time_episode in parameters['time']])
        print(f"Total training time: {str(datetime.timedelta(seconds=training_time))}")
        cost_to_go = parameters['cost_to_go']
        loss = parameters['loss']
        discount_factor = parameters['discount_factor']
        episodes = parameters['episodes']
        if self.plot_charts:
            self.plot_cost_and_loss(cost_to_go, loss, discount_factor, episodes)

    @staticmethod
    def plot_cost_and_loss(cost_to_go: dict, loss: List[np.ndarray], discount_factor: int, episodes: int):
        plt.figure()
        average_costs = []
        for episode in range(episodes):
            episode_costs_to_go = cost_to_go[episode]
            episode_cost = 0
            for i, cost in enumerate(episode_costs_to_go):
                episode_cost += cost * discount_factor**i
            average_costs.append(episode_cost)

        plt.plot(np.arange(episodes), average_costs, "b")
        plt.gca().set_xlabel('Episodes')
        plt.gca().set_ylabel('Cost-to-go')
        plt.legend(["Cost to go"], loc='upper right')

        plt.figure()
        plt.plot(np.arange(len(loss)), loss, "b")
        plt.gca().set_xlabel('Action')
        plt.gca().set_ylabel('Loss')
        plt.legend(["Training loss"], loc='upper right')
        plt.show()

    def load(self, model_type: DQNType, filename: str):
        model = DQNManager.load_model(model_type, self.environment.nx, self.environment.nu, filename)
        simulator = Simulator(model, self.environment)
        state_history, action_history, total_time = simulator.simulate(self.environment.setup_state)
        if self.plot_charts:
            self.plot(state_history, action_history, total_time)

    def plot(self, state_history: List[np.ndarray], action_history: List[np.ndarray], total_time: time):
        if isinstance(self.environment, SinglePendulum):
            self.plot_single_pendulum_charts(state_history, action_history, total_time)
        else:
            self.plot_double_pendulum_charts(state_history, action_history, total_time)

    def plot_single_pendulum_charts(self, state_history: List[np.ndarray], action_history: List[np.ndarray], total_time: time):
        plt.figure()
        joint_angle = [x[0] for x in state_history]
        plt.plot(np.arange(len(state_history)), joint_angle, "b")
        plt.gca().set_xlabel('Action')
        plt.gca().set_ylabel('[rad]')
        plt.legend(["Joint angle"], loc='upper right')

        plt.figure()
        joint_velocity = [x[1] for x in state_history]
        plt.plot(np.arange(len(state_history)), joint_velocity, "b")
        plt.gca().set_xlabel('Action')
        plt.gca().set_ylabel('[rad/s]')
        plt.legend(["Joint velocity"], loc='upper right')

        plt.figure()
        plt.plot(np.arange(len(action_history)), action_history, "b")
        plt.gca().set_xlabel('Action')
        plt.gca().set_ylabel('Action value')
        plt.legend(["Discrete action value"], loc='upper right')

        action_continue_history = [self.environment.d2cu(u) for u in action_history]
        plt.figure()
        plt.plot(np.arange(len(action_continue_history)), action_continue_history, "b")
        plt.gca().set_xlabel('Action')
        plt.gca().set_ylabel('[Nm]')
        plt.legend(["Joint torque"], loc='upper right')

        print(f"Total simulation time: {str(datetime.timedelta(seconds=total_time))}")

        plt.show()

    def plot_double_pendulum_charts(self, state_history: List[np.ndarray], action_history: List[np.ndarray], total_time: time):
        plt.figure()
        first_joint_angle = [x[0] for x in state_history]
        plt.plot(np.arange(len(state_history)), first_joint_angle, "b")
        plt.gca().set_xlabel('Action')
        plt.gca().set_ylabel('[rad]')
        plt.legend(["First joint angle"], loc='upper right')

        plt.figure()
        second_joint_angle = [x[1] for x in state_history]
        plt.plot(np.arange(len(state_history)), second_joint_angle, "b")
        plt.gca().set_xlabel('Action')
        plt.gca().set_ylabel('[rad]')
        plt.legend(["Second joint angle"], loc='upper right')

        plt.figure()
        first_joint_velocity = [x[2] for x in state_history]
        plt.plot(np.arange(len(state_history)), first_joint_velocity, "b")
        plt.gca().set_xlabel('Action')
        plt.gca().set_ylabel('[rad/s]')
        plt.legend(["First joint velocity"], loc='upper right')

        plt.figure()
        second_joint_velocity = [x[3] for x in state_history]
        plt.plot(np.arange(len(state_history)), second_joint_velocity, "b")
        plt.gca().set_xlabel('Action')
        plt.gca().set_ylabel('[rad/s]')
        plt.legend(["Second joint velocity"], loc='upper right')

        plt.figure()
        first_joint_action = [u[0] for u in action_history]
        plt.plot(np.arange(len(action_history)), first_joint_action, "b")
        plt.gca().set_xlabel('Action')
        plt.gca().set_ylabel('Action value')
        plt.legend(["Discrete action value"], loc='upper right')

        action_continue_history = [self.environment.d2cu(u) for u in first_joint_action]
        plt.figure()
        plt.plot(np.arange(len(action_continue_history)), action_continue_history, "b")
        plt.gca().set_xlabel('Action')
        plt.gca().set_ylabel('[Nm]')
        plt.legend(["Joint torque"], loc='upper right')

        print(f"Total simulation time: {str(datetime.timedelta(seconds=total_time))}")

        plt.show()
