from typing import List
import datetime
import time
import numpy as np
from environment.environment_type import EnvironmentType, EnvironmentManager
from evaluate.evaluate import Evaluator
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
                 max_iterations: int, episodes: int, experience_to_learn: int, update_critic: int,
                 max_iterations_eval: int, plot_charts: bool = True, save_params: bool = True):
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
        :param max_iterations_eval: the number of eval iterations
        """
        self.environment = EnvironmentManager.get_environment(env_type)
        self.critic = DQNManager.get_model(critic_type, self.environment.nx, self.environment.nu)
        self.target = DQNManager.get_model(target_type, self.environment.nx, self.environment.nu)
        self.optimizer_manager = OptimizerManager(qvalue_learning_rate=learning_rate)
        self.optimizer = self.optimizer_manager.get_optimizer(optimizer_type)
        self.evaluator = Evaluator(self.environment, max_iterations_eval)
        self.trainer = Trainer(self.critic, self.target, self.optimizer, self.environment, discount, batch_size,
                               update_target_params, epsilon_start, epsilon_decay, epsilon_min, buffer_size,
                               max_iterations, episodes, experience_to_learn, update_critic,
                               Evaluator(self.environment, max_iterations))
        self.plot_charts = plot_charts
        self.save_params = save_params

    def train(self, filename: str):
        """
        Perform the training procedure of the models.
        """
        logger.info(f"Starting to train the model [{self.critic.type.value}] and store the weights in [{filename}]")
        best_model, parameters = self.trainer.train(filename)
        if self.save_params:
            np.save(f"{self.environment.weight_path()}params.npy", parameters, allow_pickle=True)
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
        """
        Given the cost to go and the list of loss, plot the relative charts.
        :param cost_to_go: A dict with as key the number of episode and as value the list of the costs.
        :param loss: the list of loss of the training step
        :param discount_factor: the discount factor used to compute the cost-to-go
        :param episodes: the number of episodes
        """
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
        """
        Load the model and evaluate it.
        :param model_type: type of model
        :param filename: the filename
        """
        model = DQNManager.load_model(model_type, self.environment.nx, self.environment.nu, filename)
        random_episodes = self.environment.sample_random_start_episodes(10)
        # for random_episode in random_episodes:
        #     state_history, action_history, total_time, cost = self.evaluator.evaluate(model, random_episode)
        random_episode = self.environment.sample_random_start_episodes(1)[0]
        state_history, action_history, total_time, cost = self.evaluator.evaluate(model, random_episode)
        if self.plot_charts:
            self.plot(state_history, action_history, total_time)

    def plot(self, state_history: List[np.ndarray], action_history: List[np.ndarray], total_time: time):
        """
        Plot stats related to the evaluations
        :param state_history: list of states
        :param action_history: list of actions
        :param total_time: time of evaluation
        """
        if isinstance(self.environment, SinglePendulum):
            self.plot_single_pendulum_charts(state_history, action_history, total_time)
        else:
            self.plot_double_pendulum_charts(state_history, action_history, total_time)

    def plot_single_pendulum_charts(self, state_history: List[np.ndarray], action_history: List[np.ndarray], total_time: time):
        """
        Plot charts for the single pendulum
        :param state_history: list of states
        :param action_history: list of actions
        :param total_time: time of evaluation
        """
        plt.figure()
        joint_angle = [x[0] for x in state_history]
        plt.plot(np.arange(len(state_history)), joint_angle, "b")
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('[rad]')
        plt.legend(["Joint angle"], loc='upper right')
        # plt.savefig("joint-angle.png")

        plt.figure()
        joint_velocity = [x[1] for x in state_history]
        plt.plot(np.arange(len(state_history)), joint_velocity, "b")
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('[rad/s]')
        plt.legend(["Joint velocity"], loc='upper right')
        # plt.savefig("joint-velocity.png")

        plt.figure()
        plt.plot(np.arange(len(action_history)), action_history, "b")
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Action value')
        plt.legend(["Discrete action value"], loc='upper right')
        # plt.savefig("discrete-actions.png")

        action_continue_history = [self.environment.d2cu(u) for u in action_history]
        plt.figure()
        plt.plot(np.arange(len(action_continue_history)), action_continue_history, "b")
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('[Nm]')
        plt.legend(["Joint torque"], loc='upper right')
        # plt.savefig("joint-torque.png")

        print(f"Total simulation time: {str(datetime.timedelta(seconds=total_time))}")

        plt.show()

    def plot_double_pendulum_charts(self, state_history: List[np.ndarray], action_history: List[np.ndarray], total_time: time):
        """
        Plot charts for the double pendulum
        :param state_history: list of states
        :param action_history: list of actions
        :param total_time: time of evaluation
        """
        plt.figure()
        first_joint_angle = [x[0] for x in state_history]
        plt.plot(np.arange(len(state_history)), first_joint_angle, "b")
        second_joint_angle = [x[1] for x in state_history]
        plt.plot(np.arange(len(state_history)), second_joint_angle, "r")
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('[rad]')
        plt.legend(["First joint angle", "Second joint angle"], loc='upper right')
        # plt.savefig("dp_joint_angles.png")

        plt.figure()
        first_joint_velocity = [x[2] for x in state_history]
        plt.plot(np.arange(len(state_history)), first_joint_velocity, "b")
        second_joint_velocity = [x[3] for x in state_history]
        plt.plot(np.arange(len(state_history)), second_joint_velocity, "r")
        plt.gca().set_xlabel('Interations')
        plt.gca().set_ylabel('[rad/s]')
        plt.legend(["First joint velocity", "Second joint velocity"], loc='upper right')
        # plt.savefig("dp_joint_velocities.png")

        plt.figure()
        first_joint_action = [u[0] for u in action_history]
        plt.plot(np.arange(len(action_history)), first_joint_action, "b")
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('Action value')
        plt.legend(["Discrete action value"], loc='upper right')
        # plt.savefig("dp_action_discrete.png")

        action_continue_history = [self.environment.d2cu(u) for u in first_joint_action]
        plt.figure()
        plt.plot(np.arange(len(action_continue_history)), action_continue_history, "b")
        plt.gca().set_xlabel('Iterations')
        plt.gca().set_ylabel('[Nm]')
        plt.legend(["Joint torque"], loc='upper right')
        # plt.savefig("dp_torque.png")

        print(f"Total simulation time: {str(datetime.timedelta(seconds=total_time))}")

        plt.show()
