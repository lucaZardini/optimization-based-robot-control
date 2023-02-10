import time
from typing import Tuple, List

import numpy as np
from environment.environment import Environment
from model.model import DQNModel, DQNManager
from train.experience_replay import Transition


class Evaluator:

    def __init__(self, environment: Environment, max_iterations: int):
        """
        Class used to evaluate models over an environment.

        :param environment: the environment
        :param max_iterations: the max number of iterations
        """
        self.environment = environment
        self.max_iterations = max_iterations

    def eval_to_get_best_model(self, model: DQNModel) -> Tuple[float, float]:
        """
        Function used in training phase to evaluate the model.
        :param model: the model to evaluate
        :return: the model cost and the evaluation time
        """
        total_cost = 0
        total_eval_time = None
        for episode in self.environment.fixed_episodes_to_evaluate_model():
            _, _, eval_time, cost = self.evaluate(model, episode, display=False)
            total_cost += cost
            if total_eval_time is None:
                total_eval_time = eval_time
            else:
                total_eval_time += eval_time
        return total_cost, total_eval_time

    def evaluate(self, model: DQNModel, starting_point: np.ndarray, display: bool = True) -> \
            Tuple[List[np.ndarray], List[np.ndarray], float, float]:
        """
        Evaluate a model on a single episode
        :param model: the model to evaluate
        :param starting_point: the starting point
        :param display: boolean to display or not the evaluation
        :return: list of states and actions, the eval time and the total cost
        """
        state = starting_point
        self.environment.reset(starting_point)
        if display:
            self.environment.render()
            time.sleep(1)
        state_history = []
        action_history = []
        total_cost = 0.
        start_time = time.time()
        for iteration in range(self.max_iterations):
            transition = Transition(state, 0, 0, 0)
            model_input = DQNManager.prepare_input(model, transition)
            model_output = model.model(model_input, training=False)
            u = DQNManager.get_action_from_output_model(model, model_output, self.environment)
            # take the action, get the cost and the next state
            state_history.append(state)
            action_history.append(u)
            next_state, cost = self.environment.step(u, state)
            total_cost += cost
            state = np.copy(next_state)
            if display:
                self.environment.render()

        return state_history, action_history, time.time() - start_time, total_cost
