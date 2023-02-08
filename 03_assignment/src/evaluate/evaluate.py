import time
from typing import Tuple

import numpy as np
from environment.environment import Environment
from model.model import DQNModel, DQNManager
from train.experience_replay import Transition


class Evaluator:

    def __init__(self, environment: Environment, max_iterations: int):
        self.environment = environment
        self.max_iterations = max_iterations

    def eval_to_get_best_model(self, model: DQNModel) -> Tuple[float, float]:
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

    def evaluate(self, model: DQNModel, starting_point: np.ndarray, display: bool = True):
        state = starting_point
        self.environment.reset(starting_point)
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
