import numpy as np
from environment.environment import Environment
from model.model import DQNModel, DQNManager
from train.experience_replay import Transition


class Simulator:

    def __init__(self, model: DQNModel, environment: Environment):
        self.model = model
        self.environment = environment

    def simulate(self, starting_point: np.ndarray):
        converged = False
        state = starting_point
        self.environment.reset(starting_point)
        while not converged:
            transition = Transition(state, 0, 0, 0)
            model_input = DQNManager.prepare_input(self.model, transition)
            model_output = self.model.model(model_input, training=False)
            u = DQNManager.get_action_from_output_model(self.model, model_output, self.environment)
            # take the action, get the cost and the next state
            next_state, cost = self.environment.step(u, state)
            state = np.copy(next_state)
            self.environment.render()
