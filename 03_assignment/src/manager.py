from environment.environment_type import EnvironmentType
from model.model import DQNType, DQNManager
from model.optimizer import OptimizerType, OptimizerManager
from train.trainer import Trainer


class Manager:

    def __init__(self, discount: float, learning_rate: int, optimizer_type: OptimizerType, critic_type: DQNType,
                 target_type: DQNType, nx: int, nu: int, env_type: EnvironmentType, update_target_params: int):
        """
        This class is the entry point of the project. It dispatches every possible functionality implemented.

        :param discount: the discount factor
        :param learning_rate: the learning rate
        :param optimizer_type: the type of optimizer used
        :param critic_type: the type of critic model
        :param target_type: the type of target modlel
        :param nx: number of states (for single pendulum 2, x and y ?)  TODO
        :param nu: number of actions (for single pendulum 1?)  TODO
        :param env_type: the type of environment to interact with
        :param update_target_params: every which steps update the target parameters
        """
        self.critic = DQNManager.get_model(critic_type, nx, nu)
        self.target = DQNManager.get_model(target_type, nx, nu)
        self.optimizer_manager = OptimizerManager(qvalue_learning_rate=learning_rate)
        self.optimizer = self.optimizer_manager.get_optimizer(optimizer_type)
        # self.environment = self.  # TODO
        self.trainer = Trainer(self.critic, self.target, self.optimizer, discount, update_target_params)  # TODO: environment

    def train(self):
        """
        Perform the training procedure of the models.
        """
        self.trainer.train()

    def load(self):  # TODO
        pass
