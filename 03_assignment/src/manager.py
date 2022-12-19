from environment.environment_type import EnvironmentType
from model.model import DQNType, DQNManager
from model.optimizer import OptimizerType, OptimizerManager
from train.trainer import Trainer


class Manager:

    def __init__(self, discount: float, learning_rate: int, optimizer_type: OptimizerType, critic_type: DQNType, target_type: DQNType, nx: int, nu: int, env_type: EnvironmentType):
        """

        :param discount:
        :param learning_rate:
        :param optimizer_type:
        :param critic_type:
        :param target_type:
        :param nx: number of states (for single pendulum 2, x and y ?)  TODO
        :param nu: number of actions (for single pendulum 1?)  TODO
        :param env_type:
        """
        self.critic = DQNManager.get_model(critic_type, nx, nu)
        self.target = DQNManager.get_model(target_type, nx, nu)
        self.optimizer_manager = OptimizerManager(qvalue_learning_rate=learning_rate)
        self.optimizer = self.optimizer_manager.get_optimizer(optimizer_type)
        # self.environment = self.  # TODO
        self.trainer = Trainer(self.critic, self.target, self.optimizer, discount)

    # TODO: visualizer (the environment)
    def train(self, xu_batch, cost_batch, xu_next_batch):  # TODO: typing. Understand if pass or create the batch in method.
        """

        :param xu_batch:
        :param cost_batch:
        :param xu_next_batch:
        :return:
        """
        # as far as I understood, train the model and every tot step launch an update
        self.trainer.update(xu_batch, cost_batch, xu_next_batch)

    def load(self):  # TODO
        pass
