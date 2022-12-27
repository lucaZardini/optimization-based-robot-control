from environment.environment_type import EnvironmentType, EnvironmentManager
from model.model import DQNType, DQNManager
from model.optimizer import OptimizerType, OptimizerManager
from train.trainer import Trainer


class Manager:

    def __init__(self, discount: float, learning_rate: int, optimizer_type: OptimizerType, critic_type: DQNType,
                 target_type: DQNType, env_type: EnvironmentType, batch_size: int, update_target_params: int,
                 epsilon_start: float, epsilon_decay: float, epsilon_min: float, buffer_size: int):
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
        """
        self.environment = EnvironmentManager.get_environment(env_type)
        self.critic = DQNManager.get_model(critic_type, self.environment.nx, self.environment.nu)
        self.target = DQNManager.get_model(target_type, self.environment.nx, self.environment.nu)
        self.optimizer_manager = OptimizerManager(qvalue_learning_rate=learning_rate)
        self.optimizer = self.optimizer_manager.get_optimizer(optimizer_type)
        self.trainer = Trainer(self.critic, self.target, self.optimizer, self.environment, discount, batch_size,
                               update_target_params, epsilon_start, epsilon_decay, epsilon_min, buffer_size)

    def train(self):
        """
        Perform the training procedure of the models.
        """
        self.trainer.train()

    def load(self):  # TODO
        pass
