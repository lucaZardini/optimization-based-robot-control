from enum import Enum

import tensorflow as tf
from keras.optimizers.optimizer_experimental.optimizer import Optimizer as KerasOptimizer


class OptimizerType(Enum):
    """
    Enumerator that describes which types of optimizers have been implemented.
    At every element corresponds a different optimizer.
    """
    ADAM = "adam"


class OptimizerManager:
    """
    This class is used to return the selected optimizer.
    """
    def __init__(self, qvalue_learning_rate: int = 1e-3):
        """
        Initialize the optimizer manager.

        :param qvalue_learning_rate: the learning rate.
        """
        self.qvalue_learning_rate = qvalue_learning_rate

    def get_optimizer(self, optimizer_type: OptimizerType) -> KerasOptimizer:
        """
        Return the desired optimizer.

        :param optimizer_type: the optimizer type.
        :return: the desired optimizer.
        """
        if optimizer_type == OptimizerType.ADAM:
            return tf.keras.optimizers.Adam(self.qvalue_learning_rate)
