from enum import Enum

import tensorflow as tf
from keras.optimizers.optimizer_experimental.optimizer import Optimizer as KerasOptimizer


class OptimizerType(Enum):
    ADAM = "adam"


class Optimizer:

    def __init__(self, qvalue_learning_rate: int = 1e-3):
        self.qvalue_learning_rate = qvalue_learning_rate

    def get_optimizer(self, optimizer_type: OptimizerType) -> KerasOptimizer:
        if optimizer_type == OptimizerType.ADAM:
            return tf.keras.optimizers.Adam(self.qvalue_learning_rate)
