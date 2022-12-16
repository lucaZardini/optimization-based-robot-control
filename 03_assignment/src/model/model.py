from __future__ import absolute_import, annotations

from abc import ABC, abstractmethod
from enum import Enum

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy.random import randint, uniform

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


class DQNType(Enum):
    STANDARD = "standard"


class DQNManager:

    @staticmethod
    def get_model(dqn_type: DQNType, nx: int, nu: int) -> DQNModel:
        if dqn_type.value == DQNType.STANDARD:
            return DeepQNetwork(nx, nu)


class DQNModel(ABC):

    @property
    @abstractmethod
    def model(self):
        pass


class DeepQNetwork(DQNModel):

    def __init__(self, nx: int, nu: int):
        """

        :param nx:
        :param nu:
        """
        inputs = keras.layers.Input(shape=(nx + nu, 1))
        state_out1 = keras.layers.Dense(16, activation="relu")(inputs)
        state_out2 = keras.layers.Dense(32, activation="relu")(state_out1)
        state_out3 = keras.layers.Dense(64, activation="relu")(state_out2)
        state_out4 = keras.layers.Dense(64, activation="relu")(state_out3)
        outputs = keras.layers.Dense(1)(state_out4)
        self._model = tf.keras.Model(inputs, outputs)

    @property
    def model(self):
        return self._model

    def initialize_weights(self, critic: DeepQNetwork):
        """

        :param critic:
        :return:
        """
        self.model.set_weights(critic.model.get_weights())

# nx = 2
# nu = 1
#
#
# Q.summary()
#
#
# w = Q.get_weights()
# for i in range(len(w)):
#    print("Shape Q weights layer", i, w[i].shape)
#
# for i in range(len(w)):
#    print("Norm Q weights layer", i, np.linalg.norm(w[i]))
#
# print("\nDouble the weights")
# for i in range(len(w)):
#    w[i] *= 2
# Q.set_weights(w)
#
# w = Q.get_weights()
# for i in range(len(w)):
#    print("Norm Q weights layer", i, np.linalg.norm(w[i]))
#
# print("\nSave NN weights to file (in HDF5)")
# Q.save_weights("namefile.h5")
#
# print("Load NN weights from file\n")
# Q_target.load_weights("namefile.h5")
#
# w = Q_target.get_weights()
# for i in range(len(w)):
#    print("Norm Q weights layer", i, np.linalg.norm(w[i]))
