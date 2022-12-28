from __future__ import absolute_import, annotations

from abc import ABC, abstractmethod
from enum import Enum

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.ops.numpy_ops import np_config  # TODO: spostare dove ha senso farlo.

np_config.enable_numpy_behavior()


class DQNType(Enum):
    """
    Enumerator that describes which types of Deep Q network have been implemented.
    At every element corresponds a dnn.
    """
    STANDARD = "standard"


class DQNManager:
    """
    This class is used to return the selected network.
    """

    @staticmethod
    def get_model(dqn_type: DQNType, nx: int, nu: int) -> DQNModel:
        """
        This method is used to return the desired network selected among the possible ones defined in the enumerator above.

        :param dqn_type: the type of network
        :param nx: the number of states
        :param nu: the number of controls
        :return: the desired model
        """
        if dqn_type.value == DQNType.STANDARD:
            return DeepQNetwork(nx, nu)


class DQNModel(ABC):
    """
    This abstract class has been implemented to keep the code clean and for typing reasons.
    In particular:
    - all the dqn model implemented extends this class, inherit the implemented methods and should provide the abstract
      ones.
    - all the other components (e.g. Trainer) can work with this generic class, knowing that all the property/methods
      of the abstract class have to be implemented.
    """
    @property
    @abstractmethod
    def model(self) -> keras.Model:
        """
        Property that allows to return the desired model.

        :return: the desired model.
        """
        pass

    def save_weights(self, filename: str):
        """
        Save the weights of the trained method on a desired path.

        :param filename: the filename that will contain the trained weights.
        """
        self.model.save_weights(filename)

    def load_weights(self, filename: str):
        """
        Load weights

        :param filename: the filename that contains the trained weights to load.
        """
        self.model.load_weights(filename)

    def initialize_weights(self, critic: DQNModel):
        """
        Initialize the weights of the target model, given the critic one.

        :param critic: the critic model to copy the weights.
        """
        self.model.set_weights(critic.model.get_weights())


class DeepQNetwork(DQNModel):

    def __init__(self, nx: int, nu: int):
        """
        The provided neural network, which consists of 6 layers.

        :param nx: the number of states.
        :param nu: the number of controls.
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
