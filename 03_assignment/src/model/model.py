from __future__ import absolute_import, annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple

import tensorflow as tf
from common.converter import Converter
from tensorflow import keras
import numpy as np
from tensorflow.python.framework.ops import EagerTensor, Tensor

from tensorflow.python.ops.numpy_ops import np_config  # TODO: spostare dove ha senso farlo.
from train.experience_replay import Transition

np_config.enable_numpy_behavior()


class DQNType(Enum):
    """
    Enumerator that describes which types of Deep Q network have been implemented.
    At every element corresponds a dnn.
    """
    STANDARD = "standard"
    STATE = "state"
    DISCRETE = "discrete"


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
        if dqn_type == DQNType.STANDARD:
            return DeepQNetwork(nx, nu)
        elif dqn_type == DQNType.STATE:
            return NetworkWithOnlyState(nx)
        elif dqn_type == DQNType.DISCRETE:
            return DQNDiscrete(nx, nu)

    @staticmethod
    def load_model(dqn_type: DQNType, nx: int, nu: int, filename: str) -> DQNModel:
        if dqn_type == DQNType.STANDARD:
            model = DeepQNetwork(nx, nu)
        elif dqn_type == DQNType.STATE:
            model = NetworkWithOnlyState(nx)
        elif dqn_type == DQNType.DISCRETE:
            model = DQNDiscrete(nx, nu)
        else:
            raise ValueError(f"Support for model type [{dqn_type.value}] is not still available")
        model.load_weights(filename)
        return model

    @staticmethod
    def prepare_input(dqn_model: DQNModel, transition: Transition) -> EagerTensor:
        if isinstance(dqn_model, DeepQNetwork):
            return Converter.np2tf(transition.get_state_and_control_vector())
        elif isinstance(dqn_model, NetworkWithOnlyState) or isinstance(dqn_model, DQNDiscrete):
            return Converter.np2tf(transition.get_state_vector())

    @staticmethod
    def get_action_from_output_model(dqn_model: DQNModel, model_output) -> np.ndarray:
        if isinstance(dqn_model, DeepQNetwork):
            pass  # TODO: boooh, non so che cosa possa essere
            # return Converter.tf2np(model_output[])
        elif isinstance(dqn_model, NetworkWithOnlyState):
            return Converter.tf2np(model_output)
        elif isinstance(dqn_model, DQNDiscrete):
            output = np.argmax(Converter.tf2np(model_output))
            return output

    @staticmethod
    def prepare_minibatch(dqn_model: DQNModel, minibatch: List[Transition]) -> Tuple[Tensor, Tensor]:
        if isinstance(dqn_model, DeepQNetwork):
            pass
        elif isinstance(dqn_model, NetworkWithOnlyState) or isinstance(dqn_model, DQNDiscrete):
            np_minibatch = np.array([transition.get_state_vector() for transition in minibatch])
            np_next_minibatch = np.array([transition.get_next_state_vector() for transition in minibatch])
            return Converter.batch_np2tf(np_minibatch), Converter.batch_np2tf(np_next_minibatch)


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

    @property
    @abstractmethod
    def type(self) -> DQNType:
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

    @property
    def type(self) -> DQNType:
        return DQNType.STANDARD


class NetworkWithOnlyState(DQNModel):

    def __init__(self, nx: int):
        """
        The provided neural network, which consists of 6 layers.

        :param nx: the number of states.
        """
        inputs = keras.layers.Input(shape=(nx, 1))
        state_out1 = keras.layers.Dense(16, activation="relu")(inputs)
        state_out2 = keras.layers.Dense(32, activation="relu")(state_out1)
        state_out3 = keras.layers.Dense(64, activation="relu")(state_out2)
        state_out4 = keras.layers.Dense(64, activation="relu")(state_out3)
        state_out5 = keras.layers.Dense(32, activation="relu")(state_out4)
        flattened = keras.layers.Flatten(input_shape=(2, 32))(state_out5)
        action = keras.layers.Dense(1)(flattened)
        self._model = tf.keras.Model(inputs, action)

    @property
    def model(self) -> keras.Model:
        return self._model

    @property
    def type(self) -> DQNType:
        return DQNType.STATE


class DQNDiscrete(DQNModel):

    def __init__(self, nx: int, n_discrete_u: int):
        """
        A neural network that, given an input, returns as output n-discrete values, that are the discrete actions of
        the network.

        :param nx: the number of states
        :param n_discrete_u: the number of discretization appied on control
        """

        inputs = keras.layers.Input(shape=(nx))
        state_out1 = keras.layers.Dense(16, activation="relu")(inputs)
        state_out2 = keras.layers.Dense(32, activation="relu")(state_out1)
        state_out3 = keras.layers.Dense(64, activation="relu")(state_out2)
        state_out4 = keras.layers.Dense(64, activation="relu")(state_out3)
        state_out5 = keras.layers.Dense(32, activation="relu")(state_out4)
        action = keras.layers.Dense(n_discrete_u)(state_out5)
        softmaxed = keras.layers.Softmax()(action)
        self._model = tf.keras.Model(inputs, softmaxed)

    @property
    def model(self) -> keras.Model:
        return self._model

    @property
    def type(self) -> DQNType:
        return DQNType.DISCRETE
