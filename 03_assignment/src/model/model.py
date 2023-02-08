from __future__ import absolute_import, annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple

import tensorflow as tf
from common.converter import Converter
from environment.double_pendulum.double_pendulum_template import DoublePendulum
from environment.environment import Environment
from environment.single_pendulum.single_pendulum import SinglePendulum
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
        if dqn_type == DQNType.DISCRETE:
            return DQNDiscrete(nx, nu)

    @staticmethod
    def load_model(dqn_type: DQNType, nx: int, nu: int, filename: str) -> DQNModel:
        if dqn_type == DQNType.DISCRETE:
            model = DQNDiscrete(nx, nu)
        else:
            raise ValueError(f"Support for model type [{dqn_type.value}] is not still available")
        model.load_weights(filename)
        return model

    @staticmethod
    def prepare_input(dqn_model: DQNModel, transition: Transition) -> EagerTensor: # convert the input (i.e. variable "state" from numpy type to tensorfloe type)
        if isinstance(dqn_model, DQNDiscrete):
            return Converter.np2tf(transition.get_state_vector())

    @staticmethod
    def get_action_from_output_model(dqn_model: DQNModel, model_output, env: Environment) -> np.ndarray: # pass the index where there is the best action
        if isinstance(dqn_model, DQNDiscrete):
            if isinstance(env, SinglePendulum):
                return Converter.tf2np(tf.argmin(model_output, axis=1, name=None).astype(np.float32))
            elif isinstance(env, DoublePendulum):
                return np.array([Converter.tf2np(tf.argmin(model_output, axis=1, name=None).astype(np.float32)), 0.]) # 0 for underactuated joint

    @staticmethod
    def prepare_minibatch(dqn_model: DQNModel, minibatch: List[Transition], env: Environment) -> Tuple[Tensor, Tensor, Tensor, Tensor]: # convert the minibatch from numpy type to tensorflow type
        if isinstance(dqn_model, DQNDiscrete):
            np_minibatch = np.array([transition.get_state_vector() for transition in minibatch])
            np_next_minibatch = np.array([transition.get_next_state_vector() for transition in minibatch])
            if isinstance(env, SinglePendulum):
                actions = np.array([transition.action for transition in minibatch])
            elif isinstance(env, DoublePendulum):
                actions = np.array([transition.action[0] for transition in minibatch]) # for double-pendulum it takes the action only for 1st joint
            else:
                raise ValueError(f"No preparation of action env for environment {env}")
            cost = np.array([transition.cost for transition in minibatch])
            return Converter.batch_np2tf(np_minibatch), Converter.batch_np2tf(cost), \
                   Converter.batch_np2tf(np_next_minibatch), Converter.batch_np2tf(actions)


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
        state_out4 = keras.layers.Dense(32, activation="relu")(state_out3)
        action = keras.layers.Dense(n_discrete_u)(state_out4)
        self._model = tf.keras.Model(inputs, action)

    @property
    def model(self) -> keras.Model:
        return self._model

    @property
    def type(self) -> DQNType:
        return DQNType.DISCRETE
