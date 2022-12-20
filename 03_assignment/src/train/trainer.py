from __future__ import absolute_import, annotations

import tensorflow as tf
from model.model import DQNModel
from keras.optimizers.optimizer_experimental.optimizer import Optimizer as KerasOptimizer


class Trainer:

    def __init__(self, critic: DQNModel, target: DQNModel, optimizer: KerasOptimizer, discount: float,
                 update_target_params: int):
        """
        This class perform the training of a neural network.

        :param critic: the critic model
        :param target: the target model
        :param optimizer: the optimizer
        :param discount: the discount factor
        :param update_target_params: every which steps updating the target parameters
        """
        self.critic = critic
        self.target = target
        self.optimizer = optimizer
        self.discount = discount
        # self.env = env  # TODO

    def train(self):
        """
        This method implement the algorithm seen in class, the one used to train a model with reinforcement larning.
        The class already has everything it needs:
        - the critic model (The Q model)
        - the target model (The Q target model)
        - the optimizer
        - the environment to interact with
        - the discount factor
        - the factor that describes at which steps updating the target parameters
        """
        pass

    # The trainer should predict a new state and then put it in the experience replay. Then, every n step, update the
    # Q target and run the update function below with the batches.
    def update(self, xu_batch, cost_batch, xu_next_batch):
        """
        Update the weights of the Q network using the specified batch of data

        :param xu_batch:
        :param cost_batch:
        :param xu_next_batch:
        :return:
        """
        # all inputs are tf tensors
        with tf.GradientTape() as tape:
            # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
            # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched.
            # Tensors can be manually watched by invoking the watch method on this context manager.
            target_values = self.target.model(xu_next_batch, training=True)
            # Compute 1-step targets for the critic loss
            y = cost_batch + self.discount * target_values
            # Compute batch of Values associated to the sampled batch of states
            Q_value = self.critic.model(xu_batch, training=True)
            # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
            Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))
            # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
        Q_grad = tape.gradient(Q_loss, self.critic.model.trainable_variables)
        # Update the critic backpropagating the gradients
        self.optimizer.apply_gradients(zip(Q_grad, self.critic.model.trainable_variables))
