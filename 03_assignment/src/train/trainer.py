from __future__ import absolute_import, annotations

import tensorflow as tf
from model.model import DeepQNetwork
from model.optimizer import Optimizer


class Trainer:

    def update(self, critic: DeepQNetwork, target: DeepQNetwork, optimizer: Optimizer, xu_batch, cost_batch, xu_next_batch):
        """
        Update the weights of the Q network using the specified batch of data
        """
        # all inputs are tf tensors
        with tf.GradientTape() as tape:
            # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
            # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched.
            # Tensors can be manually watched by invoking the watch method on this context manager.
            target_values = target(xu_next_batch, training=True)
            # Compute 1-step targets for the critic loss
            y = cost_batch + DISCOUNT * target_values
            # Compute batch of Values associated to the sampled batch of states
            Q_value = critic(xu_batch, training=True)
            # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
            Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))
            # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
        Q_grad = tape.gradient(Q_loss, critic.trainable_variables)
        # Update the critic backpropagating the gradients
        optimizer.apply_gradients(zip(Q_grad, critic.trainable_variables))

