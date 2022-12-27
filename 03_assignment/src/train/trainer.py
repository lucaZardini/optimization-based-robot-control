from __future__ import absolute_import, annotations

import tensorflow as tf
from environment.environment import Environment
from model.model import DQNModel
from keras.optimizers.optimizer_experimental.optimizer import Optimizer as KerasOptimizer
from train.experience_replay import ExperienceReplay


class Trainer:

    def __init__(self, critic: DQNModel, target: DQNModel, optimizer: KerasOptimizer, environment: Environment,
                 discount: float, batch_size: int, update_target_params: int, epsilon_start: float, epsilon_decay: float,
                 epsilon_min: float, buffer_size: int):
        """
        This class perform the training of a neural network.

        :param critic: the critic model
        :param target: the target model
        :param optimizer: the optimizer
        :param environment: the environment
        :param discount: the discount factor
        :param batch_size: the batch size
        :param update_target_params: every which steps updating the target parameters
        :param epsilon_start: the start value of the epsilon param
        :param epsilon_decay: decay value of the epsilon
        :param epsilon_min: minimum value of epsilon
        :param buffer_size: the buffer size of the experience replay
        """
        self.critic = critic
        self.target = target
        self.optimizer = optimizer
        self.discount = discount
        self.env = environment
        self.batch_size = batch_size
        self.update_target_params = update_target_params
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.buffer_size = buffer_size
        self.experience_replay = ExperienceReplay(self.buffer_size, self.batch_size)

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
        - the batch size
        - the epsilon values
        - the experience replay size
        """

        # Initialize critic already done
        # Initialize target
        self.target.initialize_weights(self.critic)

        # Initialize the environment
        self.env.reset()

        # Start the loop understanding the convergence condition
            # sample new epsilon
            # choose next action using epsilon greedy policy
            # take the action, get the cost and the next state (env.step(u))
            # save the transition in the experience replay buffer (self.experience_replay.append(transition))

            # if enough experience
                # sample random minibatch from experience replay
                # get the cost and call the update function

                # if step % update_target_params == 0
                    # update the target weights with the critic ones.



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
