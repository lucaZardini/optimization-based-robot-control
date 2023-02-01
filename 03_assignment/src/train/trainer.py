from __future__ import absolute_import, annotations

from numpy.random import uniform

import numpy as np
import tensorflow as tf
from environment.environment import Environment
from model.model import DQNModel, DQNManager
from keras.optimizers.optimizer_experimental.optimizer import Optimizer as KerasOptimizer
from train.experience_replay import ExperienceReplay, Transition
import logging

logger = logging.Logger("trainer")
logger.addHandler(logging.StreamHandler())


class Trainer:

    def __init__(self, critic: DQNModel, target: DQNModel, optimizer: KerasOptimizer, environment: Environment,
                 discount: float, batch_size: int, update_target_params: int, epsilon_start: float, epsilon_decay: float,
                 epsilon_min: float, buffer_size: int, max_iterations: int, episodes: int, experience_to_learn: int):
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
        :param max_iterations: max iterations
        :param episodes: the number of episodes to train the model
        :param experience_to_learn: the number of experience collected to start learning
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
        self.max_iterations = max_iterations
        self.episodes = episodes
        self.experience_to_learn = experience_to_learn

    def train(self, filename: str):
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

        start_episodes = self.env.sample_random_start_episodes(self.episodes)
        logger.info(f"Sampled the starting episodes. The length is [{len(start_episodes)}]")
        # Initialize critic already done
        # Initialize target
        self.target.initialize_weights(self.critic)

        # Initialize the environment
        total_steps = 0
        for number_episode, start_episode in enumerate(start_episodes):
            logger.info(f"Running episode [{number_episode}]")
            self.env.reset(start_episode)
            # self.experience_replay.setup()

            # sample new epsilon
            epsilon = self.epsilon_start
            u = np.random.random()  # first action
            state = start_episode
            transition = Transition(state, u, 0, 0)
            for iteration in range(self.max_iterations):
                if iteration % 100 == 0:
                    logger.info(f"Running iteration [{iteration}]")
                total_steps += 1
                epsilon *= self.epsilon_decay
                epsilon = max(epsilon, self.epsilon_min)
                # choose next action using epsilon greedy policy
                if uniform() < epsilon:
                    # logger.info("Chose random action")
                    u = np.random.randint(0, 11)
                else:
                    # logger.info("Chose model action")
                    model_input = DQNManager.prepare_input(self.critic, transition)
                    model_output = self.critic.model(model_input)
                    u = DQNManager.get_action_from_output_model(self.critic, model_output)

                # take the action, get the cost and the next state
                next_state, cost = self.env.step(u)
                # self.env.render()
                converged = cost == -1
                # save the transition in the experience replay buffer  # transition can be a class
                transition = Transition(state, u, cost, next_state)
                self.experience_replay.append(transition)

                # if self.experience_replay.size - 1 == self.experience_to_learn:
                #     logger.info("We have enough experience to train the model")
                # if enough experience
                if self.experience_replay.size > self.experience_to_learn:
                    # sample random minibatch from experience replay
                    # get the cost and call the update function
                    minibatch = self.experience_replay.sample_random_minibatch()
                    minibatch_cost = sum([transition.cost for transition in minibatch])
                    minibatch, next_minibatch = DQNManager.prepare_minibatch(self.critic, minibatch)
                    self.update(minibatch, minibatch_cost, next_minibatch)

                state = np.copy(next_state)
                if total_steps % self.update_target_params == 0:
                    self.target.initialize_weights(self.critic)

                if converged:
                    self.critic.save_weights(filename+str(total_steps)+".h5")
                    break

        self.critic.save_weights(filename)

    # The trainer should predict a new state and then put it in the experience replay. Then, every n step, update the
    # Q target and run the update function below with the batches.
    def update(self, xu_batch, cost_batch, xu_next_batch):  # TODO: update
        """
        Update the weights of the Q network using the specified batch of data

        :param xu_batch: the batch given the current x and u
        :param cost_batch: the cost of the batch given from the current x and u
        :param xu_next_batch: the batch obtained from the xu_batch applying the u get from the critic model
        :return:
        """
        # all inputs are tf tensors
        with tf.GradientTape() as tape:
            # logger.info("Updating the model...")
            # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
            # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched.
            # Tensors can be manually watched by invoking the watch method on this context manager.
            target_values = self.target.model(xu_next_batch, training=False)
            # Compute 1-step targets for the critic loss
            y = cost_batch + self.discount * target_values
            # Compute batch of Values associated to the sampled batch of states
            Q_value = self.critic.model(xu_batch, training=True)          # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
            Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))
            # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
        Q_grad = tape.gradient(Q_loss, self.critic.model.trainable_variables)
        # Update the critic backpropagating the gradients
        self.optimizer.apply_gradients(zip(Q_grad, self.critic.model.trainable_variables))
        # logger.info("Model parameters updated")
