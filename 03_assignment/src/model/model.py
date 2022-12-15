from enum import Enum

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy.random import randint, uniform

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


class DeepQNetwork(tf.keras.Model):

    def __init__(self, nx: int, nu: int, *args, **kwargs):
        self.inputs = keras.layers.Input(shape=(nx + nu, 1))
        self.state_out1 = keras.layers.Dense(16, activation="relu")(self.inputs)
        self.state_out2 = keras.layers.Dense(32, activation="relu")(self.state_out1)
        self.state_out3 = keras.layers.Dense(64, activation="relu")(self.state_out2)
        self.state_out4 = keras.layers.Dense(64, activation="relu")(self.state_out3)
        self.outputs = keras.layers.Dense(1)(self.state_out4)
        super().__init__(self.inputs, self.outputs)

    def call(self, inputs, training=None, mask=None):
        x = self.state_out1(inputs)
        x = self.state_out2(x)
        x = self.state_out3(x)
        x = self.state_out4(x)
        return self.outputs(x)

# nx = 2
# nu = 1
#
#
#
# Q.summary()
#
# # Set initial weights of targets equal to those of the critic
# Q_target.set_weights(Q.get_weights())
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
