from __future__ import absolute_import, annotations

import tensorflow as tf
import numpy as np


class Converter:
    """
    This class is able to convert values from and to numpy.
    """

    @staticmethod
    def np2tf(y: np.ndarray):
        """
        convert from numpy to tensorflow
        """
        out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
        return out

    @staticmethod
    def tf2np(y):  # TODO: guarda il tipo di  y
        """
        convert from tensorflow to numpy
        """
        return tf.squeeze(y).numpy()

    @staticmethod
    def batch_np2tf(y: np.ndarray):
        return tf.convert_to_tensor(y)
