from __future__ import absolute_import, annotations

import tensorflow as tf
import numpy as np


class Converter:

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
