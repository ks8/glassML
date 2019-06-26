# Auxiliary functions for building CNN in TensorFlow
import tensorflow as tf


def conv2d(x, w):
    """
    Returns a 2D convolutional layer.
    :param x: Input data
    :param w: Weight matrix (kernel)
    :return: 2D convolution layer
    """
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')


def weight_variable(shape, name):
    """
    Generates a weight variable of a given shape.
    :param shape: Shape of weight variable
    :param name: Name of weight variable
    :return: Weight variable
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """
    Generates a bias variable of a given shape.
    :param shape: Shape of bias variable
    :param name: Name of bias variable
    :return: Bias variable
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)
