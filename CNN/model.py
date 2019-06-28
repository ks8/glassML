# CNN TensorFlow model
import tensorflow as tf
from nn_utils import weight_variable, bias_variable, conv2d


def model(x, n_outputs):
    """
    Function that builds the graph for the neural network.
    :param x: Input data
    :param n_outputs: Number of output neurons
    :return: Output vector of network, keep probability, and weight matrices in the network
    """
    # First convolutional layer
    x_image = tf.reshape(x, [-1, 250, 250, 3])
    w_conv1 = weight_variable([10, 10, 3, 6], name="w_conv1")
    b_conv1 = bias_variable([6], name="b_conv1")
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

    # Second convolutional layer
    w_conv2 = weight_variable([5, 5, 6, 16], name="w_conv2")
    b_conv2 = bias_variable([16], name="b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_conv1, w_conv2) + b_conv2)

    # Fully connected layer
    w_fc1 = weight_variable([237 * 237 * 16, 80], name="w_fc1")
    b_fc1 = bias_variable([80], name="b_fc1")
    h_conv2_flat = tf.reshape(h_conv2, [-1, 237*237*16])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, w_fc1) + b_fc1)

    # Dropout on the fully connected layer
    dropout = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, rate=dropout)

    # Output layer
    w_fc2 = weight_variable([80, n_outputs], name="w_fc2")
    b_fc2 = bias_variable([n_outputs], name="b_fc2")
    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    # Returns the prediction and the dropout probability placeholder
    return y_conv, dropout, w_conv1, w_conv2, w_fc1, w_fc2
