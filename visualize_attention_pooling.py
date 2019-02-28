# Functions for visualizing attention pooling
# Import modules
import matplotlib.pyplot as plt
import numpy as np


def round_up_to_even(f):
    """
    Round up to the nearest even number
    :param f: Input number
    :return: Nearest even number greater than f
    """
    return np.ceil(f / 2.) * 2


def visualize_attention_pooling(atoms, weights):
    """
    Plot attention pooling weights
    :param atoms: Atom coordinates and types for a single example
    :param weights: Attention weights for each atom
    """

    # Transform weights into a range suitable for plotting sizes
    weights = np.expand_dims(weights, axis=1)
    weights = weights - np.min(weights)
    weights = weights*(50.0 / (np.max(weights) - np.min(weights))) + 1

    # Concatenate with atom array
    data = np.concatenate((atoms, weights), 1)

    # Plot parameters
    figsize = 250
    dpi = 100
    trimfrac = 0.4
    figsizeextra = round_up_to_even(figsize * (1 + trimfrac))
    figsizeextra_inch = figsizeextra / dpi

    # Plotting
    plt.figure(figsize=(figsizeextra_inch, figsizeextra_inch), dpi=100)

    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], s=data[i, 3], color='lightgreen')

    sel = data[:, 2] == 1
    x, y = data[sel, 0], data[sel, 1]
    plt.scatter(x, y, s=4, color='orange')
    sel = data[:, 2] == 2
    x, y = data[sel, 0], data[sel, 1]
    plt.scatter(x, y, s=4, color='blue')

    plt.axis('equal')

    plt.savefig('test.png')

    plt.clf()
    plt.close()

    # add unique label for each graph and include that in figure label


