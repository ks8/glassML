# Functions for visualizing bond attention pooling
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


def visualize_bond_attention_pooling(atoms, bonds, weights, id_number, label, viz_dir):
    """
    Plot attention pooling weights
    :param atoms: Atom coordinates and types for a single example
    :param bonds: Bond info for a single example
    :param weights: Attention weights for each atom
    :param id_number: uid number for this graph
    :param label: Label value for this graph
    :param viz_dir: Directory for saving images to
    """

    # Transform weights into a range suitable for plotting sizes
    weights = np.expand_dims(weights, axis=1)
    weights = weights - np.min(weights)
    weights = 4.0*(weights / (np.max(weights))) + 0.1

    # Plot parameters
    figsize = 250
    dpi = 100
    trimfrac = 0.4
    figsizeextra = round_up_to_even(figsize * (1 + trimfrac))
    figsizeextra_inch = figsizeextra / dpi

    # Plotting
    plt.figure(figsize=(figsizeextra_inch, figsizeextra_inch), dpi=100)

    for i in range(len(bonds)):
        atom_points_x = np.array([atoms[bonds[i + 1][0] - 1, 0], atoms[bonds[i + 1][1] - 1, 0]])
        atom_points_y = np.array([atoms[bonds[i + 1][0] - 1, 1], atoms[bonds[i + 1][1] - 1, 1]])
        plt.plot(atom_points_x, atom_points_y, linewidth=weights[i][0], color='lightgreen', zorder=0)

    sel = atoms[:, 2] == 1
    x, y = atoms[sel, 0], atoms[sel, 1]
    plt.scatter(x, y, s=4, color='orange', zorder=1)
    sel = atoms[:, 2] == 2
    x, y = atoms[sel, 0], atoms[sel, 1]
    plt.scatter(x, y, s=4, color='blue', zorder=1)

    plt.axis('equal')

    plt.savefig(viz_dir + '/' + str(label) + '-' + str(id_number) + '.png')

    plt.clf()
    plt.close()





