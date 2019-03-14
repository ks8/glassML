# Functions for analyzing attention pooling
# Import modules
import numpy as np


def analyze_attention_pooling(atoms, weights, id_number, label, viz_dir):
    """
    Plot attention pooling weights
    :param atoms: Atom coordinates and types for a single example
    :param weights: Attention weights for each atom
    :param id_number: uid number for this graph
    :param label: Label value for this graph
    :param viz_dir: Directory for saving analysis files to
    """

    # Transform weights into a range suitable for plotting sizes
    weights_transform = np.expand_dims(weights, axis=1)
    weights_transform = weights_transform - np.min(weights_transform)
    weights_transform = weights_transform*(200.0 / (np.max(weights_transform) - np.min(weights_transform))) + 4

    # Concatenate with atom array
    data = np.concatenate((atoms, weights_transform), 1)

    weights_sort = np.sort(weights)
    num_top_weights = weights_sort[np.argmax(np.ediff1d(weights_sort)) + 1:].shape[0]
    data_top = data[np.argsort(weights)][data.shape[0] - num_top_weights:, :]

    print(data)
    print(np.std(data[:, 3]))
    print(weights_sort)
    print(num_top_weights)
    print(data_top)
    exit()


    # plt.savefig(viz_dir + '/' + str(label) + '-' + str(id_number) + '.png')




# Identify biggest difference in successive data elements to identify the "top" attention weights
# Count the number of top, the proportion of type 1 to type 2 in the  top, and the geometric spread of the top.





