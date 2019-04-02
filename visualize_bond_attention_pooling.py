# Functions for visualizing bond attention
# Import modules
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def round_up_to_even(f):
    """
    Round up to the nearest even number
    :param f: Input number
    :return: Nearest even number greater than f
    """
    return np.ceil(f / 2.) * 2


def visualize_bond_attention_pooling(atoms, bonds, weights, id_number, label, viz_dir):
    """
    Plot attention weights.
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

    # Graph analysis
    graph = nx.Graph()  # Construct an empty graph for high attention weights
    graph_lo = nx.Graph()  # Construct an empty graph for low attention weights
    num_type_2_connections = 0  # Initialize number of type 2 bonds counter
    num_type_1_isolated = 0  # Initialize num type 1 isolated
    num_type_2_isolated = 0  # Initialize num type 2 isolated

    for i in range(len(bonds)):
        atom_points_x = np.array([atoms[bonds[i + 1][0] - 1, 0], atoms[bonds[i + 1][1] - 1, 0]])
        atom_points_y = np.array([atoms[bonds[i + 1][0] - 1, 1], atoms[bonds[i + 1][1] - 1, 1]])
        plt.plot(atom_points_x, atom_points_y, linewidth=weights[i][0], color='lightgreen', zorder=0)

        # Analysis
        if weights[i][0] > 0.11:  # Only consider high attention edges
            graph.add_nodes_from([bonds[i + 1][0], bonds[i + 1][1]])  # Record the nodes for this edge
            graph.add_edge(bonds[i + 1][0], bonds[i + 1][1])  # Record the edge
            if atoms[bonds[i + 1][0] - 1, 2] == 2.0 and atoms[bonds[i + 1][1] - 1, 2] == 2.0:  # Num type 2 bonds
                num_type_2_connections += 1

    for i in range(len(bonds)):
        if weights[i][0] < 0.11:
            for j in range(2):
                if bonds[i + 1][j] not in graph:
                    if bonds[i + 1][j] not in graph_lo:
                        if atoms[bonds[i + 1][j] - 1, 2] == 1.0:
                            num_type_1_isolated += 1
                        else:
                            num_type_2_isolated += 1
                        graph_lo.add_node(bonds[i + 1][j])

    sel = atoms[:, 2] == 1
    x, y = atoms[sel, 0], atoms[sel, 1]
    plt.scatter(x, y, s=6, color='orange', zorder=1, label='type 1')
    sel = atoms[:, 2] == 2
    x, y = atoms[sel, 0], atoms[sel, 1]
    plt.scatter(x, y, s=6, color='blue', zorder=1, label='type 2')

    plt.axis('equal')

    plt.savefig(viz_dir + '/' + str(label) + '-' + str(id_number) + '.png')

    plt.clf()
    plt.close()

    # Return label, number connected high attention components, number type 2 bonds, num type 1 isolated, num type 2 isolated
    return label, nx.number_connected_components(graph), num_type_2_connections, num_type_1_isolated, num_type_2_isolated





