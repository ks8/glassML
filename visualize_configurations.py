# Functions for visualizing configurations
# Import modules
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def round_up_to_even(f):
    """
    Round up to the nearest even number
    :param f: Input number
    :return: Nearest even number greater than f
    """
    return np.ceil(f / 2.) * 2


def visualize_configurations(args):
    """
    Plot attention pooling weights
    :param args: Args
    """

    # Data
    data = np.loadtxt(args.data)
    print(data)

    # Plot parameters
    figsize = 250
    dpi = 100
    trimfrac = 0.4
    figsizeextra = round_up_to_even(figsize * (1 + trimfrac))
    figsizeextra_inch = figsizeextra / dpi

    # Plotting
    plt.figure(figsize=(figsizeextra_inch, figsizeextra_inch), dpi=100)

    sel = data[:, 0] == 1
    x, y = data[sel, 1], data[sel, 2]
    plt.scatter(x, y, s=4, color='orange')
    sel = data[:, 0] == 2
    x, y = data[sel, 1], data[sel, 2]
    plt.scatter(x, y, s=4, color='blue')

    plt.axis('equal')
    plt.title(args.plot_title)

    plt.savefig(args.output)


def main():
    """
    Parse arguments and execute xyz file processing.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, dest='data', default=None, help='Config file')
    parser.add_argument('--plot_title', type=str, dest='plot_title', default='Atomic Coordinates', help='Plot title')
    parser.add_argument('--output', type=str, dest='output', default=None, help='File name for output files')

    args = parser.parse_args()
    visualize_configurations(args)


if __name__ == "__main__":
    main()
